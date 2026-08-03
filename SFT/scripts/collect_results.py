#!/usr/bin/env python3
"""Сводная таблица по каталогу пилота: одна строка = один эксперимент.

    python scripts/collect_results.py /path/to/exps-20260803-201500

Читает `<EID>-bench/summary.json` (метрики бенча) и `<EID>/clearml_task.json`
(ссылка на ран обучения). Нужен, чтобы утром смотреть одну таблицу, а не
пять json-ов вразнобой; в ClearML то же самое лежит под тегом pilot1k.

Базы Design2Code, которые надо побить (Evaluation/results/):
Qwen-3.5-4B 0.787, Qwen-2.5-7B 0.816, Qwen-3.5-9B 0.852.
"""

import json
import sys
from pathlib import Path

BASELINE = 0.787  # Qwen-3.5-4B, та же модель что дообучаем
COLS = ["final_score", "block_match", "text", "position", "color", "clip"]


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    rows = []
    for bench_dir in sorted(root.glob("*-bench")):
        eid = bench_dir.name[: -len("-bench")]
        summary = read_json(bench_dir / "summary.json")
        if summary is None:
            rows.append((eid, None, None, None))
            continue
        link = read_json(root / eid / "clearml_task.json") or {}
        rows.append((eid, summary, link.get("train_task_id"), summary.get("n_length_truncated")))

    if not rows:
        print("нет результатов: не найдено ни одного *-bench/summary.json")
        return

    head = f"{'эксп':6} " + " ".join(f"{c[:9]:>9}" for c in COLS) + f" {'дельта':>8} {'обрез':>6}"
    print(head)
    print("-" * len(head))

    done = []
    for eid, summary, task_id, truncated in rows:
        if summary is None:
            print(f"{eid:6} {'— бенч не дошёл —':>40}")
            continue
        cells = " ".join(
            f"{summary[c]:9.4f}" if isinstance(summary.get(c), (int, float)) else f"{'—':>9}"
            for c in COLS
        )
        fs = summary.get("final_score")
        delta = f"{fs - BASELINE:+8.4f}" if isinstance(fs, (int, float)) else f"{'—':>8}"
        print(f"{eid:6} {cells} {delta} {str(truncated or 0):>6}")
        if isinstance(fs, (int, float)):
            done.append((fs, eid, task_id))

    print(f"\nбаза Qwen-3.5-4B на Design2Code: {BASELINE}")
    if done:
        best_score, best_eid, best_task = max(done)
        verdict = "ЛУЧШЕ базы" if best_score > BASELINE else "базу не побил"
        print(f"лучший: {best_eid} — final_score {best_score:.4f} ({verdict})")
        if best_task:
            print(f"  ран обучения в ClearML: {best_task}")
    trunc = [eid for eid, summary, _, t in rows if summary and (t or 0)]
    if trunc:
        print(f"ВНИМАНИЕ: обрезаны по токенам ({', '.join(trunc)}) — метрики "
              "занижены, подними --max-new-tokens")


if __name__ == "__main__":
    main()
