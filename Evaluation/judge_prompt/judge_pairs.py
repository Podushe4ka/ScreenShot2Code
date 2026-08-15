"""
judge_pairs.py — win-rate судьи на НЕразмеченных тройках.

Зачем отдельно от run_judge_eval.py. Тот считает согласие судьи с человеком и
поэтому ТРЕБУЕТ labels.csv с колонкой `winner`. Здесь задача другая: есть пары
генераций двух моделей на наборе, где человеческой разметки нет и не будет
(484 сэмпла Design2Code), и нужен ответ «кого судья предпочитает и как часто».

Вход — каталог того же вида, что читает run_judge_eval.py:

    <data-root>/batch_XXXXX/sample_XXXXX/{ref.png, pred_baseline.png, pred_checkpoint.png}

Порядок показа (кто идёт «Candidate A») берётся из sample_order.swapped_for_sample,
то есть тем же детерминированным хэшем, что и при разметке — чтобы анти-позиционная
рандомизация была той же, а не новой.

⚠ Сырой win-rate СМЕЩЁН. На размеченной паре судья (Qwen3.5-9B, v1_baseline) давал
P(скажет checkpoint | человек сказал baseline) = 38.4% и P(checkpoint | checkpoint)
= 77.4%. Отсюда наблюдаемый win-rate связан с истинным как
    win_судьи = 0.384 + 0.390 * win_человека,
то есть при РАВНЫХ моделях судья покажет 57.9% в пользу checkpoint. Скрипт печатает
и сырой, и калиброванный по этой формуле win-rate, но калибровка верна лишь при
допущении, что поклассовые ошибки судьи переносятся на новую пару моделей.

Флаг --invert-order переворачивает порядок показа на всех сэмплах. Прогон с ним и
без него по одному набору даёт долю переворотов ответа — прямую меру позиционной
неустойчивости судьи, которой иначе нет.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from judge_client import JudgeError, judge_one
from prompts import get_prompt
from sample_order import swapped_for_sample
from vllm_client import VLLMClient

# Коэффициенты из confusion matrix Qwen3.5-9B / v1_baseline на labels_train.csv
# (69/43 и 37/127). Пересчитать при смене модели или промпта судьи.
CALIB_INTERCEPT = 0.384
CALIB_SLOPE = 0.390


def iter_sample_dirs(data_root: Path):
    """Отдаёт (sample_id, dir) для всех троек. sample_id — путь относительно
    data-root, тот же вид, что в labels.csv («batch_00000/sample_00042»), чтобы
    swapped_for_sample давал тот же порядок показа."""
    for batch_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for sample_dir in sorted(p for p in batch_dir.iterdir() if p.is_dir()):
            needed = ("ref.png", "pred_baseline.png", "pred_checkpoint.png")
            if all((sample_dir / f).exists() for f in needed):
                yield f"{batch_dir.name}/{sample_dir.name}", sample_dir


def main():
    ap = argparse.ArgumentParser(description="Win-rate судьи на неразмеченных тройках")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--prompt-key", required=True)
    ap.add_argument("--judge-url", default="http://127.0.0.1:8001")
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--invert-order", action="store_true",
                    help="Перевернуть порядок показа на всех сэмплах (проба на позиционный биас).")
    args = ap.parse_args()

    prompt_text = get_prompt(args.prompt_key)
    samples = list(iter_sample_dirs(args.data_root))
    print(f"[judge_pairs] Найдено троек: {len(samples)}")

    client = VLLMClient(args.judge_url, args.judge_model)
    print(f"[judge_pairs] Жду сервер {args.judge_url} ...")
    client.wait_until_ready()

    wins = Counter()
    n_error = 0
    per_sample = []

    for i, (sample_id, sample_dir) in enumerate(samples, start=1):
        swapped = swapped_for_sample(sample_id)
        if args.invert_order:
            swapped = not swapped
        try:
            res = judge_one(client, prompt_text, sample_dir, swapped,
                            max_tokens=args.max_tokens)
            wins[res["winner_model"]] += 1
            per_sample.append({"sample_id": sample_id, "swapped": swapped,
                               "winner_model": res["winner_model"],
                               "raw_winner": res["raw_winner"]})
        except JudgeError as e:
            n_error += 1
            print(f"[judge_pairs] {sample_id}: ошибка судьи — {e}")
            per_sample.append({"sample_id": sample_id, "swapped": swapped,
                               "winner_model": None, "error": str(e)})
        if i % 50 == 0 or i == len(samples):
            print(f"[judge_pairs] {i}/{len(samples)}")

    n_scored = len(samples) - n_error
    raw = wins["checkpoint"] / n_scored if n_scored else 0.0
    calibrated = (raw - CALIB_INTERCEPT) / CALIB_SLOPE if n_scored else 0.0

    print()
    print("=" * 64)
    print(f"Судья           : {args.judge_model}")
    print(f"Промпт          : {args.prompt_key}")
    print(f"Порядок показа  : {'ИНВЕРТИРОВАН' if args.invert_order else 'штатный'}")
    print(f"Троек           : {len(samples)}  (оценено {n_scored}, ошибок {n_error})")
    print(f"checkpoint      : {wins['checkpoint']}")
    print(f"baseline        : {wins['baseline']}")
    print(f"Сырой win-rate чекпоинта     : {raw:.1%}")
    print(f"Калиброванный (см. docstring): {calibrated:.1%}")
    print("⚠ При РАВНЫХ моделях сырой win-rate ожидается 57.9%, а не 50%.")
    print("=" * 64)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({
            "judge_model": args.judge_model,
            "prompt_key": args.prompt_key,
            "invert_order": args.invert_order,
            "data_root": str(args.data_root),
            "n_total": len(samples),
            "n_scored": n_scored,
            "n_error": n_error,
            "wins": dict(wins),
            "winrate_checkpoint_raw": raw,
            "winrate_checkpoint_calibrated": calibrated,
            "calibration": {"intercept": CALIB_INTERCEPT, "slope": CALIB_SLOPE},
            "per_sample": per_sample,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[judge_pairs] Отчёт сохранён в {args.out_json}")


if __name__ == "__main__":
    main()
