"""Выкачивает сиды ТЗ с Hugging Face для reverse-construction генерации.

Что и зачем берём (обоснование — в плане, раздел «Сиды разнообразия»):

* Flame-Code-VLM/Flame-Evo-React (157 903, Apache-2.0) — главный источник. Пара
  `task_description` + `layout_description` это готовое ТЗ на фронтенд, причём именно
  на React, то есть на нашей целевой технологии. Берём ТОЛЬКО текстовые колонки:
  колонка `image` в этом паркете весит основную часть от 2.55 ГБ, а нам нужны ТЗ, а не
  их картинки — свои мы отрендерим сами.
* Tesslate/UIGEN-T3-Dataset-Extended-Reasoning (7 957, Apache-2.0) — колонка `Question`
  даёт другой РЕГИСТР формулировок (короткий пользовательский запрос вместо развёрнутой
  спецификации) и другие темы: чат-боты, тепловые карты, онбординг, биллинг.
* darknoon/tailwind-edits (509, Apache-2.0) — before/after правки Tailwind. Здесь не ТЗ,
  а библиотека РЕАЛЬНЫХ операций редактирования для задачи editing (этап 3), чтобы не
  выдумывать список правок из головы.

Паркет колоночный, поэтому `columns=[...]` реально экономит трафик: лишние колонки
не читаются вообще, а не отбрасываются после загрузки.

Использование:
    .venv/bin/python Data/generators/synth/fetch_seeds.py --out Data/synth_pilot/seeds
"""

import argparse
import json
import sys
from pathlib import Path

import fsspec
import pyarrow.parquet as pq

# Пути внутри HF-фс: datasets/<repo>/<файл>. Колонки перечислены явно —
# см. докстринг про экономию на колонке image.
SOURCES = {
    "flame_evo_react": {
        "path": "datasets/Flame-Code-VLM/Flame-Evo-React/Flame-Evo-React.parquet",
        "columns": ["id", "task_description", "layout_description", "variation_round"],
        "license": "apache-2.0",
    },
    "uigen_t3": {
        "path": "datasets/Tesslate/UIGEN-T3-Dataset-Extended-Reasoning/responses.parquet",
        "columns": ["id", "Question"],
        "license": "apache-2.0",
    },
    "tailwind_edits": {
        "path": "datasets/darknoon/tailwind-edits/data/train-00000-of-00001.parquet",
        "columns": ["input", "edits", "output"],
        "license": "apache-2.0",
    },
}


def fetch_one(name: str, spec: dict, out_dir: Path) -> int:
    out = out_dir / f"{name}.jsonl"
    if out.exists() and out.stat().st_size > 0:
        n = sum(1 for _ in out.open(encoding="utf-8"))
        print(f"  = {name}: уже скачан, {n} строк ({out})")
        return n

    fs = fsspec.filesystem("hf")
    print(f"  ↓ {name}: {spec['path']}")
    with fs.open(spec["path"]) as fh:
        table = pq.ParquetFile(fh).read(columns=spec["columns"])

    n = table.num_rows
    tmp = out.with_suffix(".jsonl.part")
    with tmp.open("w", encoding="utf-8") as w:
        for row in table.to_pylist():
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.rename(out)
    print(f"    {n} строк, {out.stat().st_size / 1e6:.1f} МБ -> {out}")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="Data/synth_pilot/seeds", help="куда класть jsonl")
    ap.add_argument("--only", nargs="*", choices=sorted(SOURCES), help="подмножество источников")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = args.only or list(SOURCES)
    counts = {}
    for name in names:
        try:
            counts[name] = fetch_one(name, SOURCES[name], out_dir)
        except Exception as e:
            print(f"  ! {name}: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
            counts[name] = 0

    # Лицензии фиксируем рядом с данными: все три источника Apache-2.0, производные
    # использовать можно. biglab/webui-* сюда СОЗНАТЕЛЬНО не входит — там лицензия
    # `other` с ограничениями по копирайту, его можно смотреть, но нельзя тянуть
    # в таргеты, а смешивать источники с разными правами в одном файле — путь к беде.
    (out_dir / "SOURCES.json").write_text(
        json.dumps(
            {n: {"hf_path": SOURCES[n]["path"], "license": SOURCES[n]["license"],
                 "rows": counts.get(n, 0)} for n in names},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    print("\nИтого:", ", ".join(f"{k}={v}" for k, v in counts.items()))
    return 0 if all(counts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
