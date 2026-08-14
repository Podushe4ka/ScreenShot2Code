#!/usr/bin/env python3
"""build_mix.py — собрать смешанный drafting-набор из нескольких источников.

Каждый сэмпл несёт колонку `source` — это и есть смысл солянки: собрать один набор так,
чтобы потом можно было отабляции́ровать вклад каждого источника, не пересобирая данные.

Вход — части вида `МЕТКА=путь`, где путь это:
  * `features.jsonl` / `selected.jsonl` от `../complexity/` (в строках `id`, `html_path`, `png`);
  * либо staging-каталог конвертера (`manifest.jsonl` + `pages/`) — тогда берутся все `ok`.

Выход — `Dataset` по контракту `SFT/DATA_FORMAT_CONTRACT.md` (§2) плюс служебные колонки
`source` и `page_id`. Контракт лишние колонки допускает: SFT-загрузчик читает только
`task_type`/`images`/`current_html`/`target_html`/`instruction`, а `source` нужен для
абляции, `page_id` — для группового разреза.

    python build_mix.py --part "webui_inline=sel_webui.jsonl" \
                        --part "webcode2m_complex=sel_wc2m.jsonl" \
                        --part "synth=Data/synth_pilot" \
                        --out /path/mix_15k
"""
import argparse
import json
import os
import sys

from datasets import Dataset, Features, Image, Sequence, Value, concatenate_datasets

# Схема контракта §2 + две служебные колонки.
FEATURES = Features({
    "task_type":    Value("string"),
    "images":       Sequence(Image()),
    "current_html": Value("string"),
    "target_html":  Value("string"),
    "instruction":  Value("string"),
    "source":       Value("string"),
    "page_id":      Value("string"),
})

# Сколько сэмплов писать одним куском. Все картинки куска лежат в ОДНОМ бинарном массиве
# Arrow, а у него 32-битные оффсеты — предел 2 ГБ на массив. 15k сэмплов это ~7.5 ГБ, и
# `Dataset.from_list` на всём наборе сразу падает с «offset overflow while concatenating
# arrays». Значение и обоснование — из `../webcode2m/convert_parallel.py`.
CHUNK_ROWS = 500


def _find_manifest(path):
    """`manifest.jsonl` в каталоге или в его `build/` — синтетика держит его именно там
    (`Data/synth_pilot/build/manifest.jsonl`), а конвертеры кладут в корень staging'а."""
    for rel in ("manifest.jsonl", os.path.join("build", "manifest.jsonl")):
        p = os.path.join(path, rel)
        if os.path.exists(p):
            return p
    raise SystemExit(f"не нашёл manifest.jsonl ни в {path}, ни в {path}/build")


def _resolve(base, rel, sid, ext, candidates):
    """Путь к файлу сэмпла. Сначала то, что записано в манифесте, потом обычные раскладки:
    staging держит страницы в `pages/`, синтетика — исходник в `raw/`, скриншот в `build/`."""
    if rel:
        p = os.path.join(base, rel)
        if os.path.exists(p):
            return p
    for sub in candidates:
        p = os.path.normpath(os.path.join(base, sub, f"{sid}{ext}"))
        if os.path.exists(p):
            return p
    return None


def load_part(path):
    """Часть солянки -> [{id, html_path, png_path}]. Принимает и jsonl отбора, и staging."""
    if os.path.isdir(path):
        manifest = _find_manifest(path)
        base = os.path.dirname(manifest)
        out = []
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("status") != "ok":
                    continue
                sid = rec.get("sample_id") or rec.get("id")
                html_path = _resolve(base, rec.get("html"), sid, ".html",
                                     ("", "pages", "raw", "../raw"))
                png_path = _resolve(base, rec.get("png"), sid, ".png",
                                    ("", "pages", "build", "../build"))
                if not html_path or not png_path:
                    continue
                out.append({"id": sid, "html_path": html_path, "png_path": png_path})
        return out

    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            png = rec.get("png") or rec.get("png_path")
            html = rec.get("html_path")
            if not html or not png:
                raise SystemExit(f"в строке отбора нет html_path/png: {list(rec)[:8]}")
            out.append({"id": rec["id"], "html_path": html, "png_path": png})
    return out


def rows_for(items, source, instruction):
    rows = []
    missing = 0
    for it in items:
        try:
            with open(it["html_path"], encoding="utf-8") as f:
                target = f.read()
            with open(it["png_path"], "rb") as f:
                png = f.read()
        except OSError:
            missing += 1
            continue
        rows.append({"task_type": "drafting", "images": [png],
                     "current_html": "", "target_html": target,
                     "instruction": instruction, "source": source,
                     "page_id": f"{source}:{it['id']}"})
    return rows, missing


def build(rows):
    """Собрать Dataset кусками по CHUNK_ROWS — см. комментарий к константе."""
    parts = []
    for i in range(0, len(rows), CHUNK_ROWS):
        parts.append(Dataset.from_list(rows[i:i + CHUNK_ROWS], features=FEATURES))
    return concatenate_datasets(parts) if len(parts) > 1 else parts[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part", action="append", required=True,
                    help="МЕТКА=путь (jsonl отбора или staging-каталог); можно повторять")
    ap.add_argument("--out", required=True)
    ap.add_argument("--instruction", default="",
                    help="поле instruction контракта; для drafting промпт задаёт SFT-трек")
    ap.add_argument("--shuffle-seed", type=int, default=42)
    args = ap.parse_args()

    all_rows, summary = [], []
    for spec in args.part:
        if "=" not in spec:
            raise SystemExit(f"часть задаётся как МЕТКА=путь, получено: {spec!r}")
        label, path = spec.split("=", 1)
        items = load_part(path)
        rows, missing = rows_for(items, label, args.instruction)
        all_rows.extend(rows)
        summary.append((label, len(items), len(rows), missing))
        print(f"[солянка] {label}: {len(rows)} сэмплов"
              f"{f' (не нашлось файлов: {missing})' if missing else ''}")

    if not all_rows:
        raise SystemExit("ни одна часть не дала сэмплов")

    ds = build(all_rows).shuffle(seed=args.shuffle_seed)
    ds.save_to_disk(args.out)

    print(f"\n=== СОЛЯНКА: {len(ds)} сэмплов -> {args.out} ===")
    print(f"  {'источник':24s} {'кандидатов':>11s} {'вошло':>8s} {'нет файлов':>11s}")
    for label, n_items, n_rows, missing in summary:
        print(f"  {label:24s} {n_items:11d} {n_rows:8d} {missing:11d}")

    from collections import Counter
    print(f"  проверка колонки source: {Counter(ds['source'])}")
    print(f"\nДальше — разрез: python ../make_split.py {args.out} {args.out}_split")


if __name__ == "__main__":
    main()
