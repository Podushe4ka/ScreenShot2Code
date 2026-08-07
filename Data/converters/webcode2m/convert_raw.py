#!/usr/bin/env python3
"""convert_raw.py (WebCode2M) — СЫРОЙ вариант датасета: пара из корпуса как есть.

Контрольный набор к обычному `convert_parallel.py`. Отличие ровно одно и оно
намеренное: **чистки нет**. `<img>` остаются на месте, фоновые картинки тоже,
data-URI не режутся, а скриншот берётся **готовый из корпуса** — тот, где
картинки настоящие, а не серые блоки.

Почему скриншот из корпуса, а не свой рендер: наш рендер оффлайновый, внешние
URL он не подтянет, и вместо фотографий вышли бы битые иконки — то есть хуже
серых блоков, и гипотеза «дать всё как есть» осталась бы непроверенной.
Побочный выигрыш — рендерить нечего, фаза 2 отпадает целиком.

Зачем набор нужен: на бенче Design2Code все картинки заменены на ОДНУ реальную
фотографию (`rick.jpg`), а в чистом датасете на их месте плоские серые
прямоугольники. Учим на серых блоках, меряем на фотографиях. Этот набор
позволяет замерить, чего стоит это расхождение.

Метрику менять не надо: харнесс прогоняет `replace_images_with_placeholder`
и по предсказанию, и по эталону, так что сырые `<img>` сравниваются честно.

    python convert_raw.py --target 15000 --same-as <clean_htmls.jsonl.gz> --out ...

`--same-as` берёт список HTML чистого прогона и оставляет ровно те же страницы
(сверка по SHA1). Без него это был бы не A/B, а два разных набора страниц.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
from collections import Counter

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk

from convert_lib import DATASET_ID, FEATURES, HTML_FIELD, IMAGE_FIELD, RENDER_WIDTH, SPLIT

# см. комментарий в convert_parallel.py: 32-битные оффсеты Arrow, предел 2 ГБ на массив
CHUNK_ROWS = 500


def load_same_as(path):
    """SHA1 страниц чистого прогона — чтобы наборы отличались ТОЛЬКО обработкой."""
    if not path:
        return None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        keys = {hashlib.sha1(json.loads(line)["html"].encode("utf-8")).hexdigest() for line in f}
    print(f"[фаза 1] сверяюсь с чистым прогоном: {len(keys)} страниц из {path}")
    return keys


def main():
    ap = argparse.ArgumentParser(description="WebCode2M -> контракт БЕЗ чистки (контрольный набор).")
    ap.add_argument("--target", type=int, default=15000)
    ap.add_argument("--max-scan", type=int, default=200000)
    ap.add_argument("--out", default="./webcode2m_raw")
    ap.add_argument("--same-as", default=None,
                    help="html-кэш чистого прогона: взять ровно те же страницы")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    want = load_same_as(args.same_as)
    print(f"источник: {DATASET_ID} | target: {args.target} | out: {out_dir} | ЧИСТКИ НЕТ")

    parts_dir = out_dir + "_parts"
    shutil.rmtree(parts_dir, ignore_errors=True)
    os.makedirs(parts_dir, exist_ok=True)
    buf, parts, sizes = [], [], []
    skipped = Counter()

    def flush():
        if not buf:
            return
        path = os.path.join(parts_dir, f"part-{len(parts):05d}")
        Dataset.from_list(buf, features=FEATURES).save_to_disk(path)
        parts.append(path)
        buf.clear()

    stream = load_dataset(DATASET_ID, split=SPLIT, streaming=True)
    seen, scanned, kept = set(), 0, 0
    from tqdm import tqdm
    bar = tqdm(total=args.target, desc="[сырой] сбор")
    for r in stream:
        if kept >= args.target or scanned >= args.max_scan:
            break
        scanned += 1
        html = (r.get(HTML_FIELD) or "").strip()
        img = r.get(IMAGE_FIELD)
        if not html or img is None:
            skipped["пустые"] += 1
            continue
        h = hashlib.sha1(html.encode("utf-8")).hexdigest()
        if h in seen:
            skipped["дубли"] += 1
            continue
        if want is not None and h not in want:
            skipped["нет в чистом наборе"] += 1
            continue
        # Ширина — единственное, что приводим к конвенции: пиксель-бюджет обучения
        # и бенча считается от 1280, страница другой ширины меряется не тем масштабом.
        if img.width != RENDER_WIDTH:
            skipped[f"ширина != {RENDER_WIDTH}"] += 1
            continue
        seen.add(h)
        rgb = img.convert("RGB")
        b = io.BytesIO()
        rgb.save(b, "PNG")
        sizes.append(rgb.size)
        buf.append({"task_type": "drafting", "images": [{"bytes": b.getvalue(), "path": None}],
                    "current_html": "", "target_html": html, "instruction": ""})
        kept += 1
        bar.update(1)
        if len(buf) >= CHUNK_ROWS:
            flush()
    bar.close()
    flush()
    print(f"[сырой] собрано: {kept} (просмотрено {scanned}) | пропущено: {dict(skipped)}")
    if not parts:
        raise SystemExit("не собрано ни одного сэмпла")

    ds = concatenate_datasets([load_from_disk(p) for p in parts])
    ds.save_to_disk(out_dir, max_shard_size="500MB")
    del ds
    shutil.rmtree(parts_dir, ignore_errors=True)

    ds2 = load_from_disk(out_dir)
    assert len(ds2) == len(sizes), f"рассинхрон: ds2={len(ds2)} sizes={len(sizes)}"
    widths = {w for w, _ in sizes}
    assert widths == {RENDER_WIDTH}, f"ширины разные: {widths}"
    # Проверка НАОБОРОТ к чистому конвертеру: там `<img>` быть не должно, здесь
    # он обязан сохраниться — иначе набор перестаёт быть контрольным.
    n_with_img = sum(1 for s in ds2.select_columns(["target_html"])
                     if "<img" in s["target_html"].lower())
    heights = sorted(h for _, h in sizes)
    print(f"[приёмка] OK: {len(ds2)} сэмплов, ширина {RENDER_WIDTH}, load_from_disk ✓")
    print(f"[приёмка] страниц с <img>: {n_with_img} из {len(ds2)} "
          f"({100 * n_with_img / len(ds2):.1f}%) — в чистом наборе их 0")
    print(f"[приёмка] высота: min={heights[0]}, median={heights[len(heights)//2]}, max={heights[-1]}")
    print(f"\n=== ПЕРЕДАЧА ===\nпуть: {out_dir}\nзагрузка: load_from_disk(<путь>)")


if __name__ == "__main__":
    main()
