#!/usr/bin/env python3
"""export_htmls.py — выгрузить исходные HTML готового набора в html-кэш конвертера.

Нужно для честного A/B «чистый против сырого». Сырой набор собирается из пары
корпуса как есть и теряет страницы, чей исходный скриншот не 1280 пикселей
шириной (в 15k таких 11%). Если оставить всё как есть, наборы будут отличаться
не только обработкой, но и составом страниц — а тогда сравнение обучений
ничего не доказывает.

Здесь мы берём `target_html` СЫРОГО набора (это и есть исходный HTML корпуса,
он не подвергался чистке) и кладём в формат html-кэша `convert_parallel.py`.
Дальше чистый конвертер запускается с этим кэшем и рендерит ровно те же
страницы — скачивать источник заново не нужно.

    python export_htmls.py <путь_к_сырому_набору> <кэш.jsonl.gz>
"""
import argparse
import gzip
import json

from datasets import load_from_disk


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset", help="набор, собранный convert_raw.py (load_from_disk)")
    ap.add_argument("out", help="куда писать html-кэш (.jsonl.gz)")
    args = ap.parse_args()

    ds = load_from_disk(args.dataset)
    if hasattr(ds, "keys"):
        raise SystemExit(f"нужен НЕразрезанный набор, а это DatasetDict ({list(ds)})")

    n = 0
    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        # только нужная колонка: иначе тянем с сетевого диска все картинки
        for s in ds.select_columns(["target_html"]):
            f.write(json.dumps({"html": s["target_html"]}, ensure_ascii=False) + "\n")
            n += 1
    print(f"[export] выгружено {n} страниц -> {args.out}")


if __name__ == "__main__":
    main()
