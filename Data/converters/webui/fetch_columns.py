#!/usr/bin/env python3
"""fetch_columns.py (WebUI) — фаза 0: вытащить из `ronantakizawa/webui` ТОЛЬКО текстовые
колонки, без картинок, и сложить дедуплицированный по `sample_id` кэш кандидатов.

ЗАЧЕМ ОТДЕЛЬНАЯ ФАЗА, А НЕ `load_dataset`. Полный train-сплит — 4.27 ГБ в 12 parquet-шардах,
и 93% этого веса — колонка `image` (72.4 МБ из 78 МБ в замеренной row-group). Картинка WebUI
нам не нужна вообще: она снята с вьюпорта 1280×720, то есть только первый экран, а `html`
содержит страницу целиком — пара «этот скриншот ↔ этот код» была бы рассогласована, и мы
рендерим свой скриншот сами (см. `convert_lib.render_page`, шаг 5 плана). Поэтому читаем
parquet через pyarrow с проекцией колонок: parquet колоночный, и по HTTP-диапазонам
скачиваются только байты нужных колонок — ~320 МБ вместо 4.27 ГБ.

ДЕДУП ПО `sample_id`. В источнике 29 409 строк = 9 803 страницы × 3 вьюпорта
(desktop/tablet/mobile). У всех трёх строк один и тот же `html`/`css` — различаются картинка
и bbox'ы. Три вьюпорта в обучающий набор не идут: это один и тот же ответ на три разных
входа, и модель получила бы тройной вес одной страницы плюс противоречивый сигнал
«одинаковый код для разных макетов». Берём строку `viewport == "desktop"` — наш ре-рендер
тоже идёт при ширине 1280.

СПЛИТ. Берём только `train` (29 409 строк). Валидация и тест WebUI разделены ПО ИСТОЧНИКАМ
(wordpress_themes/awwwards/adobe-spectrum/chakra-ui-storybook и onepagelove/frontend_mentor/
carbon/glitch) — если когда-нибудь мерить на них, они должны остаться вне обучения.

    python fetch_columns.py --out /path/webui_desktop.jsonl.gz
"""
import argparse
import gzip
import json
import os
import time

# Колонки, которые реально нужны конвертеру и скорингу. `image` и `bboxes` НЕ берём:
# картинку рендерим свою, а сложность считаем по СВОЕМУ рендеру (bbox'ы источника сняты
# при 1280×720 и обрезаны первым экраном — по ним нельзя посчитать колонки/перекрытия
# для полной страницы).
COLUMNS = ["sample_id", "html", "css", "js", "viewport", "source_name", "source_url",
           "framework", "css_framework", "component_type", "element_count", "has_animations"]

REPO = "datasets/ronantakizawa/webui"
N_SHARDS = 12
SHARD_FMT = "data/train-{i:05d}-of-{n:05d}.parquet"


def iter_rows(shard_paths, columns):
    """Строки нужных колонок из parquet поверх HfFileSystem, по row-group'ам.

    Читаем по одной row-group, а не файл целиком: pyarrow тянет по HTTP только те
    байтовые диапазоны, где лежат запрошенные колонки, и пиковая память остаётся
    в пределах одной группы.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    for path in shard_paths:
        with fs.open(path, "rb") as fh:
            pf = pq.ParquetFile(fh)
            for gi in range(pf.metadata.num_row_groups):
                tbl = pf.read_row_group(gi, columns=columns)
                for row in tbl.to_pylist():
                    yield row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="куда сложить кэш кандидатов (.jsonl.gz)")
    ap.add_argument("--viewport", default="desktop", choices=["desktop", "tablet", "mobile"],
                    help="какой вьюпорт оставить при дедупе по sample_id")
    ap.add_argument("--shards", type=int, default=N_SHARDS, help="сколько шардов прочитать (для смоука)")
    ap.add_argument("--limit", type=int, default=0, help="ограничить число страниц (0 = все)")
    args = ap.parse_args()

    if os.path.exists(args.out):
        print(f"[фаза 0] кэш уже есть: {args.out} — удали, если нужен перезабор")
        return

    shards = [f"{REPO}/{SHARD_FMT.format(i=i, n=N_SHARDS)}" for i in range(args.shards)]
    seen, n_rows, n_wrong_vp, n_dup, n_empty = set(), 0, 0, 0, 0
    t0 = time.time()
    tmp = args.out + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as out:
        for row in iter_rows(shards, COLUMNS):
            n_rows += 1
            if row.get("viewport") != args.viewport:
                n_wrong_vp += 1
                continue
            sid = row.get("sample_id")
            if sid in seen:
                n_dup += 1          # страховка: в источнике sample_id×viewport уникален
                continue
            if not (row.get("html") or "").strip():
                n_empty += 1
                continue
            seen.add(sid)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            if args.limit and len(seen) >= args.limit:
                break
    os.replace(tmp, args.out)       # атомарно: недокачанный кэш не должен выглядеть готовым
    print(f"[фаза 0] строк просмотрено {n_rows}; не тот вьюпорт {n_wrong_vp}; "
          f"дублей sample_id {n_dup}; пустой html {n_empty}")
    print(f"[фаза 0] страниц в кэше: {len(seen)} -> {args.out}  ({time.time() - t0:.0f} с)")


if __name__ == "__main__":
    main()
