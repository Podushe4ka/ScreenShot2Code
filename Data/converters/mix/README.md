# Солянка: смешанный drafting-набор

Собирает один `Dataset` из нескольких источников и проставляет каждому сэмплу колонку
`source`. Колонка — это и есть смысл сборки: она позволяет потом мерить вклад каждого
источника **абляцией на уже собранных данных**, не пересобирая набор.

```bash
python build_mix.py \
    --part "webui_inline=sel_webui.jsonl" \
    --part "webcode2m_complex=sel_wc2m.jsonl" \
    --part "synth=../../synth_pilot" \
    --out /path/mix
```

Часть задаётся как `МЕТКА=путь`, где путь — либо `jsonl` отбора от
[`../complexity/select_by_complexity.py`](../complexity/select_by_complexity.py), либо
staging-каталог конвертера. Раскладки файлов у источников разные (staging держит страницы
в `pages/`, синтетика — исходник в `raw/`, скриншот в `build/`), поэтому пути разрешаются
по нескольким кандидатам, а не по одной жёсткой схеме.

## Схема

Контракт [`SFT/DATA_FORMAT_CONTRACT.md`](../../../SFT/DATA_FORMAT_CONTRACT.md) §2 плюс две
служебные колонки: `source` (метка части) и `page_id` (`<source>:<id>`). Контракт лишние
колонки допускает — SFT-загрузчик читает только `task_type` / `images` / `current_html` /
`target_html` / `instruction`.

## Разрез

```bash
python ../make_split.py /path/mix /path/mix_split
```

Обычный [`make_split.py`](../make_split.py), а **не**
[`synth/make_split_grouped.py`](../synth/make_split_grouped.py). Групповой разрез нужен
там, где из одной страницы выходит несколько сэмплов (drafting + editing + polishing) с
одинаковым `target_html`: при случайном разрезе почти-дубли расползаются по обоим сплитам
и занижают `eval_loss`. Здесь набор **только drafting**, на страницу приходится ровно один
сэмпл, и `page_id` уникален в каждой строке — групповой разрез совпал бы со случайным.

Если в солянку когда-нибудь добавят polishing/editing-варианты тех же страниц, разрез
обязан переехать на `make_split_grouped.py` по ключу `page_id` — он для этого и пишется.

## Ограничение по размеру куска

`Dataset.from_list` собирается кусками по 500 строк. Все картинки куска лежат в одном
бинарном массиве Arrow, а у него 32-битные оффсеты — предел 2 ГБ на массив; 15k сэмплов
это ~7.5 ГБ и падение с `offset overflow while concatenating arrays`. Обоснование
унаследовано из [`../webcode2m/convert_parallel.py`](../webcode2m/convert_parallel.py).
