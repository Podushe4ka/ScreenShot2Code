# Data-трек — Screenshot2Code

Сбор, анализ и подготовка датасетов для дообучения Qwen-VL под генерацию UI по
скриншотам. Методический ориентир — статья **UI2Code^N**
([`papers/UI2CodeN_2511.08195.pdf`](papers/UI2CodeN_2511.08195.pdf)): pretrain → SFT → RL.

## С чего начать

1. **[`PLAN.md`](PLAN.md)** — план работ и дорожная карта трека. Главный документ.
2. **[`list_data.md`](list_data.md)** — каталог источников: train-кандидаты, бенчмарки, смежное.
3. Собрать датасет — [`converters/`](converters/) (ниже про выбор конвертера).
4. Отдать данные в SFT — контракт [`../SFT/DATA_FORMAT_CONTRACT.md`](../SFT/DATA_FORMAT_CONTRACT.md)
   и [`converters/websight/HANDOFF.md`](converters/websight/HANDOFF.md).
5. Что из этого уже прогнано и с каким результатом — [`../docs/RESULTS.md`](../docs/RESULTS.md),
   что делать дальше — [`../docs/ROADMAP.md`](../docs/ROADMAP.md).

## Структура папки

| Путь | Что внутри |
|---|---|
| [`converters/`](converters/) | источник → формат контракта. По папке на датасет + общий финальный шаг |
| [`converters/common/`](converters/common/) | **общее ядро**: бюджет токенов/пикселей (общий с SFT), рендер, плейсхолдеры, схема контракта. Всё, что раньше дублировалось по конвертерам |
| [`converters/websight/`](converters/websight/) | WebSight: ядро логики (`convert_lib.py`), батч через Docker (`convert_parallel.py`, `Dockerfile`), просмотр (`view_arrow.py`), передача в SFT (`HANDOFF.md`) |
| [`converters/webcode2m/`](converters/webcode2m/) | WebCode2M: реальные страницы. Переиспользует ядро WebSight-конвертера, не дублирует его. `convert_raw.py` собирает **сырой** набор — контроль к чистому |
| [`converters/webui/`](converters/webui/) | WebUI: CSS лежит отдельной колонкой (весь стайлшит сайта, до 470 КБ) — лечится **tree-shaking** через CDP, см. [README](converters/webui/README.md) |
| [`converters/synth/`](converters/synth/) | синтетика: приёмка и рендер сгенерированных страниц, порча под polishing/editing, сборка датасета |
| [`converters/complexity/`](converters/complexity/) | скоринг сложности по **отрендеренной** странице + отбор по перцентилям (общий для WebUI и WebCode2M), см. [README](converters/complexity/README.md) |
| [`converters/mix/`](converters/mix/) | солянка из нескольких источников с колонкой `source` под последующую абляцию |
| [`converters/make_split.py`](converters/make_split.py) | финальный шаг: разрез на `train`/`validation` для `eval_loss` |
| [`generators/synth/`](generators/synth/) | генерация синтетики: сиды, ТЗ, пачки, [промпты](generators/synth/prompts/README.md), вендоринг CDN |
| [`eda/`](eda/) | разведка корпусов: сводка [`datasets_overview.md`](eda/datasets_overview.md), методика метрик [`required_data.md`](eda/required_data.md), особенности [`dataset_notes.md`](eda/dataset_notes.md) |
| [`eda/notebooks/`](eda/notebooks/) | ноутбуки по датасетам: `webcode2m`, `websight`, `webui` |
| [`eda/tools/`](eda/tools/) | счётчики и графики: `token_len.py`, `pixel_budget.py`, `plot_hist.py`, `make_examples.py`, `design2code_study.py`, `webui_clean_eda.py` (распределения по РЕАЛЬНОМУ выходу конвертера WebUI, не по сырому источнику) |
| [`papers/`](papers/) | PDF статей ко всем датасетам и методу ([индекс](papers/README.md)) |
| [`tests/`](tests/) | pytest без сети и браузера: бюджет, плейсхолдеры, чистка HTML, CSS, признаки, барьеры отбора, разрезы, `--help` всех CLI |

Не в git (регенерируются, лежат локально): `images/`, `report.html` — выхлоп
`view_arrow.py`; `websight_drafting_pilot/` — собранный датасет, передаётся диском;
`vendor/` — локальные копии CDN-библиотек (`generators/synth/fetch_vendor.sh`);
`synth_pilot/` — сиды, сгенерированные страницы, скриншоты и готовый синтетический набор.

## Синтетика: reverse construction

Схема обратная привычной (UI2Code^N 2511.08195 §3.2.2): сначала пишется идеальный
HTML, потом из него рендерится скриншот, и уже скриншот становится запросом к модели.
Качество страницы — это буквально качество разметки, которую модель выучит наизусть,
поэтому приёмка жёсткая.

```
generators/synth/fetch_seeds.py     реальные ТЗ-сиды (не выдумываем содержание)
generators/synth/make_briefs.py     briefs.jsonl — сетка ПОКРЫТИЯ (тир, язык, impl)
generators/synth/make_batches.py    разбивка на пачки под исполнителя
        │  исполнитель пишет <id>.html по prompts/01_page_generation.md
        ▼
converters/synth/build.py           линт → рендер → отбраковка → manifest.jsonl
converters/synth/mutate.py          порча страницы → пары для editing
converters/synth/degrade.py         порча вёрстки → пары для polishing
converters/synth/pack.py            сборка под контракт (drafting+editing+polishing)
converters/synth/make_split_grouped.py   разрез, НЕ разрывая страницу между сплитами
```

Вспомогательное: `renderlib.py` — рендер с материализацией DOM (для `react_cdn`
скриншот снимается с отрисованного React, а в `target_html` едет сырой исходник);
`slop.py` — детектор машинных дефолтов, метрика разнообразия набора, **отчёт, а не
отбраковка**; `contact_sheet.py` — контактный лист для глазной проверки.

Два стиля вывода (`impl`): `static_inline` и `react_cdn`. Стиль едет в датасет
отдельной колонкой и выбирает промпт в `SFT/train/formatting.py` — иначе один и тот же
скриншот отображался бы в два разных валидных таргета, а это противоречивый супервижн.

## Почему конвертеры названы по источнику

Оба делают одно и то же — приводят пару «скриншот + HTML» к формату контракта, —
и различаются только тем, откуда берут сырьё. Раньше один назывался `drafting/`
(по задаче), другой `webcode2m/` (по датасету), и из имён нельзя было понять, что
это одна и та же операция над разными корпусами.

Задача (drafting / polishing / editing) — это поле в контракте, а не папка:
polishing- и editing-данные будут собираться теми же конвертерами.

## Быстрый старт: собрать drafting-датасет из WebSight

```bash
docker build -t ws-conv -f Data/converters/websight/Dockerfile .
docker run --rm -v "$PWD":/work --shm-size=2g ws-conv --target 5000 --n-workers 64
```

Результат — `Data/websight_drafting_pilot/`. Дальше разрезать на сплиты:

```bash
.venv/bin/python Data/converters/make_split.py Data/websight_drafting_pilot <OUT> --val-frac 0.05
```

Подробности — [`converters/websight/README.md`](converters/websight/README.md),
передача в SFT — [`converters/websight/HANDOFF.md`](converters/websight/HANDOFF.md).

## Тесты

```bash
.venv/bin/python -m pytest Data/tests -q
```

Сети и браузера не требуют, идут ~10 секунд. Что держат: пиксельный бюджет сходится с
`SFT/train/formatting.py`, конвенция серых плейсхолдеров не поехала, из реальных страниц
не остаётся внешних ссылок, барьеры отбора не срабатывают вхолостую, разрезы не пускают
одну страницу в оба сплита, `--help` каждого CLI жив. Новый CLI без теста на `--help`
роняет `test_script_list_is_complete` — это намеренно.

## Статус

- ✅ **Этап 0** — разведка корпусов и токенные метрики.
- ✅ **Этап 1** — drafting-конвертер работает: параллельный батч через Docker,
  self-contained, ~5k за пару минут. Оба источника (WebSight, WebCode2M) отданы в SFT.
- ⏳ **Этапы 2–4** — детерминированный рендерер (переиспользуем eval-трек),
  reverse-construction синтетика, масштабирование претрейна.

⚠ **Главный вывод трека на 6 августа:** дообучение на WebCode2M стабильно **хуже**
необученной базы, и разбор в [`../docs/ROADMAP.md`](../docs/ROADMAP.md) связывает это
с самими данными: WebCode2M — pretrain-корпус, а не SFT-набор. Приоритет Этапа 3
(reverse-construction) от этого сильно вырос.
