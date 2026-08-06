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
| [`converters/websight/`](converters/websight/) | WebSight: ядро логики (`convert_lib.py`), батч через Docker (`convert_parallel.py`, `Dockerfile`), просмотр (`view_arrow.py`), передача в SFT (`HANDOFF.md`) |
| [`converters/webcode2m/`](converters/webcode2m/) | WebCode2M: реальные страницы. Переиспользует ядро WebSight-конвертера, не дублирует его |
| [`converters/make_split.py`](converters/make_split.py) | финальный шаг обоих: разрез на `train`/`validation` для `eval_loss` |
| [`eda/`](eda/) | разведка корпусов: сводка [`datasets_overview.md`](eda/datasets_overview.md), методика метрик [`required_data.md`](eda/required_data.md), особенности [`dataset_notes.md`](eda/dataset_notes.md) |
| [`eda/notebooks/`](eda/notebooks/) | ноутбуки по датасетам: `webcode2m`, `websight`, `webui` |
| [`eda/tools/`](eda/tools/) | счётчики и графики: `token_len.py`, `pixel_budget.py`, `plot_hist.py`, `compare_datasets.py` |
| [`papers/`](papers/) | PDF статей ко всем датасетам и методу ([индекс](papers/README.md)) |

Не в git (регенерируются, лежат локально): `images/`, `report.html` — выхлоп
`view_arrow.py`; `websight_drafting_pilot/` — собранный датасет, передаётся диском.

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
