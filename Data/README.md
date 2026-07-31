# Data-трек — Screenshot2Code

Сбор, анализ и подготовка данных для дообучения Qwen-VL под генерацию UI по
скриншотам (drafting / polishing / editing). Методический ориентир — статья
**UI2Code^N** (`papers/UI2CodeN_2511.08195.pdf`): pretrain → SFT → RL.

## С чего начать

1. **[`PLAN.md`](PLAN.md)** — план работ, дорожная карта, ключевые решения и открытые
   вопросы. Главный документ трека — читать первым.
2. **[`list_data.md`](list_data.md)** — каталог датасетов (train-кандидаты, бенчмарки, смежное).
3. Хочешь собрать drafting-датасет — [`drafting/README.md`](drafting/README.md).
4. Отдаёшь данные в SFT — [`drafting/HANDOFF.md`](drafting/HANDOFF.md) + контракт
   [`../SFT/DATA_FORMAT_CONTRACT.md`](../SFT/DATA_FORMAT_CONTRACT.md).

## Структура папки

| Путь | Что внутри |
|---|---|
| [`PLAN.md`](PLAN.md) | План работ и дорожная карта (этапы 0–4) |
| [`list_data.md`](list_data.md) | Каталог датасетов |
| [`analysis/`](analysis/) | **Этап 0** — EDA корпусов: **сводка по всем датасетам ([`datasets_overview.md`](analysis/datasets_overview.md))**, ноутбуки по датасетам, методика метрик (`required_data.md`), заметки (`dataset_notes.md`), токен-счётчик (`token_len.py`), визуальное сравнение (`compare_datasets.py`) |
| [`drafting/`](drafting/) | **Этап 1** — конвертер WebSight → формат контракта (логика в `convert_lib.py`, батч `convert_parallel.py`, `Dockerfile`, просмотр `view_arrow.py`, передача `HANDOFF.md`) |
| [`papers/`](papers/) | PDF статей ко всем датасетам и методу ([индекс](papers/README.md)) |

> `pretrain/` (стриминговый претрейн-микс) живёт на ветке `data/pretrain` — см. ниже.

## Ветки трека

Три ветки = общий baseline (`analysis/` + `PLAN.md` + `list_data.md`) плюс свои файлы:

| Ветка | Добавляет к baseline |
|---|---|
| `data/analysis` | только baseline (EDA + токенные метрики) |
| `data/drafting` | `drafting/` — drafting-конвертер |
| `data/pretrain` | `pretrain/` — претрейн-микс |

Baseline держим синхронным между ветками вручную (коммиты `sync …`), поэтому его
структуру меняем осознанно и одинаково во всех трёх.

## Статус

- ✅ **Этап 0** — разведка + токенные метрики (перепрогон на WebSight v0.2).
- ▶ **Этап 1** — drafting-конвертер работает: параллельный батч через Docker,
  self-contained, ~5k за пару минут. Готов handoff в SFT.
- ⏳ **Этапы 2–4** — детерминированный рендерер (переиспользуем eval-трек),
  reverse-construction синтетика (polishing/editing), масштабирование претрейна.

Подробности и открытые вопросы (главный — единый размер скриншота, §4) — в [`PLAN.md`](PLAN.md).

## Быстрый старт: собрать drafting-датасет

```bash
# из корня репозитория
docker build -t ws-conv -f Data/drafting/Dockerfile .
docker run --rm -v "$PWD":/work --shm-size=2g ws-conv --target 5000 --n-workers 64
```

Результат — `Data/websight_drafting_pilot/` (в `.gitignore`; передаётся диском/томом, не
через git). Дальше — по [`drafting/HANDOFF.md`](drafting/HANDOFF.md).
