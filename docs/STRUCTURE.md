# Структура репозитория и общего диска

Карта проекта для тех, кто заходит впервые. Актуально на 6 августа 2026.

---

## 1. Репозиторий

```
ScreenShot2Code/
├── SFT/            обучение (torchrun + TRL + DeepSpeed), образ `sft`
├── Evaluation/     бенч Design2Code (vLLM + playwright + CLIP), образ `design2code-bench`
├── RL/             задел под RL, пока пусто
├── Data/           данные: сборка датасетов, анализ, ноутбуки
├── experiments/    оркестраторы прогонов (.sh) — запускаются НА ХОСТЕ
└── docs/           документация: эта карта, инвентарь диска, описания экспериментов
```

### `SFT/` — обучение

| путь | что |
|---|---|
| `run.sh` | поднимает контейнер `sft` и запускает в нём команду |
| `configs/*.yaml` | конфиги обучения (`full_ft_qwen3_5_4b.yaml`, LoRA-варианты) |
| `configs/deepspeed_zero2.json` | ZeRO-2 |
| `train/train_sft.py` | точка входа; отбраковка по длине, meta-лог |
| `train/formatting.py` | промпты (`DRAFTING_PROMPT`/`POLISHING_PROMPT`/`EDITING_PROMPT`), сборка messages, коллатор с маскированием лосса по ходам ассистента |
| `train/batching.py` | бакетинг по длине |
| `train/tracking.py` | ClearML |
| `scripts/merge_lora.py` | слияние LoRA-адаптера с базой (на CPU) |
| `scripts/collect_results.py` | сбор результатов прогонов |
| `DATA_FORMAT_CONTRACT.md` | **контракт формата датасета** — читать перед сборкой данных |

Внутри образа Python — `/opt/venv/bin/python`, **не** `python3`.

### `Evaluation/` — бенч

| путь | что |
|---|---|
| `run.sh` | поднимает контейнер бенча; `--outdir` фиксирован на `/app/output` |
| `build.sh` | пересборка образа — **обязательно после правок .py** |
| `run_benchmark_batched.py` | основной прогон: батчи, resume, метрики |
| `clip_server.py` | один общий CLIP-процесс с батчингом вместо копии в каждом воркере |
| `metrics.py`, `render.py` | метрики Design2Code и рендер через playwright |
| `tracking.py` | ClearML |
| `RUNNING.md` | как запускать |

### `experiments/` — оркестраторы

Запускаются **на хосте**, не внутри контейнера: они сами поднимают контейнеры
через `SFT/run.sh` и `Evaluation/run.sh`. Все пути ведут на `/mnt/storage-1`.

| скрипт | что делает |
|---|---|
| `run_pilot.sh` | пилот на WebCode2M: E1–E6, обучение → merge LoRA → бенч |
| `run_night.sh` | ночная цепочка: бенч 1k → генерация 3k → те же экспы на 3k → бенч |
| `bench_all.sh` | добенчить все готовые чекпоинты пилота; умеет ждать конца обучения |
| `sanity_d2c.sh` | sanity-overfit на Design2Code как есть (устарел: таргеты резались) |
| `sanity_fit.sh` | sanity-overfit на **коротких** Design2Code, train == eval; собирает `d2c_short` и `d2c_short_bench` |
| `sanity_sweep.sh` | свип по эпохам: 5 эпох, чекпоинт каждую, бенч каждого |
| `sweep_greedy.sh` | перебенч готовых чекпоинтов в greedy (без переобучения) |
| `sweep_soft.sh` | мягкий рецепт (lr 5e-6, cosine, warmup) + greedy-бенч каждой эпохи |
| `sweep_wc2m.sh` | тот же оверфит, но на WebCode2M вместо Design2Code (обучение ещё не проходило — OOM) |

### `Data/`

| путь | что |
|---|---|
| `webcode2m/convert_parallel.py` | генерация WebCode2M: рендер HTML → PNG через playwright, `RENDER_WIDTH=1280` |
| `make_split.py` | разрез датасета на train/validation |
| `analysis/` | разборы: длины, пиксель-бюджет, гистограммы, обзоры датасетов, ноутбуки |
| `drafting/`, `papers/`, `images/` | вспомогательное |
| `PLAN.md`, `list_data.md` | планы и списки источников |

### `docs/`

| файл | что |
|---|---|
| `STRUCTURE.md` | этот файл |
| `RESULTS.md` | все модели, гиперпараметры и скоры в одной таблице + описание бенчмарков |
| `storage_inventory.md` | что занимает место на общем диске, что удалено и почему |
| `experiments/*.md` | по файлу на каждый проведённый эксперимент |

---

## 2. Общий диск `/mnt/storage-1`

8 ТБ, общий между **a100-2** и **a100-3**. Мой рабочий каталог —
`/mnt/storage-1/Screenshot2Code` (маленькая `s`!). Не путать с
`/mnt/storage-1/ScreenShot2Code` — это каталог t.chichkanov.

```
/mnt/storage-1/Screenshot2Code/          142 ГБ
├── checkpoints_exps/     93 ГБ   все эксперименты с бенчами
├── Data/runs/            12 ГБ   июльская линия (до пилота)
├── hf_cache/             35 ГБ   базовые модели (hub 28 ГБ) + кэши компиляции
├── data/                2.4 ГБ   обучающие датасеты
├── container-home/      1.1 ГБ   HOME контейнеров обучения
└── *.log                        логи прогонов
```

### `checkpoints_exps/`

| каталог | что | документация |
|---|---|---|
| `exps-20260803-104549` | пилот 1k, E0–E6 | [пилот 1k](experiments/2026-08-03-pilot-1k.md) |
| `exps-3k` | пилот 3k, E0–E6 | [пилот 3k](experiments/2026-08-04-pilot-3k.md) |
| `d2c-overfit` | первый sanity (таргеты резались) | [sanity](experiments/2026-08-05-sanity-overfit.md) |
| `d2c-sanity-fit` | рабочий sanity, 15 эпох | [sanity](experiments/2026-08-05-sanity-overfit.md) |
| `d2c-sweep` | свип по эпохам + перепроверка greedy | [свип](experiments/2026-08-05-sweep-epochs.md), [greedy](experiments/2026-08-06-greedy-recheck.md) |
| `d2c-sweep-soft` | мягкий рецепт lr 5e-6 | [мягкий рецепт](experiments/2026-08-06-sweep-soft.md) |
| `wc2m-sweep` | оверфит на WebCode2M: только бенч базы, весов нет | [оверфит на WebCode2M](experiments/2026-08-06-wc2m-overfit.md) |
| `d2c-compare50`, `smoke-merged` | замеры базы и смоук | [мелкие прогоны](experiments/misc-smoke-and-compare.md) |

Раскладка внутри прогона: веса в `<run_name>/checkpoint-N/`, результаты бенча
в отдельном каталоге рядом (`*-bench/summary.json`, `results.csv`, `examples/`).

### `data/` — обучающие датасеты

| каталог | что |
|---|---|
| `webcode2m_1000_split`, `webcode2m_3000_split` | WebCode2M с val-сплитом |
| `webcode2m_3000` | без сплита |
| `d2c_short` | 40 коротких Design2Code, drafting-формат — для обучения sanity/свипов |
| `wc2m_short` | 40 из WebCode2M, тот же формат — для оверфита на коротких таргетах |
| `d2c_overfit` | первый (неудачный) набор sanity |

`hf_cache/d2c_short_bench` и `hf_cache/wc2m_short_bench` — те же 40 в формате
`image`/`text` для бенча.

---

## 3. Что надо знать до первого запуска

**Всё на серверах — только через docker.** Два образа: `sft` (обучение) и
`design2code-bench` (бенч). Собираются локально на каждой машине, поэтому
**могут различаться между a100-2 и a100-3** — после правок .py обязательно
`cd Evaluation && bash build.sh` на той машине, где будете гонять.

**Два образа монтируют диск по-разному, пути НЕ взаимозаменяемы:**

| | `sft` | `design2code-bench` |
|---|---|---|
| общий диск | `/storage` | `/mnt/storage-1` (read-only) |
| HF-кэш | `$HF_HOME` | `/root/.cache/huggingface` |

Путь `/storage/...`, переданный в бенч, внутри него не существует.

**⚠ UID `v.kozlov` разный на машинах:** 1037 на a100-2, 1004 на a100-3. Файлы,
созданные на одной машине, на другой видны как чужие и **недоступны на запись** —
редирект в существующий лог падает с `Permission denied`, и вся команда молча
не выполняется. Переносите прогон на другую машину — отложите старые логи `mv`.

**Файлы, созданные docker'ом, принадлежат root.** Удалять/переименовывать их —
из контейнера (`--entrypoint bash "$IMG" -c "rm -rf ..."`), либо `mv`/`rm` из
каталога, на который у вас есть право записи.

**GPU общие с коллегами.** Перед запуском — `nvidia-smi`, брать только свободные
карты, чужие контейнеры не трогать. Пробрасывать `GPUS='"device=N"'` —
**внутренние кавычки обязательны**, без них docker падает с rc=125.
`--tensor-parallel-size` должен делить 32 головы внимания → 1, 2 или 4, **не 3**.

**Локальный диск a100-2 (`/`, 2 ТБ) забит под завязку** — там нельзя пересобрать
образ. Все результаты, кэши и `CONTAINER_HOME` слать на `/mnt/storage-1`.
Но `TMPDIR` Chromium'а на сетевой диск ставить нельзя — падает `Target crashed`.

**Декодирование в бенче — greedy по умолчанию** (`--temperature 0`).
До 6 августа параметры не задавались вовсе, и бенч работал на дефолтах vLLM
(сэмплинг T=1.0 без seed): **все числа, снятые раньше, с новыми несравнимы.**
