# Запуск бенчмарка в Docker (pairwise LLM-judge: checkpoint vs baseline, 1 GPU)

Файлы: `Dockerfile`, `build.sh`, `run.sh`, `serve_models.sh`, `vllm_server_manager.py`,
плюс код: `render.py`, `metrics.py`, `clip_server.py`, `vllm_client.py`,
`judge_client.py`, `run_benchmark_batched.py`.

## Что считается

- **Чекпоинт** (`MODEL_CHECKPOINT`) — модель, которую реально оцениваем: 5
  официальных метрик Design2Code (block_match, text, position, color, clip,
  final_score, final_score_arithmetic) через `metrics.score_pair`.
- **Baseline** (`MODEL_BASELINE`, по умолчанию `Qwen/Qwen3.5-4B`) — генерирует
  HTML только для сравнения судьёй. Официальные метрики по ней не считаются.
- **Judge** (`MODEL_JUDGE`, по умолчанию тоже `Qwen/Qwen3.5-4B`) — получает
  ref.png + pred чекпоинта + pred baseline и решает, что ближе к ref. Итог —
  winrate чекпоинта относительно baseline.

Anti-position-bias: порядок показа (A/B) судье выбирается монеткой на каждом
сэмпле независимо (см. `judge_client.py`).

## Архитектура: 1 GPU, модели грузятся по очереди

Одна GPU. На каждом батче модели загружаются и выгружаются последовательно
(`vllm_server_manager.py` сам поднимает/гасит `vllm serve` подпроцессом):

1. Поднять **checkpoint** → сгенерировать HTML по батчу → выгрузить
2. Рендер эталона + рендер и 5 метрик чекпоинта (Playwright + CLIP; GPU уже
   свободен от vLLM). Browser-bound этап — конкурентность задаётся отдельно
   от остальных через `--render-workers`/`RENDER_WORKERS` (по умолчанию
   `min(--num-workers, 8)`), т.к. на сэмпл здесь до 6 скриншотов (pred +
   OCR-free блоки, для чекпоинта и при необходимости эталона) — заметно
   больше нагрузки на браузер, чем на этапе 4.
3. Поднять **baseline** → сгенерировать HTML по батчу → выгрузить
4. Рендер baseline (только PNG, без метрик; один скриншот на сэмпл, тот же
   `--render-workers`)
5. Поднять **judge** → pairwise-сравнение по батчу → выгрузить (I/O-bound
   HTTP, конкурентность — `--num-workers`)
6. Следующий батч — снова с шага 1

Один и тот же порт (`--vllm-port`, дефолт 8001) переиспользуется всеми тремя
моделями по очереди. Между выгрузкой одной модели и загрузкой следующей —
`kill` + пауза (см. `D2C_VLLM_SHUTDOWN_GRACE_SEC` в `vllm_server_manager.py`),
без явной проверки `nvidia-smi`.

**CLIP-сервер** — единственное исключение: маленький (~350MB) отдельный
процесс, стартует один раз в начале всего прогона и живёт постоянно, деля
GPU с текущей vLLM-моделью. Поэтому `--gpu-memory-utilization` здесь
оставляет запас под него (по умолчанию 0.85).

## Устойчивость к сетевым сбоям и рендер-таймаутам

- **vLLM-запросы** (`vllm_client.py`): каждый запрос ретраится до 3 раз (с
  линейным backoff) при таймауте или обрыве соединения, прежде чем сэмпл
  считается `generation_error`. HTTP-ответ с ошибкой от самого сервера (не
  сетевой сбой) не ретраится — это не транзиент.
- **Рендер** (`render.py`): таймаут одного `page.goto`/`page.screenshot`
  настраивается через `--render-timeout-ms`/`RENDER_TIMEOUT_MS` (по
  умолчанию 90с). При сбое — один повтор (`--render-retries`, по умолчанию
  1) с пересозданием браузера перед повтором. Причина сбоя (timeout/other)
  сохраняется в `results.csv` (`ref_render_fail_reason`,
  `baseline_render_fail_reason`) вместо того чтобы молча превратиться в
  белую заглушку без объяснения.

## 0. Требования к железу

**1 GPU** (гарантированно на A100 80GB, но конкретный объём VRAM зависит от
модели). Индекс карты и порт настраиваются через `GPU_INDEX`/`VLLM_PORT`.

## 1. Собрать образ

```bash
./build.sh
# или свой тег:
./build.sh design2code-bench:v1
```

## 2. Запустить

```bash
MODEL_CHECKPOINT=/mnt/storage-1/checkpoints/my-run/step-12000 \
    ./run.sh --n-samples 2000 --batch-size 2000 --num-workers 112
```

Явно задать все три модели, GPU, порт и конкурентность рендера:

```bash
MODEL_CHECKPOINT=/mnt/storage-1/checkpoints/my-run \
MODEL_BASELINE=Qwen/Qwen3.5-4B \
MODEL_JUDGE=Qwen/Qwen3.5-4B \
GPU_INDEX=0 VLLM_PORT=8001 \
RENDER_WORKERS=8 RENDER_TIMEOUT_MS=120000 \
    ./run.sh --n-samples 70000 --batch-size 10000
```

`run.sh`:
- пробрасывает GPU (`--gpus all` по умолчанию, см. `GPUS`);
- передаёт `MODEL_*`/`GPU_INDEX`/`VLLM_PORT`/`GPU_MEMORY_UTILIZATION`/
  `RENDER_WORKERS`/`RENDER_TIMEOUT_MS` в контейнер как env — их читает
  `serve_models.sh` (ENTRYPOINT), тонкий скрипт, который сразу передаёт
  управление `run_benchmark_batched.py` (сам поднимает/гасит `vllm serve` по
  ходу батчей, см. выше);
- монтирует `./bench_results` → `/app/output` (`results.csv`, `examples/`,
  `progress.json`, `summary.json`, `server_logs/{checkpoint,baseline,judge}.log`);
- монтирует `./hf_cache` → HF-кэш моделей/датасета;
- монтирует `HOST_STORAGE` (по умолчанию `/mnt/storage-1`, read-only) → тот
  же путь в контейнере — общий диск с чекпоинтами;
- `--shm-size 2g --ipc=host` — для стабильности Chromium/Playwright.

**Не передавайте** `--model-checkpoint`/`--model-baseline`/`--judge-model`/
`--gpu-index`/`--vllm-port`/`--vllm-gpu-memory-utilization`/`--render-workers`/
`--render-timeout-ms`/`--outdir` напрямую — они выводятся из переменных
окружения; `run.sh` остановится с понятной ошибкой при попытке.

## 3. Если прогон упал на середине

Перезапустите ту же команду:

```bash
MODEL_CHECKPOINT=/mnt/storage-1/checkpoints/my-run ./run.sh --n-samples 70000 --batch-size 10000
```

`run_benchmark_batched.py` читает `/app/output/progress.json` и продолжает с
первого необработанного батча. `run.sh` перед запуском удаляет старый
контейнер с тем же именем, если он не запущен — volume-данные не трогаются.

Начать полностью заново: `--no-resume` (сотрёт `progress.json`,
`results.csv`, `examples/`).

## 4. Проверить прогресс

Тайминг каждого этапа (генерация/рендер/judge, отдельно деплой модели и сама
работа) печатается в лог контейнера сразу по завершении этапа — не нужно
ждать конца батча, чтобы увидеть, где сейчас прогон:

```bash
cat bench_results/progress.json
tail -f bench_results/time_results.txt
tail -f bench_results/server_logs/judge.log   # логи каждого из этапов — checkpoint/baseline/judge
```

## 5. Результаты

- `bench_results/summary.json` — средние по 5 метрикам чекпоинта +
  `judge_winrate_checkpoint`/`judge_winrate_baseline` + `judge_n_scored`;
- `bench_results/results.csv` — строки-примеры по батчам, включая причину
  сбоя рендера там, где он не удался (`*_render_fail_reason`);
- `bench_results/examples/batch_XXXXX/` — `ref.*`, `pred_checkpoint.*`,
  `pred_baseline.*` для нескольких сэмплов батча.

## Известные ограничения

- Плейсхолдер `replace_images_with_placeholder` (`render.py`) заменяет
  только теги `<img>`. Если модель рисует изображение другими средствами
  (div с `background-color`/CSS-классом), эта функция его не увидит и не
  тронет — это не баг постобработки, а следствие того, что модель не
  выразила изображение тегом `<img>` (например, не следуя инструкции
  промпта "plain gray placeholder boxes"). Стоит проверять сырой HTML
  чекпоинта, если метрики/скриншоты выглядят неожиданно.
