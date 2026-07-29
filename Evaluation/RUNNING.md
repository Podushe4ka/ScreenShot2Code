# Запуск бенчмарка в Docker

Файлы: `Dockerfile`, `build.sh`, `run.sh` (или `docker-compose.yml` +
`compose-run.sh` как альтернатива), плюс сам код: `render.py`, `metrics.py`,
`run_benchmark.py`, `run_benchmark_batched.py`.

`test.py` в Dockerfile через `COPY` — файла не было среди присланных, так что
либо положите его рядом перед сборкой, либо уберите его из строки `COPY` в
Dockerfile, если он не нужен. `build.sh` проверяет это и остановится с понятной
ошибкой, если файла нет.

## 1. Собрать образ

```bash
./build.sh
# или свой тег:
./build.sh design2code-bench:v1
```

## 2. Запустить

Вариант А — `run.sh` (docker run напрямую):

```bash
./run.sh --n-samples 70000 --batch-size 10000
```

Вариант Б — docker compose:

```bash
./compose-run.sh --n-samples 70000 --batch-size 10000
```

Оба варианта делают одно и то же:
- пробрасывают GPU (`--gpus all` / nvidia reservation);
- монтируют `./bench_results` (хост) → `/app/output` (контейнер) — здесь
  будут `results.csv`, `examples/`, `progress.json`, `summary.json`;
- монтируют `./hf_cache` (хост) → `/root/.cache/huggingface` — модель и
  датасет скачиваются один раз и переиспользуются между запусками контейнера;
- задают `--shm-size 2g` и `--ipc=host` — без этого Chromium (Playwright)
  может падать при нескольких параллельных воркерах.

Любые аргументы `run_benchmark_batched.py` (см. `--help` там) передаются как
есть после имени скрипта: `--model`, `--num-workers`,
`--tensor-parallel-size`, `--n-examples-per-batch` и т.д.

**Не передавайте `--outdir`** в `run.sh`/`compose-run.sh` — путь внутри
контейнера всегда фиксирован на `/app/output` (единственный смонтированный
путь); оба скрипта явно это проверяют и остановятся с ошибкой, если вы всё
же попытаетесь его передать. Чтобы результаты лежали в другом месте на
хосте — задайте `HOST_OUTDIR` (для `run.sh`) или поменяйте volume в
`docker-compose.yml`.

## 3. Если прогон упал на середине (OOM, краш и т.п.)

Просто запустите ту же команду ещё раз:

```bash
./run.sh --n-samples 70000 --batch-size 10000
```

`run_benchmark_batched.py` сам читает `/app/output/progress.json` и
продолжает с первого необработанного батча — уже посчитанные метрики не
пересчитываются заново (см. `MetricAccumulator` в коде). `run.sh` перед
запуском удаляет старый контейнер с тем же именем (`docker rm -f`), если он
остался в статусе "не запущен" — данные в volume `bench_results`/`hf_cache`
при этом не трогаются.

Если вместо resume нужно начать полностью заново — добавьте `--no-resume`
(он сотрёт `progress.json`, `results.csv` и `examples/` внутри `/app/output`):

```bash
./run.sh --no-resume --n-samples 1000
```

## 4. Проверить прогресс во время долгого прогона

Файлы обновляются по ходу дела на хосте (в `./bench_results`), не только по
завершении:

```bash
cat bench_results/progress.json    # текущий батч, накопленные метрики
tail -f bench_results/time_results.txt
```

## 5. Результаты

После завершения (или в любой момент во время прогона — числа там неполные,
но валидные):
- `bench_results/summary.json` — итоговые средние по 5 метрикам +
  `final_score`/`final_score_arithmetic`, число оценённых и исключённых
  сэмплов;
- `bench_results/results.csv` — по ~5 строк-примеров на батч;
- `bench_results/examples/batch_XXXXX/` — html+png нескольких случайных
  сэмплов из каждого батча.
