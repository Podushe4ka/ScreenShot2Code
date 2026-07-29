#!/usr/bin/env bash
# run.sh — запускает (или РЕЗЮМИРУЕТ) прогон бенчмарка в контейнере.
#
# Это один и тот же скрипт для первого запуска и для resume после падения:
# resume работает через outdir/progress.json внутри python-скрипта, так что
# просто перезапускаешь ту же команду и всё продолжится с того места, где
# упало (см. run_benchmark_batched.py). Отдельного "resume.sh" не нужно —
# см. также флаг --no-resume ниже, если нужно вместо этого начать с нуля.
#
# ВАЖНО про volume: --outdir и HF-кэш ОБЯЗАНЫ жить на хосте (bind mount), а
# не только внутри контейнера. Если контейнер после падения будет удалён
# (docker rm) или просто пересоздан, а outdir не был смонтирован — progress.json
# и накопленные examples/ исчезнут вместе с контейнером, и resume будет
# нечем резюмировать. Оба volume ниже монтируются с хоста именно поэтому.
#
# Примеры:
#   ./run.sh --n-samples 70000 --batch-size 10000
#   ./run.sh --n-samples 70000 --num-workers 4 --tensor-parallel-size 2
#   ./run.sh --no-resume --n-samples 1000        # начать заново, игнорируя чекпоинт
#
# Переменные окружения (можно переопределить перед вызовом):
#   IMAGE_TAG      - какой образ запускать (по умолчанию design2code-bench:latest)
#   HOST_OUTDIR    - куда на хосте класть результаты/чекпоинт (по умолчанию ./bench_results)
#   HOST_HF_CACHE  - куда на хосте класть кэш HF моделей/датасетов (по умолчанию ./hf_cache)
#   GPUS           - какие GPU пробросить (по умолчанию all)
#   CONTAINER_NAME - имя контейнера (по умолчанию design2code-bench)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_TAG="${IMAGE_TAG:-design2code-bench:latest}"
HOST_OUTDIR="${HOST_OUTDIR:-$SCRIPT_DIR/bench_results}"
HOST_HF_CACHE="${HOST_HF_CACHE:-$HOME/.cache/huggingface}"
GPUS="${GPUS:-all}"
CONTAINER_NAME="${CONTAINER_NAME:-design2code-bench}"

mkdir -p "$HOST_OUTDIR" "$HOST_HF_CACHE"

# Защита: --outdir всегда должен указывать на /app/output (единственный путь,
# смонтированный volume-ом с хоста, см. -v ниже). Если пользователь передаст
# свой --outdir в "$@", argparse внутри python молча возьмёт ПОСЛЕДНИЙ (свой),
# он окажется внутри контейнера без volume - и после docker rm/падения все
# результаты и progress.json потеряются без единого предупреждения. Поэтому
# ловим это здесь явно, а не полагаемся на argparse.
for arg in "$@"; do
    if [[ "$arg" == "--outdir" || "$arg" == --outdir=* ]]; then
        echo "ОШИБКА: не передавайте --outdir в run.sh — путь внутри контейнера" >&2
        echo "всегда фиксирован на /app/output. Чтобы изменить место на ХОСТЕ," >&2
        echo "задайте переменную HOST_OUTDIR перед запуском, например:" >&2
        echo "  HOST_OUTDIR=/data/my_run ./run.sh --n-samples 70000" >&2
        exit 1
    fi
done

# Если контейнер с таким именем уже существует, но не запущен (например,
# после docker stop) — уберём его перед пересозданием, иначе `docker run
# --name` откажется стартовать с "the container name is already in use".
# Volume-данные (outdir/hf_cache) при этом не трогаются - они на хосте.
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "[run] Удаляю старый контейнер $CONTAINER_NAME (данные в volume сохранены)..."
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

echo "[run] Образ:        $IMAGE_TAG"
echo "[run] outdir (хост): $HOST_OUTDIR  ->  /app/output (в контейнере)"
echo "[run] HF cache:      $HOST_HF_CACHE  ->  /root/.cache/huggingface"
echo "[run] GPU:           $GPUS"
echo "[run] Аргументы скрипту: $* "
echo

# --shm-size: и vLLM (KV-cache/тензорный параллелизм), и Playwright/Chromium
# (см. render.py — --disable-gpu, но /dev/shm по умолчанию у Docker всего
# 64MB) требуют больше дефолтного /dev/shm; без этого Chromium периодически
# падает с "Target crashed"/"Target closed" под нагрузкой в несколько
# параллельных воркеров (--num-workers > 1).
#
# --outdir всегда фиксирован на /app/output внутри контейнера (примонтирован
# с хоста) - остальные аргументы (--n-samples, --batch-size, --model, и т.д.)
# прозрачно прокидываются как есть в run_benchmark_batched.py через "$@".
docker run \
    --name "$CONTAINER_NAME" \
    --gpus "$GPUS" \
    --shm-size=1g \
    --ipc=host \
    -v "$HOST_OUTDIR:/app/output" \
    -v "$HOST_HF_CACHE:/root/.cache/huggingface" \
    "$IMAGE_TAG" \
    --outdir /app/output \
    "$@"
