#!/usr/bin/env bash
# run.sh — запускает контейнер с демо, выбирая ОДНУ конкретную GPU из
# нескольких на сервере (см. переменную GPUS ниже).
#
# Примеры:
#   GPUS='"device=0"' ./run.sh --model Qwen/Qwen3.5-4B
#   GPUS='"device=2"' ./run.sh --model /mnt/storage-1/ScreenShot2Code/model_weights/<run>/<step>/
#   ./run.sh --model ...                       # GPUS по умолчанию = device=0
#
# После запуска — проброс порта СО СВОЕГО компьютера (не отсюда):
#   ssh -N -L 8501:localhost:8501 user@gpu-server
#   затем открыть http://localhost:8501 в локальном браузере
#
# Переменные окружения (можно переопределить перед вызовом):
#   IMAGE_TAG      - какой образ запускать (по умолчанию screenshot2code-demo:latest)
#   GPUS           - какую GPU пробросить, формат Docker --gpus (по умолчанию '"device=0"')
#   HOST_HF_CACHE  - куда на хосте класть кэш HF моделей (по умолчанию ~/.cache/huggingface)
#   HOST_STORAGE   - путь к общему диску с чекпоинтами на ХОСТЕ (по умолчанию /mnt/storage-1),
#                    монтируется В КОНТЕЙНЕР ПО ТОМУ ЖЕ ПУТИ, read-only — см. run.sh
#                    основного бенчмарк-пайплайна, тот же принцип.
#   PORT           - порт на хосте, слушаемый контейнером (по умолчанию 8501)
#   CONTAINER_NAME - имя контейнера (по умолчанию screenshot2code-demo)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_TAG="${IMAGE_TAG:-screenshot2code-demo:latest}"
GPUS="${GPUS:-\"device=0\"}"
HOST_HF_CACHE="${HOST_HF_CACHE:-$HOME/.cache/huggingface}"
HOST_STORAGE="${HOST_STORAGE:-/mnt/storage-1}"
PORT="${PORT:-8501}"
CONTAINER_NAME="${CONTAINER_NAME:-screenshot2code-demo}"

mkdir -p "$HOST_HF_CACHE"

# HOST_STORAGE монтируется, только если реально существует на хосте — иначе
# либо чекпоинт берётся с HF hub (--model Qwen/Qwen3.5-4B как есть), либо
# общий диск ещё не примонтирован на хосте, и лучше сообщить явно, чем
# упасть внутри transformers невнятной ошибкой "path not found".
STORAGE_MOUNT_ARGS=()
if [[ -d "$HOST_STORAGE" ]]; then
    STORAGE_MOUNT_ARGS=(-v "$HOST_STORAGE:/mnt/storage-1:ro")
else
    echo "[run] Внимание: $HOST_STORAGE не найден на хосте — общий диск с чекпоинтами" >&2
    echo "не будет примонтирован. Если --model указывает на /mnt/storage-1/..., это упадёт." >&2
    echo "Если диск смонтирован в другом месте, задайте HOST_STORAGE=<путь>." >&2
fi

# Если контейнер с таким именем уже существует, но не запущен — уберём его
# перед пересозданием (тот же паттерн, что в run.sh основного пайплайна).
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "[run] Удаляю старый контейнер $CONTAINER_NAME..."
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

echo "[run] Образ:      $IMAGE_TAG"
echo "[run] GPU:         $GPUS"
echo "[run] Порт (хост): $PORT -> 8501 (в контейнере)"
echo "[run] HF cache:    $HOST_HF_CACHE -> /root/.cache/huggingface"
if [[ ${#STORAGE_MOUNT_ARGS[@]} -gt 0 ]]; then
    echo "[run] Storage:     $HOST_STORAGE -> /mnt/storage-1 (в контейнере, read-only)"
fi
echo "[run] Аргументы app.py: $*"
echo
echo "[run] После старта пробрось порт со своего компьютера:"
echo "  ssh -N -L $PORT:localhost:$PORT user@<этот сервер>"
echo "  затем открой http://localhost:$PORT"
echo

# --shm-size: Playwright/Chromium (см. render.py) по умолчанию у Docker
# получает всего 64MB /dev/shm — без увеличения Chromium периодически падает
# с "Target crashed" (тот же нюанс, что в run.sh основного пайплайна).
# --gpus "$GPUS" — выбор ОДНОЙ конкретной GPU из нескольких на сервере,
# формат '"device=N"' (Docker понимает именно такую кавычку внутри значения).
docker run \
    --name "$CONTAINER_NAME" \
    --gpus "$GPUS" \
    --shm-size=1g \
    --ipc=host \
    -p "$PORT:8501" \
    -v "$HOST_HF_CACHE:/root/.cache/huggingface" \
    "${STORAGE_MOUNT_ARGS[@]}" \
    "$IMAGE_TAG" \
    "$@"
