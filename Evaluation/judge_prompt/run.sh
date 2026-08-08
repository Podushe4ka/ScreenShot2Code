#!/usr/bin/env bash
# run.sh — обёртка над `docker run` для judge-prompt-lab: монтирует данные
# (batch_00000 c картинками + labels csv) и выходную папку внутрь контейнера,
# пробрасывает GPU, передаёт остальные аргументы в run_judge_eval.py.
#
# Пример:
#   MODEL_JUDGE=Qwen/Qwen3.5-4B ./run.sh \
#       --data-root-host /path/to/folder/with/batch_00000 \
#       --labels-host ./labels_train.csv \
#       --prompt-key v1_baseline
#
# Всё, что не является --data-root-host/--labels-host/--out-host, пробрасывается
# как есть в run_judge_eval.py (например --max-tokens 256).
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-judge-prompt-lab:latest}"
MODEL_JUDGE="${MODEL_JUDGE:?Задайте MODEL_JUDGE, напр. Qwen/Qwen3.5-4B}"
GPU_JUDGE="${GPU_JUDGE:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
DATA_ROOT_HOST=""
LABELS_HOST=""
OUT_HOST="${OUT_HOST:-$(pwd)/output}"
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root-host) DATA_ROOT_HOST="$2"; shift 2 ;;
        --labels-host) LABELS_HOST="$2"; shift 2 ;;
        --out-host) OUT_HOST="$2"; shift 2 ;;
        *) PASSTHROUGH_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$DATA_ROOT_HOST" ]]; then
    echo "ОШИБКА: укажите --data-root-host /путь/к/папке/с/batch_00000" >&2
    exit 1
fi
if [[ -z "$LABELS_HOST" ]]; then
    echo "ОШИБКА: укажите --labels-host /путь/к/labels_train.csv (или test)" >&2
    exit 1
fi

mkdir -p "$OUT_HOST"

docker run --rm \
    --gpus "device=${GPU_JUDGE}" \
    -e MODEL_JUDGE="$MODEL_JUDGE" \
    -e GPU_JUDGE=0 \
    -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    -v "$DATA_ROOT_HOST:/data:ro" \
    -v "$(dirname "$(realpath "$LABELS_HOST")"):/labels:ro" \
    -v "$OUT_HOST:/app/output" \
    "$IMAGE_TAG" \
    --data-root /data \
    --labels "/labels/$(basename "$LABELS_HOST")" \
    "${PASSTHROUGH_ARGS[@]}"
