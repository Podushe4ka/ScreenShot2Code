#!/usr/bin/env bash
# Запуск контейнера с SFT-окружением.
#
#   ./run.sh                                   # интерактивный bash
#   ./run.sh python -m scripts.smoke_test      # разовая команда
#   CLEARML_TASK=qwen3_5_4b_lora ./run.sh
#   GPUS='"device=0,1"' ./run.sh
#   DATA_DIR=/mnt/storage-1/data ./run.sh       # каталог с датасетами -> /data
#   RUN_AS_USER=1 ./run.sh                     # от uid/gid хоста, а не root
#
# По умолчанию контейнер работает от root: эксперименты запускает один человек,
# и файлы, созданные root-ом, ему не мешают. RUN_AS_USER=1 нужен, если к
# чекпоинтам и кэшу HF будет ходить кто-то ещё — тогда они создаются от
# текущего пользователя. Учтите: смешивать режимы нельзя, root-запуск оставляет
# в кэше файлы, которые потом не перезапишет обычный uid.
set -euo pipefail

cd "$(dirname "$0")"

ENV_FILE="${ENV_FILE:-.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  echo "[run] загружен $ENV_FILE"
fi

PROJECT_SUBDIR="$(basename "$PWD")"
if [[ -d "../.git" ]]; then
  MOUNT_SRC="$(cd .. && pwd)"
  WORKDIR="/workspace/$PROJECT_SUBDIR"
else
  MOUNT_SRC="$PWD"
  WORKDIR="/workspace"
fi

IMAGE="${IMAGE:-sft}"
GPUS="${GPUS:-all}"
SHM_SIZE="${SHM_SIZE:-16g}"

CONTAINER_HOME="${CONTAINER_HOME:-$PWD/.container-home}"
mkdir -p "$CONTAINER_HOME" "$CONTAINER_HOME/tmp"

HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

OUT_DIR="${OUT_DIR:-}"
if [[ -n "$OUT_DIR" ]]; then
  mkdir -p "$OUT_DIR"
fi

DATA_DIR="${DATA_DIR:-/mnt/storage-1/data}"
if [[ ! -d "$DATA_DIR" ]]; then
  echo "ОШИБКА: каталога с данными нет: $DATA_DIR" >&2
  echo "Задайте DATA_DIR=/путь/к/данным перед запуском." >&2
  exit 1
fi

args=(
  --gpus "$GPUS"
  --rm
  --shm-size="$SHM_SIZE"
  -v "$MOUNT_SRC":/workspace
  -v "$CONTAINER_HOME":/container-home
  -v "$HF_CACHE":/hf-cache
  -v "$DATA_DIR":/data
  -w "$WORKDIR"
)

if [[ -n "$OUT_DIR" ]]; then
  args+=(-v "$OUT_DIR":/out)
fi

args+=(
  -e HOME=/container-home
  -e HF_HOME=/hf-cache
  -e TMPDIR=/container-home/tmp
  -e TORCHINDUCTOR_CACHE_DIR=/container-home/torchinductor
  -e TRITON_CACHE_DIR=/container-home/triton
  -e PATH=/opt/venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  -e PYTHONFAULTHANDLER=1
  -e CLEARML_API_HOST="${CLEARML_API_HOST:-https://api.clear.ml}"
  -e CLEARML_PROJECT="${CLEARML_PROJECT:-Screenshot2Code}"
  -e CLEARML_LOG_MODEL="${CLEARML_LOG_MODEL:-FALSE}"
)

if [[ "${RUN_AS_USER:-0}" == "1" ]]; then
  args+=(--user "$(id -u):$(id -g)")
  args+=(-e USER="$(id -un)" -e LOGNAME="$(id -un)")
  if grep -q "^[^:]*:[^:]*:$(id -u):" /etc/passwd 2>/dev/null; then
    args+=(-v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro)
  fi
fi

for var in HF_TOKEN CLEARML_API_ACCESS_KEY CLEARML_API_SECRET_KEY \
           CLEARML_WEB_HOST CLEARML_FILES_HOST CLEARML_TASK CUDA_VISIBLE_DEVICES \
           RESULT_DIR DATASET CONFIG NPROC ONLY STEPS SKIP_EXISTING; do
  if [[ -n "${!var:-}" ]]; then
    args+=(-e "$var=${!var}")
  fi
done

if [[ -t 0 && -t 1 ]]; then
  args+=(-it)
else
  args+=(-i)
fi

exec docker run "${args[@]}" "$IMAGE" "${@:-bash}"
