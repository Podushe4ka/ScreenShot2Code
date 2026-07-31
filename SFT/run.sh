#!/usr/bin/env bash
# Запуск контейнера с SFT-окружением.
#
#   ./run.sh                                   # интерактивный bash
#   ./run.sh python -m scripts.smoke_test      # разовая команда
#   CLEARML_TASK=qwen3_5_4b_lora ./run.sh
#   GPUS='"device=0,1"' ./run.sh
#   RUN_AS_ROOT=1 ./run.sh                     # без подмены uid/gid
#
# Контейнер работает от uid/gid хоста, поэтому чекпоинты и логи в /workspace
# создаются с правами текущего пользователя, а не root.
set -euo pipefail

cd "$(dirname "$0")"

IMAGE="${IMAGE:-sft}"
GPUS="${GPUS:-all}"
SHM_SIZE="${SHM_SIZE:-16g}"

CONTAINER_HOME="${CONTAINER_HOME:-$PWD/.container-home}"
mkdir -p "$CONTAINER_HOME"

# Кэш HF с хоста. Под non-root /root/.cache недоступен, поэтому монтируем в
# отдельную точку и указываем на неё через HF_HOME.
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

args=(
  --gpus "$GPUS"
  --rm
  --shm-size="$SHM_SIZE"
  -v "$PWD":/workspace
  -v "$CONTAINER_HOME":/container-home
  -v "$HF_CACHE":/hf-cache
  -w /workspace
  -e HOME=/container-home
  -e HF_HOME=/hf-cache
  # /opt/venv в PATH, чтобы работали python/torchrun без полного пути.
  # Остальное — PATH базового образа nvidia/cuda (нужен nvcc).
  -e PATH=/opt/venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  -e CLEARML_API_HOST="${CLEARML_API_HOST:-https://api.clear.ml}"
  -e CLEARML_PROJECT="${CLEARML_PROJECT:-Screenshot2Code}"
  -e CLEARML_TASK="${CLEARML_TASK:-qwen3_5_4b_full_ft}"
  -e CLEARML_LOG_MODEL="${CLEARML_LOG_MODEL:-FALSE}"
)

if [[ "${RUN_AS_ROOT:-0}" != "1" ]]; then
  args+=(--user "$(id -u):$(id -g)")
fi

# Секреты пробрасываем только если заданы, иначе внутри окажется пустая
# переменная, которая ломает автологин HF/ClearML вместо fallback на конфиг.
for var in HF_TOKEN CLEARML_API_ACCESS_KEY CLEARML_API_SECRET_KEY CUDA_VISIBLE_DEVICES; do
  if [[ -n "${!var:-}" ]]; then
    args+=(-e "$var=${!var}")
  fi
done

# -t только при реальном терминале, иначе ./run.sh не запустить из скрипта/CI.
if [[ -t 0 && -t 1 ]]; then
  args+=(-it)
else
  args+=(-i)
fi

exec docker run "${args[@]}" "$IMAGE" "${@:-bash}"
