#!/usr/bin/env bash
# Сборка образа для SFT-трека. Один раз, ~30-40 мин: компилируется flash-attn.
#
#   ./build.sh              # тег sft
#   IMAGE=sft:cu126 ./build.sh
#   ./build.sh --no-cache   # любые доп. флаги уходят в docker build
set -euo pipefail

cd "$(dirname "$0")"

IMAGE="${IMAGE:-sft}"

docker build -t "$IMAGE" -f Dockerfile "$@" .

echo
echo "Готово: $IMAGE"
echo "Дальше: ./run.sh"
