#!/usr/bin/env bash
# build.sh — собирает образ для judge-экспериментов.
#
# Запуск:
#   ./build.sh                       # тег по умолчанию: judge-prompt-lab:latest
#   ./build.sh myname:v1             # свой тег
set -euo pipefail

IMAGE_TAG="${1:-judge-prompt-lab:latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

required_files=(vllm_client.py judge_client.py sample_order.py prompts.py run_judge_eval.py split_data.py serve_judge.sh)
missing=()
for f in "${required_files[@]}"; do
    if [[ ! -f "$f" ]]; then
        missing+=("$f")
    fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
    echo "ОШИБКА: рядом с Dockerfile не найдены файлы: ${missing[*]}"
    exit 1
fi

echo "[build] Собираю образ $IMAGE_TAG ..."
docker build -t "$IMAGE_TAG" .
echo "[build] Готово: $IMAGE_TAG"
