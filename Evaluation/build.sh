#!/usr/bin/env bash
# build.sh — собирает образ бенчмарка.
#
# Запуск:
#   ./build.sh                 # тег по умолчанию: design2code-bench:latest
#   ./build.sh myname:v2       # свой тег
set -euo pipefail

IMAGE_TAG="${1:-design2code-bench:latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Dockerfile ожидает COPY render.py metrics.py
# run_benchmark_batched.py — проверяем ДО docker build, а не
# получаем невнятную ошибку "COPY failed: file not found" посреди сборки.
required_files=(render.py metrics.py run_benchmark_batched.py tracking.py)
missing=()
for f in "${required_files[@]}"; do
    if [[ ! -f "$f" ]]; then
        missing+=("$f")
    fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
    echo "ОШИБКА: рядом с Dockerfile не найдены файлы: ${missing[*]}"
    echo "Положите их в $SCRIPT_DIR перед сборкой (или уберите test.py из COPY в Dockerfile, если он не нужен)."
    exit 1
fi

echo "[build] Собираю образ $IMAGE_TAG ..."
docker build -t "$IMAGE_TAG" .
echo "[build] Готово: $IMAGE_TAG"
