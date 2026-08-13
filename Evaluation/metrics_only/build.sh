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

# Список обязан совпадать со строкой COPY в Dockerfile. Проверяем ДО docker build,
# иначе получаем невнятное "COPY failed: file not found" посреди долгой сборки.
required_files=(render.py metrics.py clip_server.py run_benchmark_batched.py tracking.py)
missing=()
for f in "${required_files[@]}"; do
    if [[ ! -f "$f" ]]; then
        missing+=("$f")
    fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
    echo "ОШИБКА: рядом с Dockerfile не найдены файлы: ${missing[*]}"
    echo "Положите их в $SCRIPT_DIR перед сборкой либо уберите из COPY в Dockerfile."
    exit 1
fi

echo "[build] Собираю образ $IMAGE_TAG ..."
docker build -t "$IMAGE_TAG" .
echo "[build] Готово: $IMAGE_TAG"
