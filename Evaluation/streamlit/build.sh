#!/usr/bin/env bash
# build.sh — собирает образ демо.
#
# Запуск:
#   ./build.sh                      # тег по умолчанию: screenshot2code-demo:latest
#   ./build.sh myname:v1            # свой тег
set -euo pipefail

IMAGE_TAG="${1:-screenshot2code-demo:latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Dockerfile ожидает COPY render.py app.py entrypoint.sh — проверяем ДО
# docker build, чтобы не получить невнятную ошибку "COPY failed" посреди
# сборки (тот же паттерн, что в build.sh основного бенчмарк-пайплайна).
required_files=(render.py app.py entrypoint.sh requirements.txt)
missing=()
for f in "${required_files[@]}"; do
    if [[ ! -f "$f" ]]; then
        missing+=("$f")
    fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
    echo "ОШИБКА: рядом с Dockerfile не найдены файлы: ${missing[*]}"
    echo "Положите их в $SCRIPT_DIR перед сборкой."
    exit 1
fi

echo "[build] Собираю образ $IMAGE_TAG ..."
docker build -t "$IMAGE_TAG" .
echo "[build] Готово: $IMAGE_TAG"
