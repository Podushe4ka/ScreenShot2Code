# Docker для параллельного конвертера (convert_parallel.py, production).
#
# База mcr.microsoft.com/playwright/python = chromium + системные либы (libasound и пр.) —
# не нужны sudo / `playwright install-deps`.
#
# Сборка (из КОРНЯ репозитория):
#     docker build -t ws-conv -f Data/drafting/Dockerfile .
# Запуск (код+датасет монтируем томом -> результат в Data/drafting/websight_drafting_pilot/ на хосте):
#     docker run --rm -v "$PWD":/work --shm-size=2g ws-conv --target 5000 --n-workers 32
#   быстрая проверка: ... ws-conv --target 200 --n-workers 16
#   с токеном:        docker run --rm -e HF_TOKEN=hf_xxx -v "$PWD":/work --shm-size=2g ws-conv
#
# Заметки: --shm-size=2g обязателен (Chromium/dev-shm); GPU не нужен; playwright пинаем под версию образа.

FROM mcr.microsoft.com/playwright/python:v1.55.0-noble

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
        datasets transformers pillow beautifulsoup4 pytailwindcss tqdm playwright==1.55.0 && \
    python -m playwright install chromium

# Код монтируется томом: -v "$PWD":/work (convert_parallel.py + convert_lib.py + ../analysis/token_len.py)
WORKDIR /work/Data/drafting

ENTRYPOINT ["python", "convert_parallel.py"]
CMD ["--target", "5000", "--n-workers", "32"]
