# Docker для параллельного конвертера WebSight -> формат контракта (production).
#
# База mcr.microsoft.com/playwright/python уже содержит chromium + системные либы
# (libasound и пр.) — не нужны sudo / `playwright install-deps`.
#
# Сборка (из КОРНЯ репозитория):
#     docker build -t ws-conv -f Data/Dockerfile .
#
# Запуск (код и датасет монтируем томом -> результат остаётся на хосте в Data/websight_drafting_pilot/):
#     docker run --rm -v "$PWD":/work --shm-size=2g ws-conv --target 5000 --n-workers 32
#   быстрая проверка:
#     docker run --rm -v "$PWD":/work --shm-size=2g ws-conv --target 200 --n-workers 16
#   с HF-токеном (если попросит лимиты):
#     docker run --rm -e HF_TOKEN=hf_xxx -v "$PWD":/work --shm-size=2g ws-conv
#
# Заметки:
#  - --shm-size=2g обязателен: Chromium любит /dev/shm, дефолтные 64 МБ дают краши.
#  - GPU не нужен (это рендер, а не матрицы) — без --gpus.
#  - playwright пинаем под версию образа (1.55.0), браузеры уже внутри.

FROM mcr.microsoft.com/playwright/python:v1.55.0-noble

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
        datasets transformers pillow beautifulsoup4 pytailwindcss tqdm playwright==1.55.0 && \
    python -m playwright install chromium

# Код монтируется томом: -v "$PWD":/work  (в нём Data/websight_to_contract_parallel.py + Data/analysis/token_len.py)
WORKDIR /work/Data

ENTRYPOINT ["python", "websight_to_contract_parallel.py"]
CMD ["--target", "5000", "--n-workers", "32"]
