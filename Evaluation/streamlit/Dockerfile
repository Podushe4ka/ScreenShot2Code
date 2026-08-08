# Dockerfile для Streamlit-демо ScreenShot2Code.
#
# Базовый образ — обычный CUDA runtime (не vllm/vllm-openai, как в основном
# бенчмарк-пайплайне): здесь инференс идёт через transformers на одном
# запросе за раз, vLLM не нужен. Под A100 80GB — bfloat16 без каких-либо
# спецсборок PyTorch (в отличие от V100 в основном пайплайне, где нужен был
# cu126 + float16 — здесь этого ограничения нет).
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# python3 -> python, чтобы дальнейшие вызовы (playwright install и т.п.)
# не зависели от того, какой бинарь есть в конкретном базовом образе.
RUN ln -sf /usr/bin/python3 /usr/bin/python

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Playwright/Chromium — тот же движок рендера, что в основном
# бенчмарк-пайплайне (render.py копируется 1-в-1 ниже).
RUN playwright install --with-deps chromium

COPY render.py app.py ./

EXPOSE 8501

# --server.address 0.0.0.0 нужен, чтобы порт был слушаем СНАРУЖИ контейнера
# (доступ снаружи хоста всё равно ограничивается тем, что наружу с сервера
# порт не пробрасывается — только через SSH-туннель, см. README.md).
# --model и прочие параметры app.py передаются через "$@" в entrypoint.sh,
# то есть при `docker run ... IMAGE --model ... --device cuda` они уходят
# именно в app.py, а не в streamlit run.
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
