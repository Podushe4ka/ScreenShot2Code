FROM vllm/vllm-openai:latest

WORKDIR /app

# git: нужен для `pip install git+https://github.com/openai/CLIP.git` ниже.
# nodejs/npm/git-lfs убраны — ни один файл проекта их не вызывает (были
# рудиментом от более ранней версии образа).
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Только пакеты, которые реально импортируются кодом проекта:
#   pillow (PIL) — render.py, metrics.py
#   datasets — run_benchmark_batched.py (стриминг HF-датасета); тянет
#     huggingface_hub транзитивно, отдельно его ставить не нужно
#   pandas, tqdm — run_benchmark_batched.py
#   playwright — render.py
#   beautifulsoup4 (bs4) — render.py, metrics.py
#   colormath, scipy — metrics.py
#   opencv-python (cv2) — metrics.py, clip_server.py
#   requests — vllm_client.py
#   git+CLIP.git — metrics.py, clip_server.py; сам тянет ftfy/regex/tqdm/
#     torch/torchvision транзитивно (см. его requirements.txt), отдельно
#     их указывать не нужно.
# Убраны (не импортируются нигде в коде): accelerate, scikit-learn, joblib.
RUN pip install --no-cache-dir \
    "setuptools<81" \
    pillow \
    datasets \
    pandas \
    tqdm \
    playwright \
    beautifulsoup4 \
    colormath \
    scipy \
    opencv-python \
    requests \
    git+https://github.com/openai/CLIP.git

RUN playwright install --with-deps chromium

COPY render.py metrics.py clip_server.py vllm_client.py vllm_server_manager.py judge_client.py run_benchmark_batched.py serve_models.sh ./
RUN chmod +x serve_models.sh

ENV HF_HOME=/root/.cache/huggingface

ENV PYTHONUNBUFFERED=1

# serve_models.sh — тонкий entrypoint, готовит окружение и сразу передаёт
# управление run_benchmark_batched.py, который сам поднимает и гасит один
# vllm serve процесс, переключая его между checkpoint/baseline/judge на
# каждом батче (см. vllm_server_manager.py). Аргументы run.sh
# (--n-samples, --model-checkpoint и т.д.) пробрасываются как есть в конец
# команды serve_models.sh -> run_benchmark_batched.py.
ENTRYPOINT ["./serve_models.sh"]
