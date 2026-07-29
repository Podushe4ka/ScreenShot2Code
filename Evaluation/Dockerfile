FROM vllm/vllm-openai:latest

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    nodejs \
    npm \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "setuptools<81" \
    accelerate \
    pillow \
    datasets \
    huggingface_hub \
    pandas \
    tqdm \
    playwright \
    ftfy \
    regex \
    beautifulsoup4 \
    colormath \
    scipy \
    scikit-learn \
    joblib \
    opencv-python \
    git+https://github.com/openai/CLIP.git

RUN playwright install --with-deps chromium

COPY render.py metrics.py run_benchmark_batched.py ./

ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python3", "-u", "run_benchmark_batched.py"]
