FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential ninja-build \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*


RUN python3 -m pip install --no-cache-dir uv
ENV TORCH_CUDA_ARCH_LIST="8.0"
ENV MAX_JOBS=4

ENV CAUSAL_CONV1D_FORCE_BUILD=TRUE
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

ENV UV_PYTHON_DOWNLOADS=never
WORKDIR /build
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project \
 && uv sync --frozen --no-install-project --extra gpu --no-build-isolation
WORKDIR /workspace
CMD ["bash"]