FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*\
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV TORCH_CUDA_ARCH_LIST="8.0"
ENV MAX_JOBS=4
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /build
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project \
 && uv sync --frozen --no-install-project --extra gpu --no-build-isolation
WORKDIR /workspace
CMD ["bash"]