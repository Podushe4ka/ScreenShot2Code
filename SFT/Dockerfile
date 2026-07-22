FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV TORCH_CUDA_ARCH_LIST="8.0"
ENV MAX_JOBS=4
# Compile from source instead of fetching a prebuilt wheel off GitHub Releases
# (GitHub is unreachable from the build network here). The devel image has nvcc.
ENV CAUSAL_CONV1D_FORCE_BUILD=TRUE
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /build
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project \
 && uv sync --frozen --no-install-project --extra gpu --no-build-isolation
WORKDIR /workspace
CMD ["bash"]