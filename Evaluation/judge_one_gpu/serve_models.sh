#!/usr/bin/env bash
# serve_models.sh — ENTRYPOINT образа.
#
# Тонкий entrypoint: готовит окружение (лог-уровни, дефолты переменных) и
# передаёт управление run_benchmark_batched.py, который сам стартует и
# останавливает `vllm serve` по ходу батчей — на каждом батче переключаясь
# между checkpoint, baseline и judge (subprocess.Popen + health-check + kill,
# см. vllm_server_manager.VLLMServerManager.switch_to). Сам serve_models.sh
# не поднимает vLLM.
#
# Использование:
#   ./serve_models.sh                    # запускает run_benchmark_batched.py
#   ./serve_models.sh --n-samples 5000   # доп. аргументы пробрасываются как есть
#
# Переменные окружения (можно переопределить перед вызовом):
#   MODEL_CHECKPOINT          - путь/имя модели-чекпоинта (обязательно)
#   MODEL_BASELINE             - путь/имя baseline-модели (по умолчанию Qwen/Qwen3.5-4B)
#   MODEL_JUDGE                 - путь/имя модели-судьи (по умолчанию тоже Qwen/Qwen3.5-4B)
#   GPU_INDEX                   - индекс GPU (CUDA_VISIBLE_DEVICES) для единственного
#                                 vllm serve процесса, используется последовательно
#                                 всеми тремя моделями (по умолчанию 0)
#   VLLM_PORT                   - порт vllm serve, тоже один на все три модели по
#                                 очереди (по умолчанию 8001)
#   GPU_MEMORY_UTILIZATION       - gpu-memory-utilization для vllm serve. Оставляет
#                                 запас под CLIP-сервер, который держит часть VRAM
#                                 той же карты постоянно (по умолчанию 0.85, см.
#                                 vllm_server_manager.py).
#   MAX_MODEL_LEN                 - max-model-len (по умолчанию 16384)
#   RENDER_WORKERS                 - конкурентность browser-bound рендер-этапов
#                                 (по умолчанию: min(num-workers, 8), см. --render-workers)
#   RENDER_TIMEOUT_MS               - таймаут одного page.goto/screenshot, мс
#   CHECKPOINT_LOG_LEVEL/BASELINE_LOG_LEVEL/JUDGE_LOG_LEVEL - VLLM_LOGGING_LEVEL
#     для каждого из трёх запусков (checkpoint/baseline: INFO по умолчанию,
#     judge: DEBUG по умолчанию — см. пояснение ниже про EngineDeadError).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_CHECKPOINT="${MODEL_CHECKPOINT:?Задайте MODEL_CHECKPOINT (путь к чекпоинту, напр. /mnt/storage-1/...)}"
MODEL_BASELINE="${MODEL_BASELINE:-Qwen/Qwen3.5-4B}"
MODEL_JUDGE="${MODEL_JUDGE:-Qwen/Qwen3.5-4B}"

GPU_INDEX="${GPU_INDEX:-0}"
VLLM_PORT="${VLLM_PORT:-8001}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

# Лог-директория (server_logs/{checkpoint,baseline,judge}.log) создаётся и
# пишется из python (VLLMServerManager, log_dir=outdir/"server_logs") — не
# здесь. outdir внутри контейнера фиксирован на /app/output (см. run.sh), так
# что логи окажутся в /app/output/server_logs/*.log.

# Судья уже падал с EngineDeadError без единого traceback в логе — на
# INFO-уровне (дефолт vLLM) EngineCore не обязательно печатает полный стек
# необработанного исключения перед смертью. DEBUG у checkpoint/baseline не
# включаем по умолчанию — там нет истории крашей, а DEBUG сильно раздувает
# лог при высокой конкурентности (--generation-concurrency). Эти три
# переменные читает сам run_benchmark_batched.py (через os.environ) при
# каждом switch_to — передавать их явным аргументом не нужно.
export CHECKPOINT_LOG_LEVEL="${CHECKPOINT_LOG_LEVEL:-INFO}"
export BASELINE_LOG_LEVEL="${BASELINE_LOG_LEVEL:-INFO}"
export JUDGE_LOG_LEVEL="${JUDGE_LOG_LEVEL:-DEBUG}"
export D2C_CHECKPOINT_LOG_LEVEL="$CHECKPOINT_LOG_LEVEL"
export D2C_BASELINE_LOG_LEVEL="$BASELINE_LOG_LEVEL"
export D2C_JUDGE_LOG_LEVEL="$JUDGE_LOG_LEVEL"

# Опциональные переопределения конкурентности/таймаута рендера — если не
# заданы, run_benchmark_batched.py использует свои дефолты (см. parse_args).
EXTRA_RENDER_ARGS=()
if [[ -n "${RENDER_WORKERS:-}" ]]; then
    EXTRA_RENDER_ARGS+=(--render-workers "$RENDER_WORKERS")
fi
if [[ -n "${RENDER_TIMEOUT_MS:-}" ]]; then
    EXTRA_RENDER_ARGS+=(--render-timeout-ms "$RENDER_TIMEOUT_MS")
fi

echo "[serve_models] GPU $GPU_INDEX, порт $VLLM_PORT, "
echo "[serve_models] модели переключаются последовательно по ходу батчей: "
echo "[serve_models]   checkpoint=$MODEL_CHECKPOINT baseline=$MODEL_BASELINE judge=$MODEL_JUDGE"
echo "[serve_models] Логи каждого запуска модели — в /app/output/server_logs/{checkpoint,baseline,judge}.log"

exec python3 -u "$SCRIPT_DIR/run_benchmark_batched.py" \
    --model-checkpoint "$MODEL_CHECKPOINT" \
    --model-baseline "$MODEL_BASELINE" \
    --judge-model "$MODEL_JUDGE" \
    --gpu-index "$GPU_INDEX" \
    --vllm-port "$VLLM_PORT" \
    --vllm-gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    "${EXTRA_RENDER_ARGS[@]}" \
    "$@"
