#!/usr/bin/env bash
# serve_judge.sh — поднимает ОДИН vLLM OpenAI-совместимый сервер с
# judge-моделью, ждёт готовности, затем запускает run_judge_eval.py с
# переданными аргументами. В отличие от serve_models.sh из основного
# eval-пайплайна (три сервера — checkpoint/baseline/judge), здесь нужна
# только judge-модель: картинки-кандидаты уже сгенерированы заранее и лежат
# на диске, генерировать заново не нужно.
#
# Использование (внутри контейнера, обычно как ENTRYPOINT):
#   MODEL_JUDGE=Qwen/Qwen3.5-4B ./serve_judge.sh \
#       --data-root /data --labels /data/labels_train.csv --prompt-key v1_baseline
#
# Переменные окружения:
#   MODEL_JUDGE             - путь/имя judge-модели (обязательно). Для линейки
#                              3.5 подставляй нужный размер, напр.:
#                                Qwen/Qwen3.5-2B, Qwen/Qwen3.5-4B,
#                                Qwen/Qwen3.5-9B, Qwen/Qwen3.5-27B
#   GPU_JUDGE                - индекс GPU (по умолчанию 0 — тут всего одна модель,
#                              не нужно разносить по нескольким GPU, как в
#                              основном пайплайне с тремя одновременными моделями)
#   PORT_JUDGE                - порт сервера (по умолчанию 8001)
#   GPU_MEMORY_UTILIZATION   - gpu-memory-utilization (по умолчанию 0.9)
#   MAX_MODEL_LEN             - max-model-len (по умолчанию 16384)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_JUDGE="${MODEL_JUDGE:?Задайте MODEL_JUDGE, напр. Qwen/Qwen3.5-4B}"
GPU_JUDGE="${GPU_JUDGE:-0}"
PORT_JUDGE="${PORT_JUDGE:-8001}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
LOG_DIR="${LOG_DIR:-/app/output/server_logs}"
mkdir -p "$LOG_DIR"

SERVER_PID=""

cleanup() {
    echo "[serve_judge] Останавливаю vLLM-сервер..."
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "[serve_judge] Запускаю judge: модель=$MODEL_JUDGE, GPU=$GPU_JUDGE, порт=$PORT_JUDGE"
CUDA_VISIBLE_DEVICES="$GPU_JUDGE" vllm serve "$MODEL_JUDGE" \
    --port "$PORT_JUDGE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    --limit-mm-per-prompt '{"image": 3}' \
    > "$LOG_DIR/judge.log" 2>&1 &
SERVER_PID=$!

# Быстрый отказ, если сервер упал сразу после старта (битый путь к модели,
# OOM при загрузке) — не ждать полный health-check таймаут вслепую.
sleep 5
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ОШИБКА: judge-сервер (pid $SERVER_PID) упал сразу после старта." >&2
    echo "Последние строки лога ($LOG_DIR/judge.log):" >&2
    tail -n 40 "$LOG_DIR/judge.log" >&2 2>/dev/null || true
    exit 1
fi

echo "[serve_judge] Сервер запущен (лог в $LOG_DIR/judge.log)."
echo "[serve_judge] Передаю управление run_judge_eval.py (сам дождётся /health)..."

python3 -u "$SCRIPT_DIR/run_judge_eval.py" \
    --judge-url "http://127.0.0.1:$PORT_JUDGE" \
    --judge-model "$MODEL_JUDGE" \
    "$@"

# cleanup() сработает автоматически через trap EXIT после run_judge_eval.py.
