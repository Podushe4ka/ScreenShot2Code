#!/usr/bin/env bash
# run.sh — запускает (или РЕЗЮМИРУЕТ) прогон бенчмарка в контейнере.
#
# Контейнеру достаточно одной GPU — внутри run_benchmark_batched.py сам
# поднимает и гасит один `vllm serve` процесс, переключая его между тремя
# моделями (checkpoint/baseline/judge) на каждом батче (см.
# vllm_server_manager.py и serve_models.sh — тонкий entrypoint, не место,
# где поднимаются серверы).
#
# Это один и тот же скрипт для первого запуска и для resume после падения:
# resume работает через outdir/progress.json внутри python-скрипта, так что
# просто перезапускаешь ту же команду и всё продолжится с того места, где
# упало (см. run_benchmark_batched.py). Отдельного "resume.sh" не нужно —
# см. также флаг --no-resume ниже, если нужно вместо этого начать с нуля.
#
# ВАЖНО про volume: --outdir и HF-кэш ОБЯЗАНЫ жить на хосте (bind mount), а
# не только внутри контейнера. Если контейнер после падения будет удалён
# (docker rm) или просто пересоздан, а outdir не был смонтирован — progress.json
# и накопленные examples/ исчезнут вместе с контейнером, и resume будет
# нечем резюмировать. Оба volume ниже монтируются с хоста именно поэтому.
#
# Примеры:
#   MODEL_CHECKPOINT=/mnt/storage-1/checkpoints/my-run/step-12000 \
#       ./run.sh --n-samples 2000 --batch-size 2000 --num-workers 112
#
#   # свой baseline вместо дефолтного Qwen/Qwen3.5-4B, конкретная GPU и порт:
#   MODEL_CHECKPOINT=/mnt/storage-1/checkpoints/qwen3.5-4b-step12000 \
#   MODEL_BASELINE=Qwen/Qwen3.5-4B \
#   GPU_INDEX=0 VLLM_PORT=8001 \
#       ./run.sh --n-samples 5000
#
#   # ниже конкурентность рендера, если таймауты рендера чекпоинта частые:
#   MODEL_CHECKPOINT=/mnt/storage-1/checkpoints/my-run RENDER_WORKERS=4 \
#       ./run.sh --n-samples 5000
#
#   ./run.sh --no-resume --n-samples 1000        # начать заново, игнорируя чекпоинт прогресса
#
# Переменные окружения (можно переопределить перед вызовом):
#   MODEL_CHECKPOINT        - ОБЯЗАТЕЛЬНО. Путь/имя модели-чекпоинта (см. HOST_STORAGE
#                             ниже про пути с общего диска).
#   MODEL_BASELINE          - baseline-модель для сравнения (по умолчанию Qwen/Qwen3.5-4B).
#   MODEL_JUDGE              - модель-судья (по умолчанию Qwen/Qwen3.5-4B).
#   GPU_INDEX                - индекс ЕДИНСТВЕННОЙ GPU (CUDA_VISIBLE_DEVICES), которую
#                             последовательно делят между собой checkpoint/baseline/judge
#                             (по умолчанию 0). CLIP-сервер тоже работает на этой GPU.
#   VLLM_PORT                - порт, на котором поднимается vllm serve — ОДИН и тот же
#                             порт переиспользуется для всех трёх моделей по очереди
#                             (по умолчанию 8001).
#   GPU_MEMORY_UTILIZATION  - gpu-memory-utilization для vllm serve (по умолчанию 0.85 —
#                             оставляет запас под CLIP-сервер, который держит часть VRAM
#                             той же карты постоянно, см. vllm_server_manager.py).
#   RENDER_WORKERS           - конкурентность browser-bound рендер-этапов (эталон+метрики
#                             чекпоинта, рендер baseline) — отдельно от --num-workers
#                             (I/O-bound judge). По умолчанию min(--num-workers, 8).
#                             Понизьте, если рендер чекпоинта часто уходит в таймаут.
#   RENDER_TIMEOUT_MS         - таймаут одного page.goto/screenshot внутри рендера, мс
#                             (по умолчанию 90000, см. render.DEFAULT_RENDER_TIMEOUT_MS).
#   IMAGE_TAG      - какой образ запускать (по умолчанию design2code-bench:latest)
#   HOST_OUTDIR    - куда на хосте класть результаты/чекпоинт (по умолчанию ./bench_results)
#   HOST_HF_CACHE  - куда на хосте класть кэш HF моделей/датасетов (по умолчанию ./hf_cache)
#   HOST_VLLM_CACHE - куда на хосте класть кэш компиляции vLLM (torch.compile/triton,
#                    ~/.cache/vllm внутри контейнера) — по умолчанию ~/.cache/vllm на
#                    хосте. Без этого volume компиляция пишется на overlay-слой
#                    контейнера и может упасть с "No space left on device", даже
#                    если места на хосте в целом достаточно.
#   TMPFS_SIZE     - размер /tmp внутри контейнера как tmpfs (RAM хоста), по умолчанию
#                    8g. /tmp — куда Chromium (Playwright) пишет временный профиль на
#                    каждый скриншот; если диск хоста (обычно /dev/vda1) занят другими
#                    процессами на сервере, запись профиля обрывается и Chromium падает
#                    нативно ("Target crashed"), из-за чего дальнейшие скриншоты в этом
#                    же CPU-воркере валятся с "Target page, context or browser has been
#                    closed" (браузер воркера остаётся мёртвым). tmpfs в RAM не зависит
#                    от занятости диска хоста. Задайте TMPFS_SIZE=0, чтобы отключить
#                    (тогда /tmp — обычный overlay, как раньше).
#   HOST_STORAGE   - путь к общему диску с чекпоинтами на ХОСТЕ (по умолчанию /mnt/storage-1).
#                    Монтируется В КОНТЕЙНЕР ПО ТОМУ ЖЕ ПУТИ (см. -v ниже), read-only —
#                    поэтому MODEL_CHECKPOINT можно указывать путём вида
#                    /mnt/storage-1/checkpoints/<run>/<step> напрямую, без пересчёта пути
#                    под контейнер. Если на хосте общий диск смонтирован не в /mnt/storage-1,
#                    задайте HOST_STORAGE=<реальный путь> — путь ВНУТРИ контейнера всё равно
#                    останется /mnt/storage-1, так что MODEL_CHECKPOINT из примеров не меняется.
#   GPUS           - какие GPU пробросить в docker run (по умолчанию all — ВСЕ GPU хоста;
#                    если хотите ограничить конкретной физической картой, задайте, например,
#                    GPUS='"device=0"' — тогда GPU_INDEX (0 по умолчанию) адресует её В
#                    ПРЕДЕЛАХ этого проброшенного набора, а не по глобальному индексу хоста)
#   CONTAINER_NAME - имя контейнера (по умолчанию design2code-bench)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_CHECKPOINT="${MODEL_CHECKPOINT:?Задайте MODEL_CHECKPOINT — путь/имя модели-чекпоинта, напр. MODEL_CHECKPOINT=/mnt/storage-1/... ./run.sh ...}"
MODEL_BASELINE="${MODEL_BASELINE:-Qwen/Qwen3.5-4B}"
MODEL_JUDGE="${MODEL_JUDGE:-Qwen/Qwen3.5-4B}"

GPU_INDEX="${GPU_INDEX:-0}"
VLLM_PORT="${VLLM_PORT:-8001}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
RENDER_WORKERS="${RENDER_WORKERS:-}"
RENDER_TIMEOUT_MS="${RENDER_TIMEOUT_MS:-}"

IMAGE_TAG="${IMAGE_TAG:-design2code-bench:latest}"
HOST_OUTDIR="${HOST_OUTDIR:-$SCRIPT_DIR/bench_results}"
HOST_HF_CACHE="${HOST_HF_CACHE:-$HOME/.cache/huggingface}"
# vLLM пишет компиляцию torch.compile/triton (inductor_cache) в ~/.cache/vllm
# ВНУТРИ контейнера (/root/.cache/vllm) — это НЕ то же самое, что HF-кэш выше.
# Без отдельного volume это пишется на overlay-слой контейнера, который часто
# маленький/ограниченный -> "No space left on device" даже при свободном месте
# на хосте в целом (см. HOST_VLLM_CACHE ниже).
HOST_VLLM_CACHE="${HOST_VLLM_CACHE:-$HOME/.cache/vllm}"
HOST_STORAGE="${HOST_STORAGE:-/mnt/storage-1}"
GPUS="${GPUS:-all}"
CONTAINER_NAME="${CONTAINER_NAME:-design2code-bench}"
# /tmp внутри контейнера по умолчанию живёт на overlay-слое, то есть на том же
# разделе диска хоста (обычно /dev/vda1), что и всё остальное на сервере —
# если он забит ДРУГИМИ процессами (не связанными с этим контейнером), Chromium
# (Playwright, см. render.py) не может дописать свой временный
# --user-data-dir=/tmp/playwright_... и падает нативным крашем ("Target
# crashed"/segfault в логе), после чего общий на CPU-воркер браузер остаётся
# мёртвым и ВСЕ дальнейшие скриншоты в этом воркере валятся с "Target page,
# context or browser has been closed" — внешне похоже на нехватку shm, но
# причина именно в диске (--ipc=host ниже уже даёт shm без лимита, так что
# на shm это не похоже). Монтируем /tmp контейнера как tmpfs (в RAM хоста) —
# независимо от того, сколько места на диске хоста уже заняли другие
# процессы. Задайте TMPFS_SIZE=0, чтобы отключить (тогда /tmp — обычный
# overlay, как раньше).
TMPFS_SIZE="${TMPFS_SIZE:-0}"

mkdir -p "$HOST_OUTDIR" "$HOST_HF_CACHE" "$HOST_VLLM_CACHE"

# HOST_STORAGE монтируется, только если реально существует на хосте — иначе
# либо MODEL_CHECKPOINT/MODEL_BASELINE берутся с HF hub, либо пользователь ещё
# не примонтировал общий диск на хосте, и лучше явно сообщить об этом, чем
# тихо стартовать без него и упасть непонятной ошибкой "path not found" уже
# внутри vLLM.
STORAGE_MOUNT_ARGS=()
if [[ -d "$HOST_STORAGE" ]]; then
    STORAGE_MOUNT_ARGS=(-v "$HOST_STORAGE:/mnt/storage-1:ro")
else
    echo "[run] Внимание: $HOST_STORAGE не найден на хосте — общий диск с чекпоинтами" >&2
    echo "не будет примонтирован. Если MODEL_CHECKPOINT указывает на /mnt/storage-1/..., это упадёт." >&2
    echo "Если диск смонтирован в другом месте, задайте HOST_STORAGE=<путь>." >&2
fi

TMPFS_MOUNT_ARGS=()
if [[ "$TMPFS_SIZE" != "0" ]]; then
    TMPFS_MOUNT_ARGS=(--tmpfs "/tmp:size=$TMPFS_SIZE,exec")
fi

# Защита: --outdir всегда должен указывать на /app/output (единственный путь,
# смонтированный volume-ом с хоста) — та же причина, что и раньше: свой
# --outdir в "$@" окажется внутри контейнера без volume и после docker rm/
# падения все результаты и progress.json потеряются без единого предупреждения.
for arg in "$@"; do
    if [[ "$arg" == "--outdir" || "$arg" == --outdir=* ]]; then
        echo "ОШИБКА: не передавайте --outdir в run.sh — путь внутри контейнера" >&2
        echo "всегда фиксирован на /app/output. Чтобы изменить место на ХОСТЕ," >&2
        echo "задайте переменную HOST_OUTDIR перед запуском, например:" >&2
        echo "  HOST_OUTDIR=/data/my_run ./run.sh --n-samples 70000" >&2
        exit 1
    fi
    # Аналогично --model-checkpoint/--model-baseline/--judge-model/--gpu-index/
    # --vllm-port/--render-workers/--render-timeout-ms: они выводятся из
    # MODEL_CHECKPOINT/MODEL_BASELINE/MODEL_JUDGE/GPU_INDEX/VLLM_PORT/
    # RENDER_WORKERS/RENDER_TIMEOUT_MS и передаются serve_models.sh
    # автоматически (см. -e ниже) — передать их ЕЩЁ РАЗ через "$@" значило бы
    # продублировать argparse-флаг, и (как и с --outdir) молча победит
    # последний, что запутывает, откуда на самом деле взялось значение
    # (переменная окружения или явный флаг в "$@").
    if [[ "$arg" == "--model-checkpoint"* || "$arg" == "--model-baseline"* || "$arg" == "--judge-model"* \
          || "$arg" == "--gpu-index"* || "$arg" == "--vllm-port"* || "$arg" == "--vllm-gpu-memory-utilization"* \
          || "$arg" == "--render-workers"* || "$arg" == "--render-timeout-ms"* ]]; then
        echo "ОШИБКА: не передавайте $arg напрямую в run.sh — это значение" >&2
        echo "выводится из переменной окружения (MODEL_CHECKPOINT/MODEL_BASELINE/MODEL_JUDGE/" >&2
        echo "GPU_INDEX/VLLM_PORT/GPU_MEMORY_UTILIZATION/RENDER_WORKERS/RENDER_TIMEOUT_MS) внутри" >&2
        echo "serve_models.sh автоматически. Задайте, например:" >&2
        echo "  MODEL_CHECKPOINT=/mnt/storage-1/... ./run.sh --n-samples 5000" >&2
        exit 1
    fi
done

# Если контейнер с таким именем уже существует, но не запущен (например,
# после docker stop) — уберём его перед пересозданием, иначе `docker run
# --name` откажется стартовать с "the container name is already in use".
# Volume-данные (outdir/hf_cache) при этом не трогаются - они на хосте.
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "[run] Удаляю старый контейнер $CONTAINER_NAME (данные в volume сохранены)..."
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

echo "[run] Образ:            $IMAGE_TAG"
echo "[run] GPU (1 карта):    индекс $GPU_INDEX, порт $VLLM_PORT, gpu-memory-utilization $GPU_MEMORY_UTILIZATION"
echo "[run] Чекпоинт:         $MODEL_CHECKPOINT"
echo "[run] Baseline:         $MODEL_BASELINE"
echo "[run] Judge:            $MODEL_JUDGE"
echo "[run]                   (checkpoint -> baseline -> judge грузятся ПО ОЧЕРЕДИ на каждом батче)"
echo "[run] outdir (хост):    $HOST_OUTDIR  ->  /app/output (в контейнере)"
echo "[run] HF cache:         $HOST_HF_CACHE  ->  /root/.cache/huggingface"
echo "[run] vLLM cache:       $HOST_VLLM_CACHE  ->  /root/.cache/vllm"
if [[ ${#STORAGE_MOUNT_ARGS[@]} -gt 0 ]]; then
    echo "[run] Storage:          $HOST_STORAGE  ->  /mnt/storage-1 (в контейнере, read-only)"
fi
echo "[run] GPU (docker run): $GPUS"
if [[ ${#TMPFS_MOUNT_ARGS[@]} -gt 0 ]]; then
    echo "[run] /tmp:             tmpfs в RAM хоста, size=$TMPFS_SIZE (см. TMPFS_SIZE)"
else
    echo "[run] /tmp:             обычный overlay (TMPFS_SIZE=0) — делит место с диском хоста"
fi
echo "[run] Аргументы скрипту: $* "
echo

# --shm-size: и vLLM (KV-cache), и Playwright/Chromium (см. render.py —
# --disable-gpu, но /dev/shm по умолчанию у Docker всего 64MB) требуют
# больше дефолтного /dev/shm; --ipc=host ниже уже снимает этот лимит
# полностью (shm-size контейнера в этом режиме не действует), --shm-size=2g
# оставлен как явный fallback на случай запуска без --ipc=host.
#
# --tmpfs /tmp (см. TMPFS_MOUNT_ARGS выше): решает ДРУГУЮ проблему — нехватку
# места на ДИСКЕ хоста (/dev/vda1), если его забили другие процессы на
# сервере, не связанные с этим контейнером. /tmp в RAM не зависит от
# состояния диска хоста вообще.
#
# -e MODEL_CHECKPOINT/... и GPU_INDEX/VLLM_PORT/... — serve_models.sh
# (ENTRYPOINT образа) читает их из окружения контейнера напрямую (см.
# serve_models.sh), это не аргументы python-скрипта напрямую — serve_models.sh
# сам транслирует их в --gpu-index/--vllm-port/... при вызове python.
docker run \
    --name "$CONTAINER_NAME" \
    --gpus "$GPUS" \
    --shm-size=2g \
    --ipc=host \
    "${TMPFS_MOUNT_ARGS[@]}" \
    -e MODEL_CHECKPOINT="$MODEL_CHECKPOINT" \
    -e MODEL_BASELINE="$MODEL_BASELINE" \
    -e MODEL_JUDGE="$MODEL_JUDGE" \
    -e GPU_INDEX="$GPU_INDEX" \
    -e VLLM_PORT="$VLLM_PORT" \
    -e GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
    -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    -e RENDER_WORKERS="$RENDER_WORKERS" \
    -e RENDER_TIMEOUT_MS="$RENDER_TIMEOUT_MS" \
    -v "$HOST_OUTDIR:/app/output" \
    -v "$HOST_HF_CACHE:/root/.cache/huggingface" \
    -v "$HOST_VLLM_CACHE:/root/.cache/vllm" \
    "${STORAGE_MOUNT_ARGS[@]}" \
    "$IMAGE_TAG" \
    --outdir /app/output \
    "$@"
