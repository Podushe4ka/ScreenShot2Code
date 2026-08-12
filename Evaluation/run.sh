#!/usr/bin/env bash
# run.sh — запускает (или РЕЗЮМИРУЕТ) прогон бенчмарка в контейнере.
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
#   ./run.sh --n-samples 70000 --batch-size 10000
#   ./run.sh --n-samples 70000 --num-workers 4 --tensor-parallel-size 2
#   ./run.sh --no-resume --n-samples 1000        # начать заново, игнорируя чекпоинт
#   ./run.sh --model /mnt/storage-1/checkpoints/qwen3.5-9b-step12000 --n-samples 5000
#                                                  # чекпоинт с общего диска (см. HOST_STORAGE ниже)
#
# Переменные окружения (можно переопределить перед вызовом):
#   IMAGE_TAG      - какой образ запускать (по умолчанию design2code-bench:latest)
#   HOST_OUTDIR    - куда на хосте класть результаты/чекпоинт (по умолчанию ./bench_results)
#   HOST_HF_CACHE  - куда на хосте класть кэш HF моделей/датасетов (по умолчанию ./hf_cache)
#   HOST_STORAGE   - путь к общему диску с чекпоинтами на ХОСТЕ (по умолчанию /mnt/storage-1).
#                    Монтируется В КОНТЕЙНЕР ПО ТОМУ ЖЕ ПУТИ (см. -v ниже), read-only —
#                    поэтому --model можно передавать с путём вида
#                    /mnt/storage-1/checkpoints/<run>/<step> напрямую, без пересчёта пути
#                    под контейнер. Если на хосте общий диск смонтирован не в /mnt/storage-1,
#                    задайте HOST_STORAGE=<реальный путь> — путь ВНУТРИ контейнера всё равно
#                    останется /mnt/storage-1, так что --model из примеров выше не меняется.
#   GPUS           - какие GPU пробросить (по умолчанию all)
#   CONTAINER_NAME - имя контейнера. По умолчанию design2code-bench-$(id -un):
#                    имя ПЕРСОНАЛЬНОЕ, потому что машины общие — см. ниже.
#   SHM_SIZE       - размер /dev/shm контейнера (по умолчанию 4g). Поднимать при
#                    большом --num-workers: Chromium падает с "Target crashed".
#   HOST_MODEL_DIR - каталог с чекпоинтом вне HF-кэша и вне HOST_STORAGE;
#                    монтируется в контейнер по тому же пути, read-only.
#   FORCE_RM       - 1, чтобы снести РАБОТАЮЩИЙ контейнер-тёзку (по умолчанию отказ).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_TAG="${IMAGE_TAG:-design2code-bench:latest}"
HOST_OUTDIR="${HOST_OUTDIR:-$SCRIPT_DIR/bench_results}"
HOST_HF_CACHE="${HOST_HF_CACHE:-$HOME/.cache/huggingface}"
HOST_STORAGE="${HOST_STORAGE:-/mnt/storage-1}"
GPUS="${GPUS:-all}"
# Имя контейнера по умолчанию — С ИМЕНЕМ ПОЛЬЗОВАТЕЛЯ. Прежний дефолт был
# просто `design2code-bench`, то есть один и тот же у всех, кто запускает бенч
# из этого репозитория на общей машине. Ниже стоит `docker rm -f` по точному
# совпадению имени — значит второй запустившийся МОЛЧА сносил чужой идущий
# прогон. Машины общие, так что дефолт обязан быть персональным.
CONTAINER_NAME="${CONTAINER_NAME:-design2code-bench-$(id -un)}"

mkdir -p "$HOST_OUTDIR" "$HOST_HF_CACHE"

# HOST_STORAGE монтируется, только если реально существует на хосте — иначе
# либо чекпоинты берутся с HF hub (--model Qwen/Qwen3.5-9B как раньше), либо
# пользователь ещё не примонтировал общий диск на хосте, и лучше явно
# сообщить об этом, чем тихо стартовать без него и упасть непонятной
# ошибкой "path not found" уже внутри vLLM.
STORAGE_MOUNT_ARGS=()
if [[ -d "$HOST_STORAGE" ]]; then
    STORAGE_MOUNT_ARGS=(-v "$HOST_STORAGE:/mnt/storage-1:ro")
else
    echo "[run] Внимание: $HOST_STORAGE не найден на хосте — общий диск с чекпоинтами" >&2
    echo "не будет примонтирован. Если --model указывает на /mnt/storage-1/..., это упадёт." >&2
    echo "Если диск смонтирован в другом месте, задайте HOST_STORAGE=<путь>." >&2
fi

# Защита: --outdir всегда должен указывать на /app/output (единственный путь,
# смонтированный volume-ом с хоста, см. -v ниже). Если пользователь передаст
# свой --outdir в "$@", argparse внутри python молча возьмёт ПОСЛЕДНИЙ (свой),
# он окажется внутри контейнера без volume - и после docker rm/падения все
# результаты и progress.json потеряются без единого предупреждения. Поэтому
# ловим это здесь явно, а не полагаемся на argparse.
for arg in "$@"; do
    if [[ "$arg" == "--outdir" || "$arg" == --outdir=* ]]; then
        echo "ОШИБКА: не передавайте --outdir в run.sh — путь внутри контейнера" >&2
        echo "всегда фиксирован на /app/output. Чтобы изменить место на ХОСТЕ," >&2
        echo "задайте переменную HOST_OUTDIR перед запуском, например:" >&2
        echo "  HOST_OUTDIR=/data/my_run ./run.sh --n-samples 70000" >&2
        exit 1
    fi
done

# Если контейнер с таким именем уже существует, но не запущен (например,
# после docker stop) — уберём его перед пересозданием, иначе `docker run
# --name` откажется стартовать с "the container name is already in use".
# Volume-данные (outdir/hf_cache) при этом не трогаются - они на хосте.
# ⚠ РАБОТАЮЩИЙ контейнер не трогаем: на общих машинах это чужой прогон на
# несколько часов, и снести его молча — худшее, что может сделать скрипт.
# Останавливаем только УЖЕ ЗАВЕРШЁННЫЙ тёзка (иначе docker run --name
# откажется стартовать). Перебить живой можно явно: FORCE_RM=1.
if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    if [[ "${FORCE_RM:-0}" == "1" ]]; then
        echo "[run] FORCE_RM=1 — сношу РАБОТАЮЩИЙ контейнер $CONTAINER_NAME" >&2
        docker rm -f "$CONTAINER_NAME" >/dev/null
    else
        echo "ОШИБКА: контейнер $CONTAINER_NAME уже РАБОТАЕТ." >&2
        echo "Это может быть чужой прогон на общей машине. Проверьте:" >&2
        echo "  docker ps --filter name=$CONTAINER_NAME" >&2
        echo "Запуститесь под своим именем (CONTAINER_NAME=...) либо, если" >&2
        echo "контейнер точно ваш и его не жалко, повторите с FORCE_RM=1." >&2
        exit 1
    fi
elif docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "[run] Убираю завершённый контейнер $CONTAINER_NAME (данные в volume сохранены)..."
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

echo "[run] Образ:        $IMAGE_TAG"
echo "[run] outdir (хост): $HOST_OUTDIR  ->  /app/output (в контейнере)"
echo "[run] HF cache:      $HOST_HF_CACHE  ->  /root/.cache/huggingface"
if [[ ${#STORAGE_MOUNT_ARGS[@]} -gt 0 ]]; then
    echo "[run] Storage:       $HOST_STORAGE  ->  /mnt/storage-1 (в контейнере, read-only)"
fi
echo "[run] GPU:           $GPUS"
echo "[run] Аргументы скрипту: $* "
echo

# --outdir всегда фиксирован на /app/output внутри контейнера (примонтирован
# с хоста) - остальные аргументы (--n-samples, --batch-size, --model, и т.д.)
# прозрачно прокидываются как есть в run_benchmark_batched.py через "$@".
# STORAGE_MOUNT_ARGS - монтирование общего диска с чекпоинтами (см. выше),
# пустой массив если HOST_STORAGE не существовал на хосте - `"${arr[@]}"` с
# пустым массивом безопасен под `set -u` начиная с Bash 4.4+.
# Креды ClearML пробрасываем, только если есть в окружении: без них tracking.py
# тихо no-op'ит. HOST_MODEL_DIR — чекпоинт вне HF-кэша и вне /mnt/storage-1.
ENV_ARGS=()
for var in CLEARML_API_ACCESS_KEY CLEARML_API_SECRET_KEY CLEARML_API_HOST \
           CLEARML_WEB_HOST CLEARML_FILES_HOST CLEARML_PROJECT CLEARML_TAGS \
           CLEARML_TRAIN_TASK CLEARML_DISABLE HF_TOKEN; do
    [[ -n "${!var:-}" ]] && ENV_ARGS+=(-e "$var=${!var}")
done
MODEL_MOUNT=()
if [[ -n "${HOST_MODEL_DIR:-}" ]]; then
    MODEL_MOUNT=(-v "$HOST_MODEL_DIR:$HOST_MODEL_DIR:ro")
fi

# --shm-size: дефолтные 64 МБ у Docker малы и vLLM (KV-cache, тензорный
# параллелизм), и Chromium под несколькими воркерами — тот падает с
# "Target crashed"/"Target closed". 1g не хватало на 16 воркеров и высокие
# страницы, поэтому 4g; при --num-workers под сотню задавайте SHM_SIZE=32g.
docker run \
    --name "$CONTAINER_NAME" \
    --gpus "$GPUS" \
    --shm-size="${SHM_SIZE:-4g}" \
    --ipc=host \
    -v "$HOST_OUTDIR:/app/output" \
    -v "$HOST_HF_CACHE:/root/.cache/huggingface" \
    "${STORAGE_MOUNT_ARGS[@]}" \
    "${MODEL_MOUNT[@]}" \
    "${ENV_ARGS[@]}" \
    "$IMAGE_TAG" \
    --outdir /app/output \
    "$@"
