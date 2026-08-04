#!/usr/bin/env bash
# Ночная цепочка целиком: бенч чекпоинтов 1k -> генерация 3k -> те же шесть
# экспериментов на 3k -> бенч 3k -> сводка.
#
# Запуск на ХОСТЕ в tmux:
#   tmux new -s night
#   cd ~/ScreenShot2Code && source .env && ./run_night.sh 2>&1 | tee night.out
#
# Всё лежит под /mnt/storage-1/Screenshot2Code (локальный диск на a100-2 забит).
# Каждая фаза пишет отметку в NIGHT.log; падение фазы не роняет цепочку —
# следующая всё равно стартует, а в сводке будет видно, чего не хватает.
set -uo pipefail

cd "$(dirname "$0")"
REPO="$PWD"

BASE=/mnt/storage-1/Screenshot2Code
DATA_DIR="${DATA_DIR:-$BASE/data}"
HF_CACHE="${HF_CACHE:-$BASE/hf_cache}"
CKPT_ROOT="${CKPT_ROOT:-$BASE/checkpoints_exps}"
CKPT_1K="${CKPT_1K:-$CKPT_ROOT/exps-20260803-104549}"
CKPT_3K="${CKPT_3K:-$CKPT_ROOT/exps-3k}"

GPUS="${GPUS:-\"device=0,1\"}"
GPU_IDS="${GPU_IDS:-0,1}"        # для проверки занятости
NPROC="${NPROC:-2}"
BENCH_TP="${BENCH_TP:-2}"
BENCH_N="${BENCH_N:-484}"
TARGET_3K="${TARGET_3K:-3000}"
N_WORKERS="${N_WORKERS:-96}"
FREE_MB="${FREE_MB:-2000}"        # карта считается свободной, если занято меньше

LOG="$BASE/NIGHT.log"
mkdir -p "$BASE"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
phase() { echo | tee -a "$LOG"; say "############ $* ############"; }

say "СТАРТ ночной цепочки. Лог: $LOG"
say "данные: $DATA_DIR | кэш: $HF_CACHE | чекпоинты: $CKPT_ROOT"
say "GPU: $GPUS | бенч TP=$BENCH_TP, N=$BENCH_N | 3k: target=$TARGET_3K, workers=$N_WORKERS"

# ------------------------------------------------------ ждём образ и копии --
phase "ФАЗА 0: жду готовности образа и подготовки данных"
while ! grep -q BUILD_DONE ~/build.log 2>/dev/null; do sleep 30; done
say "образ бенча собран"
while [[ ! -f ~/prep.log ]]; do sleep 30; done
say "данные и модель скопированы"

if ! docker run --rm --entrypoint ls design2code-bench:latest tracking.py >/dev/null 2>&1; then
  say "ОШИБКА: образ собрался без tracking.py — дальше нет смысла"; exit 1
fi
say "образ проверен: tracking.py на месте"

# --------------------------------------------------------- ждём GPU -------
wait_for_gpus() {
  local waited=0
  while true; do
    local busy=0
    for g in ${GPU_IDS//,/ }; do
      local used
      used=$(nvidia-smi --id="$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
      [[ -z "$used" ]] && used=999999
      (( used > FREE_MB )) && busy=1
    done
    (( busy == 0 )) && break
    (( waited % 10 == 0 )) && say "жду свободных GPU $GPU_IDS (прошло $waited мин)"
    sleep 60; waited=$((waited+1))
  done
  say "GPU $GPU_IDS свободны — поехали"
}

phase "ФАЗА 1: бенч чекпоинтов 1k"
wait_for_gpus
if [[ -d "$CKPT_1K" ]]; then
  DATA_DIR="$DATA_DIR" HF_CACHE="$HF_CACHE" GPUS="$GPUS" \
  BENCH_TP="$BENCH_TP" BENCH_N="$BENCH_N" NOWAIT=1 \
    "$REPO/bench_all.sh" "$CKPT_1K" 2>&1 | tee -a "$LOG"
  say "ФАЗА 1 закончена (rc=${PIPESTATUS[0]})"
else
  say "ПРОПУСК: нет $CKPT_1K"
fi

phase "ФАЗА 2: генерация датасета на $TARGET_3K примеров"
if [[ -d "$DATA_DIR/webcode2m_${TARGET_3K}_split/train" ]]; then
  say "датасет уже есть — пропускаю генерацию"
else
  say "конвертация (playwright, $N_WORKERS воркеров)..."
  docker run --rm -v "$REPO":/w -v /mnt/storage-1:/storage --shm-size=2g \
    -w /w/Data/webcode2m --entrypoint python3 design2code-bench:latest \
    convert_parallel.py --target "$TARGET_3K" --n-workers "$N_WORKERS" \
    --out "/storage/Screenshot2Code/data/webcode2m_$TARGET_3K" \
    > "$BASE/convert_3k.log" 2>&1
  say "конвертация: rc=$? (лог: $BASE/convert_3k.log)"

  say "разрез train/validation..."
  docker run --rm -v "$REPO":/w -v /mnt/storage-1:/storage -w /w \
    --entrypoint /opt/venv/bin/python sft \
    Data/make_split.py "/storage/Screenshot2Code/data/webcode2m_$TARGET_3K" \
    "/storage/Screenshot2Code/data/webcode2m_${TARGET_3K}_split" \
    --val-frac 0.05 --seed 42 > "$BASE/split_3k.log" 2>&1
  say "разрез: rc=$? (лог: $BASE/split_3k.log)"
fi

if [[ ! -d "$DATA_DIR/webcode2m_${TARGET_3K}_split/validation" ]]; then
  say "ОШИБКА: нет val-сплита -> eval_loss не будет считаться. Дальше не иду."
  exit 1
fi
say "val-сплит на месте"

phase "ФАЗА 3: шесть экспериментов на ${TARGET_3K} примерах"
wait_for_gpus
# SKIP_BENCH=1 намеренно: в run_pilot.sh осталась старая ошибка с путём к
# весам (ищет их в E<N>/, а они в E<N>/<run_name>/). Бенчит потом bench_all.sh,
# где путь разрешается правильно.
DATA_DIR="$DATA_DIR" DATASET_NAME="webcode2m_${TARGET_3K}_split" \
HF_CACHE="$HF_CACHE" RESULT_DIR="$CKPT_3K" \
GPUS="$GPUS" NPROC="$NPROC" WAVE=all SKIP_BENCH=1 \
  "$REPO/run_pilot.sh" 2>&1 | tee -a "$LOG"
say "ФАЗА 3 закончена (rc=${PIPESTATUS[0]})"

phase "ФАЗА 4: бенч чекпоинтов ${TARGET_3K}"
wait_for_gpus
DATA_DIR="$DATA_DIR" HF_CACHE="$HF_CACHE" GPUS="$GPUS" \
BENCH_TP="$BENCH_TP" BENCH_N="$BENCH_N" NOWAIT=1 \
  "$REPO/bench_all.sh" "$CKPT_3K" 2>&1 | tee -a "$LOG"
say "ФАЗА 4 закончена (rc=${PIPESTATUS[0]})"

phase "ИТОГИ"
say "=== 1000 примеров ==="
python3 "$REPO/SFT/scripts/collect_results.py" "$CKPT_1K" 2>&1 | tee -a "$LOG"
say "=== ${TARGET_3K} примеров ==="
python3 "$REPO/SFT/scripts/collect_results.py" "$CKPT_3K" 2>&1 | tee -a "$LOG"
say "ВСЁ ЗАКОНЧЕНО. В ClearML — теги pilot1k и pilot3k."
