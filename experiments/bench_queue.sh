#!/usr/bin/env bash
# Очередь одиночных бенчей на ОДНОЙ карте: модель за моделью, без обучения.
#
# Нужна для контрольных замеров вокруг A/B на 15k: перемер пилотного E3
# нынешним прибором, повтор базы на другой машине (контроль сравнимости),
# база со штрафом за повторы. Каждый пункт — «имя|путь-или-hf-id|доп.флаги».
#
# Все прогоны идут ОДНИМ образом и одними параметрами, кроме явно указанных
# в доп.флагах: иначе замеры снова окажутся несравнимы между собой, а именно
# из-за этого в проекте уже сгорело полтора месяца.
#
#   GPU=3 ./experiments/bench_queue.sh
set -uo pipefail

cd "$(dirname "$0")/.."
REPO="$PWD"

# Общий диск. Путь монтирования переопределяется через STORAGE, дефолт — тот же,
# что был вбит раньше, поэтому поведение прогонов не меняется.
: "${STORAGE:=/mnt/storage-1}"
BASE="$STORAGE/Screenshot2Code"
OUT_ROOT="${OUT_ROOT:-$BASE/checkpoints_exps/wc2m-15k-ab/bench}"
HF_CACHE="${HF_CACHE:-$BASE/hf_cache}"
GPU="${GPU:-3}"
IMAGE="${IMAGE:-design2code-bench:kozlov}"
WORKERS="${WORKERS:-96}"
SHM="${SHM:-32g}"
PIXELS="${PIXELS:-2097152}"
BENCH_N="${BENCH_N:-484}"
MAX_NEW="${MAX_NEW:-16384}"
GPU_UTIL="${GPU_UTIL:-0.50}"
LOGS="$OUT_ROOT/../logs"; mkdir -p "$LOGS"
REPORT="$OUT_ROOT/../REPORT-queue.txt"

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$REPORT"; }

E3=$BASE/checkpoints_exps/exps-3k/E3/full_ft_qwen3_5_4b_s42_20260804-174927

# Порядок намеренный: сперва то, что отвечает на открытый вопрос (E3), потом
# контроль сравнимости машин (та же база на другой машине), потом гипотеза
# про зацикливание базы.
JOBS=(
  "E3-pilot3k|$E3|"
  "E0-base-a100-3|Qwen/Qwen3.5-4B|"
  "E0-base-reppen105|Qwen/Qwen3.5-4B|--repetition-penalty 1.05"
)

say "очередь из ${#JOBS[@]} бенчей на карте $GPU, образ $IMAGE, воркеров $WORKERS"

for job in "${JOBS[@]}"; do
  IFS='|' read -r NAME MODEL EXTRA <<< "$job"
  if [[ -f "$OUT_ROOT/$NAME/summary.json" ]]; then
    say "$NAME: уже посчитан — пропускаю"; continue
  fi
  # Локальные веса надо ещё и примонтировать; hf-id монтировать нечего.
  MOUNT=""
  [[ -d "$MODEL" ]] && MOUNT="$MODEL"

  say "$NAME: старт ($MODEL $EXTRA)"
  start=$SECONDS
  env IMAGE_TAG="$IMAGE" HOST_OUTDIR="$OUT_ROOT/$NAME" HOST_HF_CACHE="$HF_CACHE" \
      ${MOUNT:+HOST_MODEL_DIR="$MOUNT"} \
      CONTAINER_NAME="bench-$NAME" GPUS="\"device=$GPU\"" SHM_SIZE="$SHM" \
      CLEARML_TAGS="wc2m15k-ab,$NAME" \
    bash "$REPO/Evaluation/run.sh" \
      --model "$MODEL" \
      --hf-dataset SALT-NLP/Design2Code-hf --hf-config default --hf-split train \
      --n-samples "$BENCH_N" --batch-size "$BENCH_N" \
      --max-pixels "$PIXELS" --tensor-parallel-size 1 \
      --gpu-memory-utilization "$GPU_UTIL" --max-new-tokens "$MAX_NEW" \
      --num-workers "$WORKERS" $EXTRA \
    > "$LOGS/$NAME.bench.log" 2>&1
  rc=$?
  say "$NAME: rc=$rc, $(( (SECONDS-start)/60 )) мин"
  if (( rc != 0 && SECONDS - start < 60 )); then
    say "$NAME: упал мгновенно — конвейер сломан, очередь останавливаю"
    say "  смотреть: $LOGS/$NAME.bench.log"
    exit 1
  fi
  if [[ -f "$OUT_ROOT/$NAME/summary.json" ]]; then
    say "  $(python3 -c "
import json; d=json.load(open('$OUT_ROOT/$NAME/summary.json'))
print(' '.join(f'{k}={d[k]:.4f}' for k in ('final_score','block_match','text','position','color','clip') if k in d))")"
  fi
done

say "очередь закончена"
