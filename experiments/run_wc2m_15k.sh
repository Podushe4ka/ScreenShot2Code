#!/usr/bin/env bash
# WebCode2M 15k: генерация датасета -> full-FT лучшим рецептом на 4 картах -> бенч Design2Code-484.
#
# Рецепт — E3 из пилота (единственный сетап, который подошёл к базе ближе всех:
# 0.732 против 0.826): full-FT, lr 1e-5, cosine, warmup 0.03, wd 0.05, 1 эпоха,
# ZeRO-2, max_length 16384, пиксель-бюджет 2.10 Мп. Отличие от пилота — объём
# данных (15k против 3k) и эффективный батч 64 вместо 16: батч берётся из
# конфига, сгенерированного при N_GPUS=4 (микробатч 4 x accum 4 x 4 карты).
# Ось батча — P1.3 из ROADMAP, единственная дешёвая, которую ещё не трогали.
#
# Запускается НА ХОСТЕ (a100-3), не внутри контейнера, в tmux:
#   tmux new -s wc2m15k
#   cd ~/ScreenShot2Code && source .env && ./experiments/run_wc2m_15k.sh
#
# Переопределяется: TARGET, GPUS, GPU_IDS, NPROC, BENCH_TP, BENCH_N, LR,
# N_WORKERS, RUN_DIR, SKIP_BENCH, SKIP_TRAIN.
set -uo pipefail

cd "$(dirname "$0")/.."
REPO="$PWD"
source "$REPO/experiments/lib/common.sh"

BASE="$STORAGE/Screenshot2Code"
DATA_DIR="${DATA_DIR:-$BASE/data}"
HF_CACHE="${HF_CACHE:-$BASE/hf_cache}"
TARGET="${TARGET:-15000}"
DATASET_NAME="webcode2m_${TARGET}_split"
RUN_DIR="${RUN_DIR:-$BASE/checkpoints_exps/wc2m-15k}"

GPUS="${GPUS:-\"device=0,1,2,3\"}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC="${NPROC:-4}"
BENCH_TP="${BENCH_TP:-4}"          # должен делить 32 головы: 1, 2 или 4, НЕ 3
BENCH_N="${BENCH_N:-484}"
BENCH_MAX_NEW="${BENCH_MAX_NEW:-16384}"
GPU_UTIL="${GPU_UTIL:-0.50}"       # остальное — CLIP в 16 воркерах метрики
PIXELS="${PIXELS:-2097152}"        # обязан совпадать между обучением и бенчем
LR="${LR:-1e-5}"
CONFIG="${CONFIG:-configs/full_ft_qwen3_5_4b.yaml}"
N_WORKERS="${N_WORKERS:-96}"
BENCH_IMAGE="${BENCH_IMAGE:-design2code-bench:latest}"
BENCH_DATASET="${BENCH_DATASET:-SALT-NLP/Design2Code-hf}"
TRAIN_FREE_MB="${TRAIN_FREE_MB:-5000}"
BENCH_FREE_MB="${BENCH_FREE_MB:-25000}"

# Кэши компиляции и tmp контейнеров — на общий диск, не на локальный.
export CONTAINER_HOME="${CONTAINER_HOME:-$BASE/container-home}"

LOGS="$RUN_DIR/logs"
REPORT="$RUN_DIR/REPORT.txt"
mkdir -p "$LOGS" "$CONTAINER_HOME"

RUN_LOG="$REPORT"

# Проверки
docker image inspect sft >/dev/null 2>&1 || { say "НЕТ образа sft"; exit 1; }
docker image inspect "$BENCH_IMAGE" >/dev/null 2>&1 || { say "НЕТ образа $BENCH_IMAGE"; exit 1; }
if [[ -z "${CLEARML_API_ACCESS_KEY:-}" ]]; then
  say "CLEARML_API_ACCESS_KEY не задан — трекинга не будет. source .env перед запуском."
  [[ "${CLEARML_DISABLE:-0}" == "1" ]] || exit 1
fi
# Конфиг сгенерирован при N_GPUS=4: accum 4 при микробатче 4. Если запустить
# такой конфиг на другом числе карт, эффективный батч молча уедет.
ACCUM=$(grep -oP '^gradient_accumulation_steps:\s*\K[0-9]+' "$REPO/SFT/$CONFIG")
MICRO=$(grep -oP '^per_device_train_batch_size:\s*\K[0-9]+' "$REPO/SFT/$CONFIG")
say "эфф. батч = $MICRO x $ACCUM x $NPROC = $(( MICRO * ACCUM * NPROC ))"

wait_for_gpus() {
  local limit="$1" waited=0
  while true; do
    local busy=0 used
    for g in ${GPU_IDS//,/ }; do
      used=$(nvidia-smi --id="$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
      [[ -z "$used" ]] && used=999999
      (( used > limit )) && busy=1
    done
    (( busy == 0 )) && break
    (( waited % 10 == 0 )) && say "жду GPU $GPU_IDS (< ${limit} МБ занято), прошло $waited мин"
    sleep 60; waited=$((waited+1))
  done
}

say "каталог прогона: $RUN_DIR"
say "датасет: $DATA_DIR/$DATASET_NAME | GPU: $GPUS | обучение на $NPROC карт | бенч TP=$BENCH_TP"
say "рецепт: $(basename "$CONFIG") lr=$LR, ${PIXELS} px"

# ФАЗА 1: датасет 15k
phase "ФАЗА 1: датасет WebCode2M на $TARGET примеров"
if [[ -d "$DATA_DIR/${DATASET_NAME}/train" ]]; then
  say "датасет уже есть — пропускаю генерацию"
else
  if [[ -d "$DATA_DIR/webcode2m_$TARGET" ]]; then
    say "сырой конвертированный набор уже есть — пропускаю рендер"
  else
    # TMPDIR НЕ переопределяем: Chromium падает "Target crashed", если его
    # временные файлы лежат на сетевом диске.
    say "конвертация: playwright, $N_WORKERS воркеров (для 3k занимало ~2 мин рендера)"
    # HF_TOKEN пробрасывается, только если он есть в окружении: без него всё
    # работает, просто скачивание идёт на лимитах для анонимных запросов.
    docker run --rm -v "$REPO":/w -v /mnt/storage-1:/storage --shm-size=2g \
      -e HF_HOME=/storage/Screenshot2Code/hf_cache ${HF_TOKEN:+-e HF_TOKEN} \
      -w /w/Data/converters/webcode2m --entrypoint python3 "$BENCH_IMAGE" \
      convert_parallel.py --target "$TARGET" --n-workers "$N_WORKERS" \
      --max-scan $(( TARGET * 4 )) \
      --html-cache "/storage/Screenshot2Code/data/webcode2m_${TARGET}_htmls.jsonl.gz" \
      --out "/storage/Screenshot2Code/data/webcode2m_$TARGET" \
      > "$LOGS/convert.log" 2>&1
    say "конвертация: rc=$? (лог: $LOGS/convert.log)"
    grep -E '^\[фаза 2\] готово|^\[приёмка\]' "$LOGS/convert.log" | tee -a "$REPORT"
  fi

  say "разрез train/validation (val-frac 0.05, seed 42)"
  docker run --rm -v "$REPO":/w -v /mnt/storage-1:/storage -w /w \
    --entrypoint /opt/venv/bin/python sft \
    Data/converters/make_split.py "/storage/Screenshot2Code/data/webcode2m_$TARGET" \
    "/storage/Screenshot2Code/data/$DATASET_NAME" \
    --val-frac 0.05 --seed 42 > "$LOGS/split.log" 2>&1
  say "разрез: rc=$? (лог: $LOGS/split.log)"
  grep '^\[split\]' "$LOGS/split.log" | tee -a "$REPORT"
fi

# Без val-сплита train_sft молча выключает eval_strategy, и eval_loss по ходу
# обучения считаться не будет — а он показывает переобучение раньше бенча.
[[ -d "$DATA_DIR/$DATASET_NAME/validation" ]] || { say "ОШИБКА: нет val-сплита"; exit 1; }

# ФАЗА 2: обучение
phase "ФАЗА 2: full-FT на $NPROC картах"
OUT_HOST="$RUN_DIR/train"
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  if compgen -G "$OUT_HOST/*/config.json" > /dev/null 2>&1; then
    say "веса уже есть — обучение пропускаю (FORCE=1 чтобы переобучить)"
  else
    wait_for_gpus "$TRAIN_FREE_MB"
    start=$SECONDS
    env DATA_DIR="$DATA_DIR" HF_CACHE="$HF_CACHE" OUT_DIR="$OUT_HOST" GPUS="$GPUS" \
        CLEARML_TAGS="wc2m15k,full_ft,lr$LR" SFT_MAX_PIXELS="$PIXELS" \
      "$REPO/SFT/run.sh" \
        torchrun --nproc_per_node="$NPROC" --tee 3 -m train.train_sft \
          --config "$CONFIG" \
          --dataset_name "/data/$DATASET_NAME" \
          --learning_rate "$LR" \
          --report_to clearml \
          --output_dir /out \
      > "$LOGS/train.log" 2>&1
    rc=$?
    say "обучение: rc=$rc, $(( (SECONDS-start)/60 )) мин (лог: $LOGS/train.log)"
    [[ $rc -ne 0 ]] && { say "дальше не иду"; exit 1; }
  fi
fi

# Трейнер кладёт веса в $OUT/<run_name>/, а не в $OUT — глоб только по верхнему
# уровню молча ничего не находит.
resolve_weights() {
  local dir="$1" sub
  [[ -f "$dir/config.json" ]] && { echo "$dir"; return 0; }
  for sub in $(ls -dt "$dir"/*/ 2>/dev/null); do
    sub="${sub%/}"
    [[ -f "$sub/config.json" ]] && { echo "$sub"; return 0; }
  done
  return 1
}

# ФАЗА 3: бенч
phase "ФАЗА 3: бенч Design2Code-$BENCH_N (greedy)"
[[ "${SKIP_BENCH:-0}" == "1" ]] && { say "SKIP_BENCH=1 — бенч пропущен"; exit 0; }

WEIGHTS=$(resolve_weights "$OUT_HOST") || { say "ОШИБКА: весов нет в $OUT_HOST"; exit 1; }
say "веса: $WEIGHTS"
wait_for_gpus "$BENCH_FREE_MB"

start=$SECONDS
env IMAGE_TAG="$BENCH_IMAGE" \
    HOST_OUTDIR="$RUN_DIR/bench" \
    HOST_HF_CACHE="$HF_CACHE" \
    HOST_MODEL_DIR="$WEIGHTS" \
    CONTAINER_NAME="bench-wc2m15k" GPUS="$GPUS" \
    CLEARML_TAGS="wc2m15k,full_ft,lr$LR" \
  "$REPO/Evaluation/run.sh" \
    --model "$WEIGHTS" \
    --hf-dataset "$BENCH_DATASET" --hf-config default --hf-split train \
    --n-samples "$BENCH_N" --batch-size "$BENCH_N" \
    --max-pixels "$PIXELS" \
    --tensor-parallel-size "$BENCH_TP" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-new-tokens "$BENCH_MAX_NEW" \
  > "$LOGS/bench.log" 2>&1
say "бенч: rc=$?, $(( (SECONDS-start)/60 )) мин (лог: $LOGS/bench.log)"

if [[ -f "$RUN_DIR/bench/summary.json" ]]; then
  say "итог: $(python3 -c "
import json; d=json.load(open('$RUN_DIR/bench/summary.json'))
print(' '.join(f'{k}={d[k]:.4f}' for k in
      ('final_score','final_score_arithmetic','block_match','text','position','color','clip') if k in d))
print('  декодирование:', d.get('decoding'), '| обрезано по токенам:', d.get('n_length_truncated'))")"
fi
say "ВСЁ ЗАКОНЧЕНО. В ClearML — тег wc2m15k."
