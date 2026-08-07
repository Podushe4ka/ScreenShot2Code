#!/usr/bin/env bash
# A/B на 15k WebCode2M: ЧИСТЫЙ набор против СЫРОГО, одинаковый рецепт.
#
# Проверяемая гипотеза: чистка заменяет каждую картинку плоским серым блоком,
# а на бенче Design2Code все картинки — одна настоящая фотография (rick.jpg).
# То есть учим на одном распределении, меряем на другом. Сырой набор (пара из
# корпуса как есть, картинки настоящие) отвечает, чего это расхождение стоит.
# Правок бенча не требует: харнесс сам заменяет <img> плейсхолдерами и в
# предсказании, и в эталоне, так что наборы сравнимы одной метрикой.
#
# Рецепт общий и равный для обеих веток — E3: full-FT, lr 1e-5, cosine,
# warmup 0.03, wd 0.05, 1 эпоха, ZeRO-2, max_length 16384, 2.10 Мп.
# Эффективный батч удерживается на 64 при ЛЮБОМ числе карт (accum считается от
# NPROC), иначе ветки были бы несравнимы между собой при разной загрузке кластера.
#
# Запуск на ХОСТЕ в tmux:
#   tmux new -s ab
#   cd ~/ScreenShot2Code && source .env && ./experiments/run_wc2m_ab.sh
#
# Переопределяется: N_CKPT, MIN_GPUS, PREFER_GPUS, WAIT_FOR_PREFER_MIN,
# ONLY (clean|raw), SKIP_BENCH, LR, BENCH_N.
set -uo pipefail

cd "$(dirname "$0")/.."
REPO="$PWD"

BASE=/mnt/storage-1/Screenshot2Code
DATA_DIR="${DATA_DIR:-$BASE/data}"
HF_CACHE="${HF_CACHE:-$BASE/hf_cache}"
CKPT_ROOT="${CKPT_ROOT:-$BASE/checkpoints_exps}"
TARGET="${TARGET:-15000}"

N_CKPT="${N_CKPT:-5}"                  # чекпоинтов за прогон (просили 3-5)
EFF_BATCH="${EFF_BATCH:-64}"
MICRO_BS="${MICRO_BS:-4}"
LR="${LR:-1e-5}"
CONFIG="${CONFIG:-configs/full_ft_qwen3_5_4b.yaml}"
PIXELS="${PIXELS:-2097152}"
BENCH_N="${BENCH_N:-484}"
BENCH_MAX_NEW="${BENCH_MAX_NEW:-16384}"
GPU_UTIL="${GPU_UTIL:-0.50}"
BENCH_IMAGE="${BENCH_IMAGE:-design2code-bench:latest}"
BENCH_DATASET="${BENCH_DATASET:-SALT-NLP/Design2Code-hf}"

# Сколько карт ждать. Полный full-FT 4B под ZeRO-2 не влезает в огрызок памяти,
# поэтому «свободная» = занято меньше FREE_MB, а не «есть сколько-то места».
MIN_GPUS="${MIN_GPUS:-2}"
PREFER_GPUS="${PREFER_GPUS:-4}"
WAIT_FOR_PREFER_MIN="${WAIT_FOR_PREFER_MIN:-45}"   # ждём PREFER, потом миримся с MIN
FREE_MB="${FREE_MB:-5000}"

export CONTAINER_HOME="${CONTAINER_HOME:-$BASE/container-home}"
RUN_ROOT="$CKPT_ROOT/wc2m-15k-ab"
LOGS="$RUN_ROOT/logs"
REPORT="$RUN_ROOT/REPORT.txt"
mkdir -p "$LOGS" "$CONTAINER_HOME"

say()   { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$REPORT"; }
phase() { echo | tee -a "$REPORT"; say "########## $* ##########"; }

docker image inspect sft >/dev/null 2>&1 || { say "НЕТ образа sft"; exit 1; }
docker image inspect "$BENCH_IMAGE" >/dev/null 2>&1 || { say "НЕТ образа $BENCH_IMAGE"; exit 1; }
if [[ -z "${CLEARML_API_ACCESS_KEY:-}" && "${CLEARML_DISABLE:-0}" != "1" ]]; then
  say "CLEARML_API_ACCESS_KEY не задан — source .env перед запуском"; exit 1
fi

# ------------------------------------------------------------- выбор карт --
# Возвращает список свободных id через запятую. Берём не больше PREFER_GPUS:
# занимать всё, что видим, на общей машине незачем.
free_gpu_ids() {
  local ids=() id used
  while IFS=', ' read -r id used; do
    [[ -z "${used:-}" ]] && continue
    (( used < FREE_MB )) && ids+=("$id")
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)
  (( ${#ids[@]} > PREFER_GPUS )) && ids=("${ids[@]:0:$PREFER_GPUS}")
  printf '%s\n' "$(IFS=,; echo "${ids[*]}")"
}

# Ждём PREFER_GPUS карт ограниченное время, дальше соглашаемся на MIN_GPUS:
# прогон на двух картах вдвое дольше, но это лучше, чем не начать вовсе.
acquire_gpus() {
  local waited=0 ids n
  while true; do
    ids="$(free_gpu_ids)"
    n=$([[ -z "$ids" ]] && echo 0 || awk -F, '{print NF}' <<< "$ids")
    if (( n >= PREFER_GPUS )); then break; fi
    if (( n >= MIN_GPUS && waited >= WAIT_FOR_PREFER_MIN )); then
      say "ждать $PREFER_GPUS карт больше не буду ($waited мин) — иду на $n"
      break
    fi
    (( waited % 10 == 0 )) && say "жду карты: свободно $n, нужно $PREFER_GPUS (согласен на $MIN_GPUS через $WAIT_FOR_PREFER_MIN мин), прошло $waited мин"
    sleep 60; waited=$((waited+1))
  done
  GPU_IDS="$ids"
  NPROC=$(awk -F, '{print NF}' <<< "$GPU_IDS")
  GPUS="\"device=$GPU_IDS\""
  # Эффективный батч держим постоянным: иначе две ветки, запущенные при разной
  # загрузке кластера, будут обучены разным рецептом и сравнивать их нельзя.
  ACCUM=$(( EFF_BATCH / (MICRO_BS * NPROC) )); (( ACCUM < 1 )) && ACCUM=1
  # TP бенча обязан делить 32 головы внимания: 1, 2 или 4, но НЕ 3.
  case "$NPROC" in 1|2|4) BENCH_TP="$NPROC" ;; *) BENCH_TP=2 ;; esac
  say "карты $GPU_IDS | NPROC=$NPROC | accum=$ACCUM -> эфф. батч $(( MICRO_BS * ACCUM * NPROC )) | бенч TP=$BENCH_TP"
}

# Каталог набора появляется в момент, когда save_to_disk только НАЧАЛ писать,
# поэтому ждать надо не его, а того, что набор реально грузится: иначе разрез
# стартует по недописанным шардам и падает (ровно это и случилось в первый раз).
wait_dataset_ready() {
  local host_path="$1" name waited=0
  name="$(basename "$host_path")"
  while true; do
    if docker run --rm -v /mnt/storage-1:/storage --entrypoint /opt/venv/bin/python sft -c "
from datasets import load_from_disk
import sys
sys.exit(0 if len(load_from_disk('/storage/Screenshot2Code/data/$name')) else 1)" >/dev/null 2>&1; then
      say "набор $name готов и грузится ✓"
      return 0
    fi
    (( waited % 10 == 0 )) && say "жду готовности набора $name, прошло $waited мин"
    sleep 60; waited=$((waited+1))
    (( waited > 150 )) && { say "набор $name не собрался за 2.5 ч"; return 1; }
  done
}

resolve_weights() {
  local dir="$1" sub
  [[ -f "$dir/config.json" ]] && { echo "$dir"; return 0; }
  for sub in $(ls -dt "$dir"/*/ 2>/dev/null); do
    sub="${sub%/}"
    [[ -f "$sub/config.json" ]] && { echo "$sub"; return 0; }
  done
  return 1
}

# --------------------------------------------------- сырой набор: разрез ---
phase "ФАЗА 0: сплиты обоих наборов"
# По умолчанию — СОПОСТАВЛЕННАЯ пара: обе ветки собраны из одних и тех же
# страниц (чистая ветка отрендерена по html-кэшу сырой). Иначе наборы отличались
# бы ещё и составом: сырой теряет 11% страниц, чей скриншот в корпусе не 1280
# пикселей шириной, и сравнение мерило бы обработку вместе с объёмом данных.
RAW_SRC="${RAW_SRC:-$DATA_DIR/webcode2m_${TARGET}_raw}"
RAW_SPLIT="${RAW_SPLIT:-$DATA_DIR/webcode2m_ab_raw_split}"
CLEAN_SPLIT="${CLEAN_SPLIT:-$DATA_DIR/webcode2m_ab_clean_split}"

CLEAN_SRC="${CLEAN_SRC:-$DATA_DIR/webcode2m_ab_clean}"
if [[ ! -d "$CLEAN_SPLIT/train" ]]; then
  wait_dataset_ready "$CLEAN_SRC" || exit 1
  say "разрез чистого набора (val-frac 0.05, seed 42)"
  docker run --rm -v "$REPO":/w -v /mnt/storage-1:/storage -w /w \
    --entrypoint /opt/venv/bin/python sft \
    Data/converters/make_split.py "/storage/Screenshot2Code/data/$(basename "$CLEAN_SRC")" \
    "/storage/Screenshot2Code/data/$(basename "$CLEAN_SPLIT")" \
    --val-frac 0.05 --seed 42 > "$LOGS/split_clean.log" 2>&1
  say "разрез чистого: rc=$? (лог: $LOGS/split_clean.log)"
  grep '^\[split\]' "$LOGS/split_clean.log" | tee -a "$REPORT"
fi
[[ -d "$CLEAN_SPLIT/validation" ]] || { say "у чистого набора нет val-сплита"; exit 1; }
say "чистый: $(basename "$CLEAN_SPLIT") ✓"

if [[ ! -d "$RAW_SPLIT/train" ]]; then
  wait_dataset_ready "$RAW_SRC" || exit 1
  say "разрез сырого набора (val-frac 0.05, seed 42 — те же, что у чистого)"
  docker run --rm -v "$REPO":/w -v /mnt/storage-1:/storage -w /w \
    --entrypoint /opt/venv/bin/python sft \
    Data/converters/make_split.py "/storage/Screenshot2Code/data/$(basename "$RAW_SRC")" \
    "/storage/Screenshot2Code/data/$(basename "$RAW_SPLIT")" \
    --val-frac 0.05 --seed 42 > "$LOGS/split_raw.log" 2>&1
  say "разрез сырого: rc=$? (лог: $LOGS/split_raw.log)"
  grep '^\[split\]' "$LOGS/split_raw.log" | tee -a "$REPORT"
fi
[[ -d "$RAW_SPLIT/validation" ]] || { say "у сырого набора нет val-сплита"; exit 1; }
say "сырой: $(basename "$RAW_SPLIT") ✓"

# ------------------------------------------------------------- обучение ----
train_variant() {
  local name="$1" ds="$2"
  local out="$RUN_ROOT/$name"

  if compgen -G "$out/*/config.json" > /dev/null 2>&1; then
    say "$name: веса уже есть — обучение пропускаю"; return 0
  fi

  acquire_gpus
  local start=$SECONDS
  # save_steps долей, а не числом: абсолютное значение на разных размерах
  # набора даёт разное число чекпоинтов, а нам нужно ровно N_CKPT в обеих ветках.
  local save_frac
  save_frac=$(awk -v n="$N_CKPT" 'BEGIN{printf "%.4f", 1.0/n}')
  env DATA_DIR="$DATA_DIR" HF_CACHE="$HF_CACHE" OUT_DIR="$out" GPUS="$GPUS" \
      CLEARML_TAGS="wc2m15k-ab,$name,full_ft,lr$LR" SFT_MAX_PIXELS="$PIXELS" \
    "$REPO/SFT/run.sh" \
      torchrun --nproc_per_node="$NPROC" --tee 3 -m train.train_sft \
        --config "$CONFIG" \
        --dataset_name "/data/$(basename "$ds")" \
        --learning_rate "$LR" \
        --per_device_train_batch_size "$MICRO_BS" \
        --gradient_accumulation_steps "$ACCUM" \
        --save_strategy steps --save_steps "$save_frac" --save_total_limit "$N_CKPT" \
        --report_to clearml \
        --output_dir /out \
    > "$LOGS/$name.train.log" 2>&1
  local rc=$?
  say "$name обучение: rc=$rc, $(( (SECONDS-start)/60 )) мин (лог: $LOGS/$name.train.log)"
  return $rc
}

# --------------------------------------------------------------- бенч ------
bench_checkpoints() {
  local name="$1"
  local out="$RUN_ROOT/$name" weights ckpt tag
  weights=$(resolve_weights "$out") || { say "$name: весов нет — бенч пропущен"; return 1; }

  # Бенчим КАЖДЫЙ чекпоинт: смысл N чекпоинтов в том, чтобы увидеть, где
  # начинается деградация, а не только куда пришли в конце.
  for ckpt in $(ls -d "$weights"/checkpoint-*/ 2>/dev/null | sort -V) "$weights"; do
    ckpt="${ckpt%/}"
    [[ -f "$ckpt/config.json" ]] || continue
    tag="$name-$(basename "$ckpt")"
    if [[ -f "$RUN_ROOT/bench/$tag/summary.json" ]]; then
      say "$tag: уже отбенчен"; continue
    fi
    acquire_gpus
    local start=$SECONDS
    env IMAGE_TAG="$BENCH_IMAGE" HOST_OUTDIR="$RUN_ROOT/bench/$tag" \
        HOST_HF_CACHE="$HF_CACHE" HOST_MODEL_DIR="$ckpt" \
        CONTAINER_NAME="bench-$tag" GPUS="$GPUS" \
        CLEARML_TAGS="wc2m15k-ab,$name,$(basename "$ckpt")" \
      "$REPO/Evaluation/run.sh" \
        --model "$ckpt" \
        --hf-dataset "$BENCH_DATASET" --hf-config default --hf-split train \
        --n-samples "$BENCH_N" --batch-size "$BENCH_N" \
        --max-pixels "$PIXELS" --tensor-parallel-size "$BENCH_TP" \
        --gpu-memory-utilization "$GPU_UTIL" --max-new-tokens "$BENCH_MAX_NEW" \
      > "$LOGS/$tag.bench.log" 2>&1
    say "$tag бенч: rc=$?, $(( (SECONDS-start)/60 )) мин"
    if [[ -f "$RUN_ROOT/bench/$tag/summary.json" ]]; then
      say "  $(python3 -c "
import json; d=json.load(open('$RUN_ROOT/bench/$tag/summary.json'))
print(' '.join(f'{k}={d[k]:.4f}' for k in ('final_score','block_match','text','position','color','clip') if k in d))")"
    fi
  done
}

# Сперва ОБА обучения, и только потом бенчи. Иначе вторая ветка ждала бы, пока
# отбенчатся пять чекпоинтов первой (~2.5 ч), а карты на общей машине за это
# время успевают уйти к соседям. Веса дороже: бенч по ним можно снять когда
# угодно, а обучение при занятых картах не начать вовсе.
for VARIANT in clean raw; do
  [[ -n "${ONLY:-}" && " $ONLY " != *" $VARIANT "* ]] && continue
  case "$VARIANT" in
    clean) DS="$CLEAN_SPLIT" ;;
    raw)   DS="$RAW_SPLIT" ;;
  esac
  phase "ОБУЧЕНИЕ: $VARIANT ($(basename "$DS")), $N_CKPT чекпоинтов"
  train_variant "$VARIANT" "$DS" || say "$VARIANT: обучение не удалось, иду дальше"
done

if [[ "${SKIP_BENCH:-0}" != "1" ]]; then
  for VARIANT in clean raw; do
    [[ -n "${ONLY:-}" && " $ONLY " != *" $VARIANT "* ]] && continue
    [[ -d "$RUN_ROOT/$VARIANT" ]] || continue
    phase "БЕНЧ: $VARIANT, все чекпоинты"
    bench_checkpoints "$VARIANT"
  done
fi

phase "ИТОГИ"
for s in "$RUN_ROOT"/bench/*/summary.json; do
  [[ -f "$s" ]] || continue
  say "$(basename "$(dirname "$s")"): $(python3 -c "
import json; d=json.load(open('$s'))
print(' '.join(f'{k}={d[k]:.4f}' for k in ('final_score','block_match','text','position','color','clip') if k in d))")"
done
say "ВСЁ. В ClearML — тег wc2m15k-ab."
