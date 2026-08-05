#!/usr/bin/env bash
# Свип по эпохам ОДНИМ прогоном: обучаем 5 эпох на 40 коротких Design2Code
# (train==eval) с сохранением чекпоинта КАЖДУЮ эпоху, потом бенчим каждый.
# Ищем, на какой эпохе overfit бьёт базу (0.835) до autoregressive drift.
# В разы дешевле, чем 4 отдельных прогона. Датасеты собраны sanity_fit.
set -uo pipefail
cd "$(dirname "$0")"; REPO="$PWD"
BASE=/mnt/storage-1/Screenshot2Code
LR="${LR:-2e-5}"; EPOCHS="${EPOCHS:-5}"; NPROC="${NPROC:-2}"
GPUS="${GPUS:-\"device=0,1\"}"
OUT="$BASE/checkpoints_exps/d2c-sweep"
BENCH_DS_C=/storage/Screenshot2Code/hf_cache/d2c_short_bench
NREAL=$(docker run --rm -v /mnt/storage-1:/storage --entrypoint /opt/venv/bin/python sft -c "from datasets import load_from_disk;print(len(load_from_disk('$BENCH_DS_C')))" 2>/dev/null)
LOG="$BASE/SWEEP.log"; say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
mkchmod(){ docker run --rm -v /mnt/storage-1:/storage --entrypoint bash sft -c "mkdir -p /storage/${1#/mnt/storage-1/} && chmod -R 777 /storage/${1#/mnt/storage-1/}" 2>/dev/null; }

say "свип ОДНИМ прогоном: $EPOCHS эпох, чекпоинт/эпоху | $NREAL сэмплов | lr $LR | база 0.835"
mkchmod "$OUT"

# --- 1. одно обучение, save каждую эпоху ---
# 40 сэмплов, эфф.батч 8 (bs1*accum4*2) -> 5 шагов/эпоха -> checkpoint-5/10/15/20/25.
if ! ls "$OUT"/checkpoint-*/config.json >/dev/null 2>&1; then
  say "обучение full-FT lr $LR, $EPOCHS эпох, save_strategy epoch..."
  DATA_DIR="$BASE/data" HF_CACHE="$BASE/hf_cache" OUT_DIR="$OUT" \
  CONTAINER_HOME="$BASE/container-home" GPUS="$GPUS" \
    "$REPO/SFT/run.sh" torchrun --nproc_per_node="$NPROC" --tee 3 -m train.train_sft \
      --config configs/full_ft_qwen3_5_4b.yaml --dataset_name /data/d2c_short \
      --num_train_epochs "$EPOCHS" --learning_rate "$LR" --lr_scheduler_type constant --warmup_ratio 0 \
      --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
      --eval_strategy no --save_strategy epoch --save_total_limit "$EPOCHS" \
      --output_dir /out > "$BASE/sweep_train.log" 2>&1
  say "обучение: rc=$?"
fi

# --- 2. бенч каждого чекпоинта ---
# checkpoint-<step>; сортируем по номеру шага = порядок эпох.
for CKPT in $(ls -d "$OUT"/checkpoint-*/ 2>/dev/null | sort -t- -k2 -n); do
  CKPT="${CKPT%/}"; STEP=$(basename "$CKPT" | sed 's/checkpoint-//'); EP=$(( STEP / 5 ))
  [[ -f "$CKPT/config.json" ]] || { say "ep$EP ($STEP): весов нет"; continue; }
  BDIR="$OUT/bench-ep$EP"
  if [[ ! -f "$BDIR/summary.json" ]]; then
    mkchmod "$BDIR"
    say "ep$EP: бенч чекпоинта $STEP шагов..."
    env IMAGE_TAG=design2code-bench:latest HOST_OUTDIR="$BDIR" HOST_HF_CACHE="$BASE/hf_cache" \
        HOST_MODEL_DIR="$CKPT" CONTAINER_NAME="sweep-ep$EP" GPUS="$GPUS" CLEARML_DISABLE=1 \
        "$REPO/Evaluation/run.sh" --no-resume --model "$CKPT" \
          --hf-dataset "$BENCH_DS_C" --hf-config default --hf-split train \
          --n-samples "$NREAL" --batch-size "$NREAL" --n-examples-per-batch 4 \
          --max-pixels 2097152 --tensor-parallel-size "$NPROC" \
          --gpu-memory-utilization 0.9 --max-new-tokens 16384 --max-model-len 24384 --num-workers 8 \
        > "$BASE/sweep_bench_ep$EP.log" 2>&1
  fi
  SC=$(python3 -c "import json;print(round(json.load(open('$BDIR/summary.json'))['final_score'],3))" 2>/dev/null)
  say "ep$EP: final_score = ${SC:-НЕТ} (база 0.835)"
done

say "=== СВОДКА СВИПА (база 0.835 block 0.860 text 0.948) ==="
for BDIR in $(ls -d "$OUT"/bench-ep*/ 2>/dev/null | sort -t p -k3 -n); do
  f="${BDIR%/}/summary.json"; ep=$(basename "${BDIR%/}" | sed 's/bench-ep//')
  [ -f "$f" ] && python3 -c "import json;d=json.load(open('$f'));print(f'  ep$ep: final {d[\"final_score\"]:.3f} block {d[\"block_match\"]:.3f} text {d[\"text\"]:.3f} pos {d[\"position\"]:.3f} color {d[\"color\"]:.3f}')"
done | tee -a "$LOG"
