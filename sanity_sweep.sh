#!/usr/bin/env bash
# Свип по эпохам на тех же 40 коротких Design2Code (train==eval). Ищем, на
# какой эпохе overfit БЬЁТ базу (0.835) — до того, как autoregressive drift
# уронит скор. Датасеты уже собраны sanity_fit (d2c_short, d2c_short_bench).
set -uo pipefail
cd "$(dirname "$0")"; REPO="$PWD"
BASE=/mnt/storage-1/Screenshot2Code
LR="${LR:-2e-5}"; NPROC="${NPROC:-2}"
GPUS="${GPUS:-\"device=1,2\"}"
EPOCHS_LIST="${EPOCHS_LIST:-1 2 3 5}"
BENCH_DS_C=/storage/Screenshot2Code/hf_cache/d2c_short_bench
NREAL=$(docker run --rm -v /mnt/storage-1:/storage --entrypoint /opt/venv/bin/python sft -c "from datasets import load_from_disk;print(len(load_from_disk('$BENCH_DS_C')))" 2>/dev/null)
LOG="$BASE/SWEEP.log"; say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
say "свип эпох: $EPOCHS_LIST | $NREAL сэмплов | lr $LR | база = 0.835"

for EP in $EPOCHS_LIST; do
  OUT="$BASE/checkpoints_exps/d2c-sweep-ep$EP"
  docker run --rm -v /mnt/storage-1:/storage --entrypoint bash sft -c "mkdir -p /storage/${OUT#/mnt/storage-1/} && chmod -R 777 /storage/${OUT#/mnt/storage-1/}" 2>/dev/null
  if ! ls "$OUT"/*/config.json >/dev/null 2>&1; then
    say "ep$EP: обучение full-FT lr $LR, $EP эпох..."
    DATA_DIR="$BASE/data" HF_CACHE="$BASE/hf_cache" OUT_DIR="$OUT" \
    CONTAINER_HOME="$BASE/container-home" GPUS="$GPUS" \
      "$REPO/SFT/run.sh" torchrun --nproc_per_node="$NPROC" --tee 3 -m train.train_sft \
        --config configs/full_ft_qwen3_5_4b.yaml --dataset_name /data/d2c_short \
        --num_train_epochs "$EP" --learning_rate "$LR" --lr_scheduler_type constant --warmup_ratio 0 \
        --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
        --eval_strategy no --save_strategy no --output_dir /out > "$BASE/sweep_train_ep$EP.log" 2>&1
    say "ep$EP: обучение rc=$?"
  fi
  W=$(ls -dt "$OUT"/*/ 2>/dev/null | head -1); W="${W%/}"
  [[ -f "$W/config.json" ]] || { say "ep$EP: весов нет"; continue; }
  if [[ ! -f "$OUT/bench/summary.json" ]]; then
    docker run --rm -v /mnt/storage-1:/storage --entrypoint bash sft -c "mkdir -p /storage/${OUT#/mnt/storage-1/}/bench && chmod -R 777 /storage/${OUT#/mnt/storage-1/}/bench" 2>/dev/null
    env IMAGE_TAG=design2code-bench:latest HOST_OUTDIR="$OUT/bench" HOST_HF_CACHE="$BASE/hf_cache" \
        HOST_MODEL_DIR="$W" CONTAINER_NAME="sweep-ep$EP" GPUS="$GPUS" CLEARML_DISABLE=1 \
        "$REPO/Evaluation/run.sh" --no-resume --model "$W" \
          --hf-dataset "$BENCH_DS_C" --hf-config default --hf-split train \
          --n-samples "$NREAL" --batch-size "$NREAL" --n-examples-per-batch 4 \
          --max-pixels 2097152 --tensor-parallel-size "$NPROC" \
          --gpu-memory-utilization 0.9 --max-new-tokens 16384 --max-model-len 24384 --num-workers 8 \
        > "$BASE/sweep_bench_ep$EP.log" 2>&1
  fi
  SC=$(python3 -c "import json;print(round(json.load(open('$OUT/bench/summary.json'))['final_score'],3))" 2>/dev/null)
  say "ep$EP: final_score = ${SC:-НЕТ} (база 0.835)"
done
say "=== СВОДКА СВИПА ==="
for EP in $EPOCHS_LIST; do
  f="$BASE/checkpoints_exps/d2c-sweep-ep$EP/bench/summary.json"
  [ -f "$f" ] && python3 -c "import json;d=json.load(open('$f'));print(f'  ep$EP: final {d[\"final_score\"]:.3f} block {d[\"block_match\"]:.3f} text {d[\"text\"]:.3f}')"
done | tee -a "$LOG"
say "база: final 0.835 block 0.860 text 0.948"
