#!/usr/bin/env bash
# Мягкий рецепт оверфита + greedy-бенч каждой эпохи, одним прогоном.
#
# Зачем: прежний рецепт (lr 2e-5, constant, warmup 0) ломает модель за первые
# же 5 шагов — под greedy чекпоинты эпох 1 и 2 дают РОВНО 0.000 на всех 40
# сэмплах: модель открывает <style>, пишет полторы строки комментария и выдаёт
# EOS (выход 156-456 байт). Прежние ненулевые числа были артефактом сэмплинга
# при temperature 1.0, который иногда случайно выбивал модель из этого пути.
#
# Гипотеза: без резкого старта (lr 5e-6, cosine, warmup 0.1) модель не срывается,
# а запоминание 40 сэмплов выводит её выше базы. База под greedy = 0.758.
#
# Запуск: GPUS='"device=1,2"' NPROC=2 ./sweep_soft.sh
set -uo pipefail
cd "$(dirname "$0")/.."; REPO="$PWD"   # скрипт лежит в experiments/, работаем от корня репо
BASE=/mnt/storage-1/Screenshot2Code
mkdir -p "$BASE/logs/soft"
LR="${LR:-5e-6}"; EPOCHS="${EPOCHS:-5}"; NPROC="${NPROC:-2}"; TP="${TP:-1}"
GPUS="${GPUS:-\"device=1,2\"}"          # обучение: обе карты
BENCH_GPUS="${BENCH_GPUS:-\"device=1\"}"  # бенч: одной хватает
OUT="$BASE/checkpoints_exps/d2c-sweep-soft"
BENCH_DS_E=/root/.cache/huggingface/d2c_short_bench   # путь ВНУТРИ образа бенча
LOG="$BASE/logs/soft/SOFT.log"; say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
mkchmod(){ docker run --rm -v /mnt/storage-1:/storage --entrypoint bash sft \
             -c "mkdir -p /storage/${1#/mnt/storage-1/} && chmod -R 777 /storage/${1#/mnt/storage-1/}" 2>/dev/null; }

NREAL=$(docker run --rm -v /mnt/storage-1:/storage --entrypoint /opt/venv/bin/python sft \
        -c "from datasets import load_from_disk;print(len(load_from_disk('/storage/Screenshot2Code/hf_cache/d2c_short_bench')))" 2>/dev/null)
[[ -n "$NREAL" ]] || { say "датасет не читается"; exit 1; }

# Трейнер кладёт чекпоинты в $OUT/<run_name>/checkpoint-N — ищем на обоих
# уровнях и сортируем по НОМЕРУ ШАГА (в run_name полно дефисов, обычный sort врёт).
ckpts(){ ls -d "$OUT"/checkpoint-*/ "$OUT"/*/checkpoint-*/ 2>/dev/null |
         sed 's:/$::' | awk -F'checkpoint-' '{print $NF" "$0}' | sort -n | cut -d' ' -f2-; }

say "мягкий рецепт: lr $LR cosine warmup 0.1, $EPOCHS эпох, $NREAL сэмплов, база greedy 0.758"
mkchmod "$OUT"

# --- 1. обучение, чекпоинт каждую эпоху ---
if [[ -z "$(ckpts)" ]]; then
  say "обучение full-FT lr $LR..."
  DATA_DIR="$BASE/data" HF_CACHE="$BASE/hf_cache" OUT_DIR="$OUT" \
  CONTAINER_HOME="$BASE/container-home" GPUS="$GPUS" \
    "$REPO/SFT/run.sh" torchrun --nproc_per_node="$NPROC" --tee 3 -m train.train_sft \
      --config configs/full_ft_qwen3_5_4b.yaml --dataset_name /data/d2c_short \
      --num_train_epochs "$EPOCHS" --learning_rate "$LR" \
      --lr_scheduler_type cosine --warmup_ratio 0.1 \
      --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
      --eval_strategy no --save_strategy epoch --save_total_limit "$EPOCHS" \
      --output_dir /out > "$BASE/logs/soft/soft_train.log" 2>&1
  say "обучение: rc=$?"
fi

# --- 2. greedy-бенч каждого чекпоинта ---
for CKPT in $(ckpts); do
  STEP=$(basename "$CKPT" | sed 's/checkpoint-//'); EP=$(( STEP / 5 ))
  [[ -f "$CKPT/config.json" ]] || { say "ep$EP ($STEP): весов нет"; continue; }
  BDIR="$OUT/bench-ep$EP"
  if [[ ! -f "$BDIR/summary.json" ]]; then
    mkchmod "$BDIR"
    say "ep$EP: greedy-бенч чекпоинта $STEP шагов..."
    env IMAGE_TAG=design2code-bench:latest HOST_OUTDIR="$BDIR" HOST_HF_CACHE="$BASE/hf_cache" \
        HOST_MODEL_DIR="$CKPT" CONTAINER_NAME="soft-ep$EP" GPUS="$BENCH_GPUS" CLEARML_DISABLE=1 \
        "$REPO/Evaluation/run.sh" --no-resume --model "$CKPT" \
          --hf-dataset "$BENCH_DS_E" --hf-config default --hf-split train \
          --n-samples "$NREAL" --batch-size "$NREAL" --n-examples-per-batch "$NREAL" \
          --temperature 0 --max-pixels 2097152 --tensor-parallel-size "$TP" \
          --gpu-memory-utilization 0.9 --max-new-tokens 16384 --max-model-len 24384 --num-workers 8 \
        > "$BASE/soft_bench_ep$EP.log" 2>&1
    docker rm "soft-ep$EP" >/dev/null 2>&1
  fi
  SC=$(python3 -c "import json;print(round(json.load(open('$BDIR/summary.json'))['final_score'],3))" 2>/dev/null)
  say "ep$EP: final_score = ${SC:-НЕТ} (база greedy 0.758)"
done

say "=== СВОДКА МЯГКОГО РЕЦЕПТА (greedy; база 0.758) ==="
python3 - "$OUT" <<'PY' | tee -a "$LOG"
import json, os, sys, glob, csv
out = sys.argv[1]
for d in sorted(glob.glob(os.path.join(out, "bench-ep*")),
                key=lambda p: int(p.rsplit("ep", 1)[1])):
    f = os.path.join(d, "summary.json")
    if not os.path.exists(f):
        continue
    s = json.load(open(f))
    ep = d.rsplit("ep", 1)[1]
    zeros = "-"
    rc = os.path.join(d, "results.csv")
    if os.path.exists(rc):
        v = [float(r["final_score"]) for r in csv.DictReader(open(rc)) if r.get("final_score")]
        if v:
            zeros = "%d/%d" % (sum(1 for x in v if x < 0.01), len(v))
    print("  ep%-2s final %.3f  block %.3f  text %.3f  pos %.3f  color %.3f  clip %.3f  нулей %s" % (
        ep, s["final_score"], s["block_match"], s["text"], s["position"],
        s["color"], s["clip"], zeros))
PY
