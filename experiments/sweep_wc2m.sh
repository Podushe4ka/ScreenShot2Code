#!/usr/bin/env bash
# Оверфит на WebCode2M вместо Design2Code + greedy-бенч каждой эпохи.
#
# Зачем: на Design2Code оверфит упирается не в обучение, а в ГЕНЕРАЦИЮ —
# модель уходит в повтор внутри <head>, проедает весь бюджет токенов и до
# <body> не доходит, поэтому рендер пустой и score ровно 0. Эталоны
# Design2Code многословные (медиана 17.5к токенов), у WebCode2M — 4.1к.
# На коротких таргетах бюджета должно хватать с запасом: если оверфит на
# WebCode2M выходит нормальным, причина именно в длине/многословности, а не
# в рецепте и не в модели.
#
# Рецепт намеренно тот же, что у первого свипа по Design2Code (lr 2e-5,
# constant, warmup 0), чтобы отличалась ТОЛЬКО обучающая выборка.
#
# Запуск: GPUS='"device=1,2"' NPROC=2 ./sweep_wc2m.sh
set -uo pipefail
cd "$(dirname "$0")/.."; REPO="$PWD"   # скрипт лежит в experiments/, работаем от корня репо
# Общий диск. Путь монтирования переопределяется через STORAGE, дефолт — тот же,
# что был вбит раньше, поэтому поведение прогонов не меняется.
: "${STORAGE:=/mnt/storage-1}"
BASE="$STORAGE/Screenshot2Code"
mkdir -p "$BASE/logs/wc2m"
N="${N:-40}"; LR="${LR:-2e-5}"; EPOCHS="${EPOCHS:-5}"; NPROC="${NPROC:-2}"; TP="${TP:-1}"
GPUS="${GPUS:-\"device=1,2\"}"; BENCH_GPUS="${BENCH_GPUS:-\"device=1\"}"
SRC=/storage/Screenshot2Code/data/webcode2m_3000_split   # уже в drafting-формате
TRAIN_DS="$BASE/data/wc2m_short"; TRAIN_DS_C="/storage/Screenshot2Code/data/wc2m_short"
BENCH_DS_C="/storage/Screenshot2Code/hf_cache/wc2m_short_bench"  # путь для образа sft
BENCH_DS_E=/root/.cache/huggingface/wc2m_short_bench             # путь для образа бенча
OUT="$BASE/checkpoints_exps/wc2m-sweep"
LOG="$BASE/logs/wc2m/WC2M.log"; say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
mkchmod(){ docker run --rm -v /mnt/storage-1:/storage --entrypoint bash sft \
             -c "mkdir -p /storage/${1#/mnt/storage-1/} && chmod -R 777 /storage/${1#/mnt/storage-1/}" 2>/dev/null; }

# --- 1. взять N сэмплов WebCode2M, сохранить в обоих форматах ---
# train == eval: бенчим ровно то, на чём учили (memorization-тест).
if [[ ! -d "$TRAIN_DS/train" ]]; then
  say "беру $N сэмплов из webcode2m_3000_split"
  docker run --rm -v /mnt/storage-1:/storage --entrypoint /opt/venv/bin/python sft -c "
from datasets import load_from_disk, Dataset, DatasetDict, Features, Sequence, Image, Value
src = load_from_disk('$SRC')['train'].select(range($N))
draft = [{'task_type':'drafting','images':r['images'],'current_html':'','target_html':r['target_html'],'instruction':''} for r in src]
bench = [{'image':r['images'][0],'text':r['target_html']} for r in src]
fd = Features({'task_type':Value('string'),'images':Sequence(Image()),'current_html':Value('string'),'target_html':Value('string'),'instruction':Value('string')})
d = Dataset.from_list(draft, features=fd)
DatasetDict({'train':d,'validation':d.select(range(min(6,len(d))))}).save_to_disk('$TRAIN_DS_C')
Dataset.from_list(bench, features=Features({'image':Image(),'text':Value('string')})).save_to_disk('$BENCH_DS_C')
ls = sorted(len(x['target_html']) for x in draft)
print('готово:', len(draft), 'сэмплов; длина target_html: медиана', ls[len(ls)//2], 'max', ls[-1])
" > "$BASE/logs/wc2m/wc2m_build.log" 2>&1
  say "сборка: $(grep -o 'готово.*' $BASE/logs/wc2m/wc2m_build.log 2>/dev/null || echo 'см. wc2m_build.log')"
fi
NREAL=$(docker run --rm -v /mnt/storage-1:/storage --entrypoint /opt/venv/bin/python sft \
        -c "from datasets import load_from_disk;print(len(load_from_disk('$BENCH_DS_C')))" 2>/dev/null)
[[ -n "$NREAL" ]] || { say "датасет не собрался — см. logs/wc2m/wc2m_build.log"; exit 1; }
say "сэмплов: $NREAL | lr $LR, $EPOCHS эпох, чекпоинт каждую"
mkchmod "$OUT"

ckpts(){ ls -d "$OUT"/checkpoint-*/ "$OUT"/*/checkpoint-*/ 2>/dev/null |
         sed 's:/$::' | awk -F'checkpoint-' '{print $NF" "$0}' | sort -n | cut -d' ' -f2-; }

# --- 2. обучение ---
if [[ -z "$(ckpts)" ]]; then
  say "обучение full-FT lr $LR constant..."
  DATA_DIR="$BASE/data" HF_CACHE="$BASE/hf_cache" OUT_DIR="$OUT" \
  CONTAINER_HOME="$BASE/container-home" GPUS="$GPUS" \
    "$REPO/SFT/run.sh" torchrun --nproc_per_node="$NPROC" --tee 3 -m train.train_sft \
      --config configs/full_ft_qwen3_5_4b.yaml --dataset_name /data/wc2m_short \
      --num_train_epochs "$EPOCHS" --learning_rate "$LR" \
      --lr_scheduler_type constant --warmup_ratio 0 \
      --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
      --eval_strategy no --save_strategy epoch --save_total_limit "$EPOCHS" \
      --output_dir /out > "$BASE/logs/wc2m/wc2m_train.log" 2>&1
  say "обучение: rc=$?"
fi

# --- 3. бенч: сначала база (точка отсчёта на ЭТОМ наборе), потом чекпоинты ---
bench(){ local name="$1" model="$2"; local bdir="$OUT/bench-$name"
  [[ -f "$bdir/summary.json" ]] && { say "$name: уже есть"; return; }
  mkchmod "$bdir"
  say "$name: greedy-бенч $model ..."
  local mnt=""; [[ -d "$model" ]] && mnt="$model"
  env IMAGE_TAG=design2code-bench:latest HOST_OUTDIR="$bdir" HOST_HF_CACHE="$BASE/hf_cache" \
      ${mnt:+HOST_MODEL_DIR="$mnt"} CONTAINER_NAME="wc2m-$name" GPUS="$BENCH_GPUS" CLEARML_DISABLE=1 \
      "$REPO/Evaluation/run.sh" --no-resume --model "$model" \
        --hf-dataset "$BENCH_DS_E" --hf-config default --hf-split train \
        --n-samples "$NREAL" --batch-size "$NREAL" --n-examples-per-batch "$NREAL" \
        --temperature 0 --max-pixels 2097152 --tensor-parallel-size "$TP" \
        --gpu-memory-utilization 0.9 --max-new-tokens 16384 --max-model-len 24384 --num-workers 8 \
      > "$BASE/logs/wc2m/wc2m_bench_$name.log" 2>&1
  docker rm "wc2m-$name" >/dev/null 2>&1
  local sc; sc=$(python3 -c "import json;print(round(json.load(open('$bdir/summary.json'))['final_score'],3))" 2>/dev/null)
  say "$name: final_score = ${sc:-НЕТ}"
}

bench base "Qwen/Qwen3.5-4B"
for CKPT in $(ckpts); do
  STEP=$(basename "$CKPT" | sed 's/checkpoint-//'); EP=$(( STEP / 5 ))
  [[ -f "$CKPT/config.json" ]] || { say "ep$EP ($STEP): весов нет"; continue; }
  bench "ep$EP" "$CKPT"
done

say "=== СВОДКА WEBCODE2M (greedy) ==="
python3 - "$OUT" <<'PY' | tee -a "$LOG"
import json, os, sys, glob, csv
out = sys.argv[1]
def key(p):
    n = os.path.basename(p).replace("bench-", "")
    return (0, 0) if n == "base" else (1, int(n.replace("ep", "")))
for d in sorted(glob.glob(os.path.join(out, "bench-*")), key=key):
    f = os.path.join(d, "summary.json")
    if not os.path.exists(f):
        continue
    s = json.load(open(f))
    zeros = "-"
    rc = os.path.join(d, "results.csv")
    if os.path.exists(rc):
        v = [float(r["final_score"]) for r in csv.DictReader(open(rc)) if r.get("final_score")]
        if v:
            zeros = "%d/%d" % (sum(1 for x in v if x < 0.01), len(v))
    print("  %-6s final %.3f  block %.3f  text %.3f  pos %.3f  color %.3f  clip %.3f  нулей %s" % (
        os.path.basename(d).replace("bench-", ""), s["final_score"], s["block_match"],
        s["text"], s["position"], s["color"], s["clip"], zeros))
PY
