#!/usr/bin/env bash
# Sanity-overfit на Design2Code: обучаем на первых N сэмплах Design2Code и
# бенчим на тех же N. Диагностика: если после жёсткого переобучения модель
# не бьёт базу на сэмплах, которые она видела, — сломан пайплайн, а не данные.
# Если бьёт — пайплайн исправен, и деградация на WebCode2M — проблема данных.
set -uo pipefail
cd "$(dirname "$0")"; REPO="$PWD"

# Общий диск. Путь монтирования переопределяется через STORAGE, дефолт — тот же,
# что был вбит раньше, поэтому поведение прогонов не меняется.
: "${STORAGE:=/mnt/storage-1}"
BASE="$STORAGE/Screenshot2Code"
mkdir -p "$BASE/logs/sanity"
N="${N:-64}"; EPOCHS="${EPOCHS:-12}"; LR="${LR:-1e-5}"
GPUS="${GPUS:-\"device=0,1\"}"; NPROC="${NPROC:-2}"
DATA="$BASE/data/d2c_overfit"
OUT="$BASE/checkpoints_exps/d2c-overfit"
LOG="$BASE/logs/sanity/SANITY.log"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# --- 1. собрать датасет из первых N Design2Code (та же выборка, что бенч) ---
if [[ ! -d "$DATA/train" ]]; then
  say "строю overfit-датасет: первые $N Design2Code (seed 0, buffer 10000)"
  docker run --rm -v /mnt/storage-1:/storage -e HF_HOME=$BASE/hf_cache \
    --entrypoint /opt/venv/bin/python sft -c "
from datasets import load_dataset, Dataset, DatasetDict, Features, Sequence, Image, Value
ds = load_dataset('SALT-NLP/Design2Code-hf', name='default', split='train', streaming=True)
ds = ds.shuffle(seed=0, buffer_size=10000)
rows=[]
for r in ds:
    rows.append({'task_type':'drafting','images':[r['image']],'current_html':'',
                 'target_html':r['text'],'instruction':''})
    if len(rows)>=$N: break
feat = Features({'task_type':Value('string'),'images':Sequence(Image()),
                 'current_html':Value('string'),'target_html':Value('string'),
                 'instruction':Value('string')})
d = Dataset.from_list(rows, features=feat)
DatasetDict({'train':d,'validation':d.select(range(min(8,len(d))))}).save_to_disk('/storage/Screenshot2Code/data/d2c_overfit')
print('готово:', len(d), 'сэмплов')
" > "$BASE/logs/sanity/d2c_build.log" 2>&1
  say "сборка датасета: rc=$? ($(grep -o 'готово.*' $BASE/logs/sanity/d2c_build.log 2>/dev/null))"
fi
[[ -d "$DATA/train" ]] || { say "датасет не собрался — см. $BASE/logs/sanity/d2c_build.log"; exit 1; }

# --- 2. обучение (жёсткий overfit) ---
if [[ ! -d "$OUT" ]] || ! ls "$OUT"/*/config.json >/dev/null 2>&1; then
  say "обучение: full-FT lr $LR, $EPOCHS эпох, $N сэмплов"
  DATA_DIR="$BASE/data" HF_CACHE="$BASE/hf_cache" OUT_DIR="$OUT" \
  CONTAINER_HOME="$BASE/container-home" GPUS="$GPUS" \
    "$REPO/SFT/run.sh" torchrun --nproc_per_node="$NPROC" --tee 3 -m train.train_sft \
      --config configs/full_ft_qwen3_5_4b.yaml \
      --dataset_name /data/d2c_overfit \
      --num_train_epochs "$EPOCHS" --learning_rate "$LR" \
      --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
      --eval_strategy no --save_strategy no \
      --output_dir /out > "$BASE/logs/sanity/d2c_train.log" 2>&1
  say "обучение: rc=$?"
fi
WEIGHTS=$(ls -dt "$OUT"/*/ 2>/dev/null | head -1); WEIGHTS="${WEIGHTS%/}"
[[ -f "$WEIGHTS/config.json" ]] || { say "весов нет — см. $BASE/logs/sanity/d2c_train.log"; exit 1; }
say "веса: $WEIGHTS"

# --- 3. бенч базы и overfit на тех же N ---
bench() {  # $1=имя $2=модель $3=монтировать?
  local name="$1" model="$2"
  say "бенч $name на $N сэмплах..."
  local mnt=""; [[ -d "$model" ]] && mnt="$model"
  env IMAGE_TAG=design2code-bench:latest \
      HOST_OUTDIR="$OUT/${name}-bench" HOST_HF_CACHE="$BASE/hf_cache" \
      ${mnt:+HOST_MODEL_DIR="$mnt"} CONTAINER_NAME="bench-sanity-$name" GPUS="$GPUS" \
      "$REPO/Evaluation/run.sh" --model "$model" \
        --hf-dataset SALT-NLP/Design2Code-hf --hf-config default --hf-split train \
        --n-samples "$N" --batch-size "$N" --seed 0 --shuffle-buffer-size 10000 \
        --max-pixels 2097152 --tensor-parallel-size "$NPROC" \
        --gpu-memory-utilization 0.5 --max-new-tokens 16384 \
      > "$BASE/d2c_bench_${name}.log" 2>&1
  say "бенч $name: rc=$?"
}
[[ -f "$OUT/base-bench/summary.json" ]] || bench base "Qwen/Qwen3.5-4B"
[[ -f "$OUT/overfit-bench/summary.json" ]] || bench overfit "$WEIGHTS"

# --- 4. вердикт ---
say "=== РЕЗУЛЬТАТ sanity-overfit ==="
/opt/venv/bin/python - <<PY 2>/dev/null || python3 - <<PY
import json
def g(p):
    try: return json.load(open(p))
    except: return None
b=g("$OUT/base-bench/summary.json"); o=g("$OUT/overfit-bench/summary.json")
if b and o:
    for k in ("final_score","block_match","text","position","color","clip"):
        print(f"  {k:12} база {b.get(k,0):.3f}  overfit {o.get(k,0):.3f}  Δ {o.get(k,0)-b.get(k,0):+.3f}")
    d=o["final_score"]-b["final_score"]
    print()
    print("  ВЕРДИКT:", "пайплайн ИСПРАВЕН (overfit > базы) -> виноваты данные WebCode2M" if d>0.02
          else "КРАСНЫЙ ФЛАГ: overfit не бьёт базу на своих же сэмплах -> копать в рецепт/харнесс")
else:
    print("  нет одной из сводок")
PY
