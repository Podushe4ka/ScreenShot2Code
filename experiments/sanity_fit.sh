#!/usr/bin/env bash
# Исправленный sanity-overfit: обучаем на КОРОТКИХ Design2Code (влезают в
# max_length) и бенчим на ТЕХ ЖЕ сэмплах. Если модель после переобучения не
# бьёт базу на данных, что она видела, — сломан рецепт/харнесс. Если бьёт —
# пайплайн исправен, а деградация на WebCode2M — проблема данных/домена.
#
# Запуск на ХОСТЕ (a100-3, GPU 1,2 свободны):
#   GPUS='"device=1,2"' NPROC=2 ./sanity_fit.sh
set -uo pipefail
cd "$(dirname "$0")"; REPO="$PWD"
# Общий диск. Путь монтирования переопределяется через STORAGE, дефолт — тот же,
# что был вбит раньше, поэтому поведение прогонов не меняется.
: "${STORAGE:=/mnt/storage-1}"
BASE="$STORAGE/Screenshot2Code"
mkdir -p "$BASE/logs/sanity"
N="${N:-40}"; EPOCHS="${EPOCHS:-15}"; LR="${LR:-2e-5}"; MAXCHARS="${MAXCHARS:-40000}"
GPUS="${GPUS:-\"device=1,2\"}"; NPROC="${NPROC:-2}"
TRAIN_DS="$BASE/data/d2c_short"                         # drafting-формат (обучение)
BENCH_DS="$BASE/hf_cache/d2c_short_bench"               # image/text (под mount HF-кэша)
OUT="$BASE/checkpoints_exps/d2c-sanity-fit"
LOG="$BASE/logs/sanity/SANITYFIT.log"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# Контейнерные пути: внутри docker /mnt/storage-1 смонтирован как /storage.
# save_to_disk ДОЛЖЕН писать по /storage/... (bind mount), иначе данные уходят
# в эфемерную ФС контейнера и исчезают с ним (ровно это уронило первый запуск).
TRAIN_DS_C="${TRAIN_DS/\/mnt\/storage-1//storage}"
BENCH_DS_C="${BENCH_DS/\/mnt\/storage-1//storage}"

# --- 1. отфильтровать короткие Design2Code, сохранить оба формата ---
if [[ ! -d "$TRAIN_DS/train" ]]; then
  say "фильтрую Design2Code до коротких (<$MAXCHARS симв), беру $N; seed 0"
  docker run --rm -v /mnt/storage-1:/storage -e HF_HOME=$BASE/hf_cache \
    --entrypoint /opt/venv/bin/python sft -c "
from datasets import load_dataset, Dataset, DatasetDict, Features, Sequence, Image, Value
ds = load_dataset('SALT-NLP/Design2Code-hf', name='default', split='train', streaming=True).shuffle(seed=0, buffer_size=10000)
draft, bench = [], []
for r in ds:
    if len(r['text']) > $MAXCHARS: continue
    draft.append({'task_type':'drafting','images':[r['image']],'current_html':'','target_html':r['text'],'instruction':''})
    bench.append({'image':r['image'],'text':r['text']})
    if len(draft) >= $N: break
fd = Features({'task_type':Value('string'),'images':Sequence(Image()),'current_html':Value('string'),'target_html':Value('string'),'instruction':Value('string')})
d = Dataset.from_list(draft, features=fd)
DatasetDict({'train':d,'validation':d.select(range(min(6,len(d))))}).save_to_disk('$TRAIN_DS_C')
Dataset.from_list(bench, features=Features({'image':Image(),'text':Value('string')})).save_to_disk('$BENCH_DS_C')
print('готово:', len(draft), 'сэмплов')
" > "$BASE/logs/sanity/sfit_build.log" 2>&1
  say "сборка: $(grep -o 'готово.*' $BASE/logs/sanity/sfit_build.log 2>/dev/null || echo 'см. sfit_build.log')"
fi
NREAL=$(docker run --rm -v /mnt/storage-1:/storage --entrypoint /opt/venv/bin/python sft -c "from datasets import load_from_disk;print(len(load_from_disk('$BENCH_DS_C')))" 2>/dev/null)
[[ -n "$NREAL" ]] || { say "датасет не собрался"; exit 1; }
say "сэмплов в наборе: $NREAL"

# --- 2. жёсткий overfit (рецепт overfit20: lr 2e-5, constant) ---
if ! ls "$OUT"/*/config.json >/dev/null 2>&1; then
  say "обучение: full-FT lr $LR constant, $EPOCHS эпох, $NREAL сэмплов"
  DATA_DIR="$BASE/data" HF_CACHE="$BASE/hf_cache" OUT_DIR="$OUT" \
  CONTAINER_HOME="$BASE/container-home" GPUS="$GPUS" \
    "$REPO/SFT/run.sh" torchrun --nproc_per_node="$NPROC" --tee 3 -m train.train_sft \
      --config configs/full_ft_qwen3_5_4b.yaml --dataset_name /data/d2c_short \
      --num_train_epochs "$EPOCHS" --learning_rate "$LR" --lr_scheduler_type constant --warmup_ratio 0 \
      --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
      --eval_strategy no --save_strategy no --output_dir /out > "$BASE/logs/sanity/sfit_train.log" 2>&1
  say "обучение: rc=$?"
fi
W=$(ls -dt "$OUT"/*/ 2>/dev/null | head -1); W="${W%/}"
[[ -f "$W/config.json" ]] || { say "весов нет — см. sfit_train.log"; exit 1; }

# --- 3. бенч базы и overfit на ТЕХ ЖЕ сэмплах (локальный датасет) ---
bench(){ local name="$1" model="$2"
  say "бенч $name на $NREAL (локальный набор)..."
  local mnt=""; [[ -d "$model" ]] && mnt="$model"
  env IMAGE_TAG=design2code-bench:latest HOST_OUTDIR="$OUT/${name}-bench" \
      HOST_HF_CACHE="$BASE/hf_cache" ${mnt:+HOST_MODEL_DIR="$mnt"} \
      CONTAINER_NAME="sfit-$name" GPUS="$GPUS" CLEARML_DISABLE=1 \
      "$REPO/Evaluation/run.sh" --model "$model" \
        --hf-dataset /root/.cache/huggingface/d2c_short_bench --hf-config default --hf-split train \
        --n-samples "$NREAL" --batch-size "$NREAL" --n-examples-per-batch 6 \
        --max-pixels 2097152 --tensor-parallel-size "$NPROC" \
        --gpu-memory-utilization 0.5 --max-new-tokens 16384 --max-model-len 24384 \
      > "$BASE/sfit_bench_${name}.log" 2>&1
  say "бенч $name: rc=$?"
}
[[ -f "$OUT/base-bench/summary.json" ]] || bench base "Qwen/Qwen3.5-4B"
[[ -f "$OUT/overfit-bench/summary.json" ]] || bench overfit "$W"

# --- 4. вердикт ---
say "=== РЕЗУЛЬТАТ sanity-fit ==="
docker run --rm -v /mnt/storage-1:/storage --entrypoint /opt/venv/bin/python sft -c "
import json
b=json.load(open('$OUT/base-bench/summary.json')); o=json.load(open('$OUT/overfit-bench/summary.json'))
for k in ('final_score','block_match','text','position','color','clip'):
    print(f'  {k:12} база {b.get(k,0):.3f}  overfit {o.get(k,0):.3f}  Δ {o.get(k,0)-b.get(k,0):+.3f}')
d=o['final_score']-b['final_score']
print('  ВЕРДИКТ:', 'пайплайн ИСПРАВЕН (overfit > базы) -> виноваты данные/домен' if d>0.02 else 'КРАСНЫЙ ФЛАГ: overfit не бьёт базу на своих же -> рецепт/харнесс')
" 2>&1 | grep -vE "Warning|warn" | tee -a "$LOG"
