#!/usr/bin/env bash
# Очередь пилотных экспериментов на 1000 примерах WebCode2M: для каждого
# обучение -> (LoRA: merge) -> бенч Design2Code. Всё уходит в ClearML, задачи
# обучения и бенча связаны (см. SFT/train/tracking.py).
#
# Запуск из /workspace/SFT внутри SFT-контейнера:
#   nohup bash scripts/run_experiments.sh > exps.out 2>&1 &
#   tail -f exps.out
#
# Падение одного эксперимента не останавливает очередь: шаг отчитывается в
# REPORT.txt и управление идёт дальше — утром видно, что прошло, а что нет.
#
# Переменные: DATASET, OUT_DIR, NPROC, ONLY (напр. ONLY="E1 E4"), BENCH_IMAGE.
set -uo pipefail

cd "$(dirname "$0")/.."

DATASET="${DATASET:-/data/webcode2m_1000_split}"
NPROC="${NPROC:-2}"
BASE=$([[ -d /out ]] && echo /out || echo .)
RESULT_DIR="${RESULT_DIR:-$BASE/exps-$(date +%Y%m%d-%H%M%S)}"
BENCH_IMAGE="${BENCH_IMAGE:-design2code-bench}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
LOGS="$RESULT_DIR/logs"
REPORT="$RESULT_DIR/REPORT.txt"
mkdir -p "$LOGS"

say()  { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$REPORT"; }
rule() { printf '%s\n' "--------------------------------------------------------" | tee -a "$REPORT"; }

# Общее для всех: 1000 примеров -> при эфф. батче 64 выходит всего 46 шагов
# оптимизатора на 3 эпохи, для сходимости мало. Берём эфф. батч 16 (187 шагов).
# Микробатч 4: замеры (SFT/THROUGHPUT.md) дают на нём максимум, bs8 вдвое хуже.
COMMON=(
  --dataset_name "$DATASET"
  --num_train_epochs 3
  --per_device_train_batch_size 4
  --gradient_accumulation_steps 2
  --report_to clearml
)

# ID | конфиг | S-ID плана | доп. аргументы | пиксель-бюджет (SFT_MAX_PIXELS)
EXPERIMENTS=(
  "E1|configs/lora_ft_qwen3_5_4b.yaml|S0|--learning_rate 1e-4|2097152"
  "E2|configs/full_ft_qwen3_5_4b.yaml|S5|--learning_rate 1e-5|2097152"
  "E3|configs/lora_ft_qwen3_5_4b.yaml|S8|--learning_rate 3e-4|2097152"
  "E4|configs/lora_ft_qwen3_5_4b.yaml|S2|--learning_rate 1e-4|3932160"
  "E5|configs/lora_ft_qwen3_5_4b.yaml|S10|--learning_rate 1e-4 --lora_r 64 --lora_alpha 128|2097152"
)

say "каталог: $RESULT_DIR"
say "датасет: $DATASET | GPU: $NPROC | образ бенча: $BENCH_IMAGE"
rule

for row in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r EID CONFIG SID EXTRA PIXELS <<< "$row"
  if [[ -n "${ONLY:-}" && " $ONLY " != *" $EID "* ]]; then
    say "$EID пропущен (ONLY=$ONLY)"; continue
  fi

  OUT="$RESULT_DIR/$EID"
  say "=== $EID ($SID) — $CONFIG $EXTRA"

  # --- обучение -------------------------------------------------------------
  # CLEARML_TAGS попадают в теги задачи -> в UI фильтруешь свип по S-ID.
  start=$SECONDS
  CLEARML_TAGS="$EID,$SID,pilot1k" \
  SFT_MAX_PIXELS="$PIXELS" \
  CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-EAGER}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /opt/venv/bin/torchrun --nproc_per_node="$NPROC" --tee 3 \
    -m train.train_sft --config "$CONFIG" "${COMMON[@]}" $EXTRA \
    --output_dir "$OUT" > "$LOGS/$EID.train.log" 2>&1
  rc=$?
  say "$EID обучение: rc=$rc, $(( (SECONDS-start)/60 )) мин"
  if [[ $rc -ne 0 ]]; then
    say "$EID ПРОПУЩЕН дальше (см. $LOGS/$EID.train.log)"; rule; continue
  fi

  # --- LoRA: слить адаптер, иначе vLLM не увидит веса -----------------------
  MODEL_DIR="$OUT"
  if [[ -f "$OUT/adapter_config.json" ]]; then
    MODEL_DIR="$HF_CACHE/merged-$EID"
    /opt/venv/bin/python -m scripts.merge_lora "$OUT" "$MODEL_DIR" \
      > "$LOGS/$EID.merge.log" 2>&1
    rc=$?
    say "$EID merge_lora: rc=$rc -> $MODEL_DIR"
    [[ $rc -ne 0 ]] && { say "$EID бенч пропущен"; rule; continue; }
  fi

  # --- бенч Design2Code -----------------------------------------------------
  # max_pixels ДОЛЖЕН совпадать с обучением, иначе чекпоинт меряется вне
  # своего распределения (H1 из плана). Для E4 бюджет другой.
  start=$SECONDS
  docker run --rm --gpus all --shm-size 2g --ipc=host \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -v "$RESULT_DIR/$EID-bench":/app/output \
    -e CLEARML_API_ACCESS_KEY -e CLEARML_API_SECRET_KEY -e CLEARML_API_HOST \
    -e CLEARML_TAGS="$EID,$SID,pilot1k" \
    "$BENCH_IMAGE" python run_benchmark_batched.py \
      --model "$MODEL_DIR" \
      --hf-dataset SALT-NLP/Design2Code-hf --hf-config default --hf-split train \
      --n-samples 484 --batch-size 484 \
      --max-pixels "$PIXELS" \
      --outdir /app/output > "$LOGS/$EID.bench.log" 2>&1
  say "$EID бенч: rc=$?, $(( (SECONDS-start)/60 )) мин"
  rule
done

say "очередь закончена. Итоги: $REPORT"
say "сводки: $RESULT_DIR/*/summary.json, в ClearML — тег pilot1k"
