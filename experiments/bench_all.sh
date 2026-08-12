#!/usr/bin/env bash
# Прогнать бенч Design2Code по ВСЕМ готовым чекпоинтам пилота.
#
# Можно запускать, когда run_pilot.sh ещё работает: скрипт сам дождётся
# строки «очередь закончена» в REPORT.txt и только потом начнёт. То есть
# ставишь его рядом и уходишь.
#
#   ./bench_all.sh                        # ждёт свежий exps-*, потом бенчит всё
#   ./bench_all.sh exps-20260803-104501   # конкретный каталог
#   NOWAIT=1 ./bench_all.sh               # не ждать, начать сразу
#   ONLY="E1 E5" ./bench_all.sh           # только часть
#
# Для LoRA сам домержит адаптер (если *-merged ещё нет); full-FT чекпоинт
# бенчится напрямую. Уже посчитанные бенчи (есть summary.json) пропускаются,
# так что скрипт безопасно перезапускать после падения.
#
# Пиксель-бюджет НЕ угадывается: берётся из meta в логе обучения, чтобы бенч
# шёл на том же разрешении, что и обучение (иначе чекпоинт меряется вне своего
# распределения — H1 из плана экспериментов).
set -uo pipefail

cd "$(dirname "$0")/.."   # скрипт лежит в experiments/, работаем от корня репо
REPO="$PWD"
source "$REPO/experiments/lib/common.sh"

RESULT_DIR="${1:-$(ls -td "$REPO"/exps-* 2>/dev/null | head -1)}"
[[ -n "$RESULT_DIR" && -d "$RESULT_DIR" ]] || { echo "не найден каталог exps-*"; exit 1; }
RESULT_DIR="$(cd "$RESULT_DIR" && pwd)"

HF_CACHE="${HF_CACHE:-/mnt/storage-1/hf_cache}"
DATA_DIR="${DATA_DIR:-/mnt/storage-1/data}"
GPUS="${GPUS:-\"device=0,3\"}"
BENCH_TP="${BENCH_TP:-2}"
BENCH_IMAGE="${BENCH_IMAGE:-design2code-bench:latest}"
BENCH_DATASET="${BENCH_DATASET:-SALT-NLP/Design2Code-hf}"
BENCH_N="${BENCH_N:-484}"
# Доля памяти карты под vLLM. Оставлять ей всю карту НЕЛЬЗЯ: метрики считаются
# в 16 процессах, каждый грузит CLIP на ту же GPU (metrics.py:51). При 0.90 им
# не остаётся памяти, воркеры падают на инициализации CUDA, и до оценки доходят
# единицы сэмплов (замерено: 33 из 484 против 484 из 484 при 0.5).
# Половина карты — то значение, на котором сняты базовые числа.
GPU_UTIL="${GPU_UTIL:-0.50}"
# Лимит генерации. Дефолт скрипта бенча — 8192, и на нём обрезается до 60%
# страниц (у E3 292 из 484): длинный HTML не помещается, метрики занижаются
# незаслуженно. 16384 влезает в max-model-len 24384 вместе с картинкой
# (2048 визуальных токенов, у 3.93 Мп — 3840) и промптом.
BENCH_MAX_NEW="${BENCH_MAX_NEW:-16384}"
DEFAULT_PIXELS=2097152
# База для сравнения — тот же прогон, что и дообученные: одинаковый лимит
# токенов и пиксель-бюджет, иначе дельта к базе ничего не значит.
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-4B}"

LOGS="$RESULT_DIR/logs"; mkdir -p "$LOGS"
REPORT="$RESULT_DIR/REPORT-bench.txt"
RUN_LOG="$REPORT"

# train_sft.py дописывает к output_dir имя рана (train_sft.py:167), поэтому веса
# лежат не в E<N>/, а в E<N>/<config>_s42_<дата>/. Ищем каталог с весами: сперва
# сам OUT, иначе самый свежий подкаталог с config.json или adapter_config.json.
# Слитая модель готова, только если рядом с весами лежит процессор. Имя файла
# разное у разных семейств, поэтому принимаем любое из двух.
merged_complete() {
  [[ -f "$1/processor_config.json" || -f "$1/preprocessor_config.json" ]]
}

resolve_weights() {
  local dir="$1"
  if [[ -f "$dir/config.json" || -f "$dir/adapter_config.json" ]]; then
    echo "$dir"; return 0
  fi
  local sub
  # ls -dt: свежие первыми, если ранов в каталоге почему-то несколько
  for sub in $(ls -dt "$dir"/*/ 2>/dev/null); do
    sub="${sub%/}"
    if [[ -f "$sub/config.json" || -f "$sub/adapter_config.json" ]]; then
      echo "$sub"; return 0
    fi
  done
  return 1
}

# Ждём конца обучения
if [[ "${NOWAIT:-0}" != "1" ]]; then
  if ! grep -q "очередь закончена" "$RESULT_DIR/REPORT.txt" 2>/dev/null; then
    say "жду, пока run_pilot.sh закончит очередь (проверка раз в минуту)..."
    while ! grep -q "очередь закончена" "$RESULT_DIR/REPORT.txt" 2>/dev/null; do
      sleep 60
    done
  fi
  say "очередь обучения завершена, начинаю бенчи"
fi

# Проверки
# Образ должен быть СВЕЖИЙ: старый не знает ни --max-pixels, ни tracking.py, и
# бенч падал бы с argparse-ошибкой (rc=2) — ровно то, что случилось в прогоне.
if ! docker run --rm --entrypoint ls "$BENCH_IMAGE" tracking.py >/dev/null 2>&1; then
  say "ОШИБКА: в образе $BENCH_IMAGE нет tracking.py — он собран до правок."
  say "  Пересобери: cd Evaluation && ./build.sh"
  exit 1
fi
if ! docker run --rm "$BENCH_IMAGE" --help 2>&1 | grep -q -- --max-pixels; then
  say "ОШИБКА: образ не понимает --max-pixels. Пересобери: cd Evaluation && ./build.sh"
  exit 1
fi
[[ -n "${CLEARML_API_ACCESS_KEY:-}" ]] || say "ВНИМАНИЕ: кред ClearML нет — метрики будут только в summary.json"

say "каталог: $RESULT_DIR | GPU: $GPUS | TP: $BENCH_TP | память $GPU_UTIL | max_new_tokens $BENCH_MAX_NEW"

# --- база (без обучения) ---
if [[ -n "$BASE_MODEL" && ! -f "$RESULT_DIR/E0-base-bench/summary.json" ]]; then
  say "E0: база $BASE_MODEL, ${DEFAULT_PIXELS} px"
  start=$SECONDS
  env IMAGE_TAG="$BENCH_IMAGE" HOST_OUTDIR="$RESULT_DIR/E0-base-bench" \
      HOST_HF_CACHE="$HF_CACHE" CONTAINER_NAME="bench-E0" GPUS="$GPUS" \
      CLEARML_TAGS="E0,base,pilot1k" \
      "$REPO/Evaluation/run.sh" --model "$BASE_MODEL" \
        --hf-dataset "$BENCH_DATASET" --hf-config default --hf-split train \
        --n-samples "$BENCH_N" --batch-size "$BENCH_N" \
        --max-pixels "$DEFAULT_PIXELS" --tensor-parallel-size "$BENCH_TP" \
        --gpu-memory-utilization "$GPU_UTIL" --max-new-tokens "$BENCH_MAX_NEW" \
      > "$LOGS/E0.bench.log" 2>&1
  say "E0 бенч: rc=$?, $(( (SECONDS-start)/60 )) мин"
fi

for OUT in "$RESULT_DIR"/E[0-9]*; do
  [[ -d "$OUT" ]] || continue
  EID="$(basename "$OUT")"
  case "$EID" in *-merged|*-bench|*-bench-*) continue ;; esac
  if [[ -n "${ONLY:-}" && " $ONLY " != *" $EID "* ]]; then continue; fi

  if [[ -f "$RESULT_DIR/$EID-bench/summary.json" ]]; then
    say "$EID уже отбенчен — пропускаю"; continue
  fi

  # Бюджет пикселей из meta обучения; если лога нет — дефолт.
  PIXELS=$(grep -ho '"image_pixels": *[0-9]*' "$LOGS/$EID.train.log" 2>/dev/null \
           | head -1 | grep -o '[0-9]*')
  PIXELS="${PIXELS:-$DEFAULT_PIXELS}"

  WEIGHTS=$(resolve_weights "$OUT") || {
    say "$EID: весов нет ни в $EID/, ни в подкаталогах — пропускаю"; continue; }
  [[ "$WEIGHTS" != "$OUT" ]] && say "$EID: веса в $(basename "$WEIGHTS")/"

  # LoRA -> нужен мердж; full-FT чекпоинт vLLM грузит напрямую.
  MODEL="$WEIGHTS"
  if [[ -f "$WEIGHTS/adapter_config.json" ]]; then
    MODEL="$RESULT_DIR/$EID-merged"
    # Полнота, а не только наличие: merge_lora пишет веса, а процессор кладёт
    # ПОСЛЕ них. Прерванный мердж оставляет config.json без процессора, и vLLM
    # падает с "Can't load image processor". Имя файла процессора зависит от
    # семейства: Qwen3.5 пишет processor_config.json, старые VL-модели —
    # preprocessor_config.json. Принимаем оба.
    if [[ -f "$MODEL/config.json" ]] && merged_complete "$MODEL"; then
      say "$EID: слитая модель уже есть"
    elif [[ -d "$MODEL" ]]; then
      say "$EID: слитая модель неполная — мержу заново"
      docker run --rm -v /mnt/storage-1:/storage --entrypoint rm sft -rf \
        "/storage/${MODEL#/mnt/storage-1/}" 2>/dev/null
    fi
    if [[ ! -f "$MODEL/config.json" ]] || ! merged_complete "$MODEL"; then
      say "$EID: мержу LoRA-адаптер..."
      # пути внутри контейнера: $RESULT_DIR смонтирован как /out
      REL="${WEIGHTS#$RESULT_DIR/}"
      env DATA_DIR="$DATA_DIR" HF_CACHE="$HF_CACHE" OUT_DIR="$RESULT_DIR" GPUS="$GPUS" \
        "$REPO/SFT/run.sh" python -m scripts.merge_lora "/out/$REL" "/out/$EID-merged" \
        > "$LOGS/$EID.merge.log" 2>&1
      rc=$?
      say "$EID merge_lora: rc=$rc"
      [[ $rc -ne 0 ]] && { say "$EID пропущен (лог: $LOGS/$EID.merge.log)"; continue; }
      if ! merged_complete "$MODEL"; then
        say "$EID: мердж вернул 0, но процессора нет — в бенч не отдаю"
        continue
      fi
    fi
  fi

  say "$EID: бенч, ${PIXELS} px, модель $(basename "$MODEL")"
  start=$SECONDS
  env IMAGE_TAG="$BENCH_IMAGE" \
      HOST_OUTDIR="$RESULT_DIR/$EID-bench" \
      HOST_HF_CACHE="$HF_CACHE" \
      HOST_MODEL_DIR="$MODEL" \
      CONTAINER_NAME="bench-$EID" GPUS="$GPUS" \
      CLEARML_TAGS="$EID,pilot1k" \
      "$REPO/Evaluation/run.sh" \
        --model "$MODEL" \
        --hf-dataset "$BENCH_DATASET" --hf-config default --hf-split train \
        --n-samples "$BENCH_N" --batch-size "$BENCH_N" \
        --max-pixels "$PIXELS" \
        --tensor-parallel-size "$BENCH_TP" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --max-new-tokens "$BENCH_MAX_NEW" \
      > "$LOGS/$EID.bench.log" 2>&1
  say "$EID бенч: rc=$?, $(( (SECONDS-start)/60 )) мин"
done

say "все бенчи закончены"
echo
python3 "$REPO/SFT/scripts/collect_results.py" "$RESULT_DIR" | tee -a "$REPORT"
