#!/usr/bin/env bash
# Пилот на 1000 примерах WebCode2M. Каждый эксперимент: обучение ->
# merge_lora (только LoRA; full-FT чекпоинт vLLM грузит напрямую) -> бенч
# Design2Code. Всё в ClearML, задачи обучения и бенча связаны.
#
# Запускается НА ХОСТЕ, НЕ внутри контейнера: сам поднимает контейнеры через
# SFT/run.sh и Evaluation/run.sh. Изнутри контейнера работать не будет —
# docker в нём недоступен.
#
#   ./run_pilot.sh                 # волна 1: LoRA vs full-FT, 4 рана, ~8.9 ч
#   WAVE=2 ./run_pilot.sh          # волна 2: пиксель-бюджет и r64, ~4.6 ч
#   DRY_RUN=1 ./run_pilot.sh       # показать команды, ничего не запуская
#   ONLY="E1 E3" ./run_pilot.sh    # только часть
#   SKIP_BENCH=1 ./run_pilot.sh    # только обучение, без бенча
#
# Обязательное окружение: CLEARML_API_ACCESS_KEY, CLEARML_API_SECRET_KEY —
# иначе трекинг молча выключится и смотреть утром будет нечего (скрипт
# проверяет это до старта и падает сразу, а не через девять часов).
#
# Что переопределяется: DATA_DIR, DATASET_NAME, HF_CACHE, GPUS, NPROC,
# BENCH_TP, BENCH_N, BENCH_IMAGE, RESULT_DIR.
#
# ВНИМАНИЕ про права: файлы, созданные контейнером, принадлежат root
# (на общем диске sticky-бит), обычным юзером их не удалить. Чистить
# результаты — из контейнера, напр.:
#   docker run --rm -v "$PWD":/w --entrypoint rm sft -rf /w/exps-СТАРЫЙ
#
# Падение одного эксперимента не останавливает очередь — шаг пишется в
# REPORT.txt, управление идёт дальше.
set -uo pipefail

cd "$(dirname "$0")"
REPO="$PWD"

DATA_DIR="${DATA_DIR:-/mnt/storage-1/data}"
DATASET_NAME="${DATASET_NAME:-webcode2m_1000_split}"
# HF-кэш на общем диске, а не в $HOME: модели по 8-18 ГБ, и качать их
# повторно в каждый контейнер незачем.
DEFAULT_HF=/mnt/storage-1/hf_cache
HF_CACHE="${HF_CACHE:-$([[ -d $DEFAULT_HF ]] && echo $DEFAULT_HF || echo "$HOME/.cache/huggingface")}"
# GPU 1 занята — по умолчанию берём только свободные.
GPUS="${GPUS:-\"device=0,2,3\"}"
NPROC="${NPROC:-2}"
# Тензорный параллелизм бенча ОТДЕЛЬНО от NPROC: vLLM требует, чтобы TP делил
# число голов внимания (32), поэтому TP=3 не стартует — 2 или 4, не 3.
BENCH_TP="${BENCH_TP:-2}"
RESULT_DIR="${RESULT_DIR:-$REPO/exps-$(date +%Y%m%d-%H%M%S)}"
BENCH_IMAGE="${BENCH_IMAGE:-design2code-bench:latest}"
BENCH_DATASET="${BENCH_DATASET:-SALT-NLP/Design2Code-hf}"
BENCH_N="${BENCH_N:-484}"
DRY_RUN="${DRY_RUN:-0}"

# Эффективный батч = bs * accum * NPROC. Держим его ~16 (на 1000 примерах при
# 64 выходит 46 шагов оптимизатора на 3 эпохи — для сходимости мало), поэтому
# accum считаем от числа карт, а не берём из конфига.
MICRO_BS=4
ACCUM=$(( 16 / (MICRO_BS * NPROC) )); (( ACCUM < 1 )) && ACCUM=1
EFF_BATCH=$(( MICRO_BS * ACCUM * NPROC ))

LOGS="$RESULT_DIR/logs"
REPORT="$RESULT_DIR/REPORT.txt"
mkdir -p "$LOGS"

# fd 3 — исходный stdout: шаги пишут свой вывод в лог-файлы, но команды в
# DRY_RUN нужно видеть на экране, а не вылавливать из логов.
exec 3>&1

say()  { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$REPORT"; }
rule() { printf '%s\n' "--------------------------------------------------------" | tee -a "$REPORT"; }
run()  { if [[ "$DRY_RUN" == "1" ]]; then echo "  DRY: $*" >&3; return 0; fi; "$@"; }

# ---------------------------------------------------------------- проверки --
# Лучше упасть здесь за секунду, чем через 11 часов обнаружить пустой ClearML.
[[ -d "$DATA_DIR/$DATASET_NAME" ]] || { say "НЕТ датасета: $DATA_DIR/$DATASET_NAME"; exit 1; }
docker image inspect sft >/dev/null 2>&1 || { say "НЕТ образа sft — собери: cd SFT && docker build -t sft ."; exit 1; }
if [[ "${SKIP_BENCH:-0}" != "1" ]]; then
  docker image inspect "$BENCH_IMAGE" >/dev/null 2>&1 || {
    say "НЕТ образа $BENCH_IMAGE — собери: cd Evaluation && ./build.sh"; exit 1; }
fi
if [[ -z "${CLEARML_API_ACCESS_KEY:-}" ]]; then
  say "ВНИМАНИЕ: CLEARML_API_ACCESS_KEY не задан — трекинга не будет."
  say "Прерываю. Если так и надо — CLEARML_DISABLE=1 ./run_pilot.sh"
  [[ "${CLEARML_DISABLE:-}" == "1" ]] || exit 1
fi

# ID | конфиг | S-ID плана | доп. аргументы обучения | пиксель-бюджет
#
# Волна 1 — решить LoRA vs full-FT. По ДВА LR на метод, потому что у них разные
# оптимумы (LoRA ~1e-4..3e-4, full ~1e-5..3e-5): сравнение «одна точка против
# одной» спутано с LR, и проигравшим может оказаться просто неудачно
# настроенный метод, а не метод как таковой. Сравнивать лучшее с лучшим.
WAVE1=(
  "E1|configs/lora_ft_qwen3_5_4b.yaml|S0|--learning_rate 1e-4|2097152"
  "E2|configs/lora_ft_qwen3_5_4b.yaml|S8|--learning_rate 3e-4|2097152"
  "E3|configs/full_ft_qwen3_5_4b.yaml|S5|--learning_rate 1e-5|2097152"
  "E4|configs/full_ft_qwen3_5_4b.yaml|S5|--learning_rate 3e-5|2097152"
)

# Волна 2 — оси, которые осмысленно крутить УЖЕ на победившем методе.
# Запуск: WAVE=2 ./run_pilot.sh (после того, как волна 1 выбрала метод;
# конфиг в строках при необходимости поменять на full_ft_*).
WAVE2=(
  "E5|configs/lora_ft_qwen3_5_4b.yaml|S2|--learning_rate 1e-4|3932160"
  "E6|configs/lora_ft_qwen3_5_4b.yaml|S10|--learning_rate 1e-4 --lora_r 64 --lora_alpha 128|2097152"
)

case "${WAVE:-1}" in
  1)   EXPERIMENTS=("${WAVE1[@]}") ;;
  2)   EXPERIMENTS=("${WAVE2[@]}") ;;
  all) EXPERIMENTS=("${WAVE1[@]}" "${WAVE2[@]}") ;;
  *)   echo "WAVE должно быть 1, 2 или all"; exit 1 ;;
esac

# Микробатч 4, а не 8: по замерам (SFT/THROUGHPUT.md) bs4 даёт 2655 ток/с
# против 1370 у bs8. accum считается выше от числа карт.
COMMON_ARGS="--num_train_epochs 3 --per_device_train_batch_size $MICRO_BS \
--gradient_accumulation_steps $ACCUM --report_to clearml"

say "каталог: $RESULT_DIR"
say "датасет: $DATA_DIR/$DATASET_NAME | HF-кэш: $HF_CACHE"
say "GPU: $GPUS | обучение на $NPROC карт | бенч TP=$BENCH_TP"
say "микробатч $MICRO_BS x accum $ACCUM x $NPROC карт = эфф. батч $EFF_BATCH"
[[ "$DRY_RUN" == "1" ]] && say "DRY_RUN — команды только печатаются"
rule

for row in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r EID CONFIG SID EXTRA PIXELS <<< "$row"
  if [[ -n "${ONLY:-}" && " $ONLY " != *" $EID "* ]]; then continue; fi

  OUT_HOST="$RESULT_DIR/$EID"
  say "=== $EID ($SID) — $(basename "$CONFIG") $EXTRA | ${PIXELS} px"

  # ------------------------------------------------------------- обучение --
  # OUT_DIR монтируется в /out, туда же трейнер кладёт clearml_task.json —
  # он и свяжет этот ран с будущим прогоном бенча.
  start=$SECONDS
  run env \
    DATA_DIR="$DATA_DIR" HF_CACHE="$HF_CACHE" OUT_DIR="$OUT_HOST" GPUS="$GPUS" \
    CLEARML_TAGS="$EID,$SID,pilot1k" SFT_MAX_PIXELS="$PIXELS" \
    "$REPO/SFT/run.sh" \
      torchrun --nproc_per_node="$NPROC" --tee 3 -m train.train_sft \
        --config "$CONFIG" \
        --dataset_name "/data/$DATASET_NAME" \
        $COMMON_ARGS $EXTRA \
        --output_dir /out \
    > "$LOGS/$EID.train.log" 2>&1
  rc=$?
  say "$EID обучение: rc=$rc, $(( (SECONDS-start)/60 )) мин"
  [[ $rc -ne 0 ]] && { say "$EID -> дальше не идём (лог: $LOGS/$EID.train.log)"; rule; continue; }

  # ------------------------------------------- LoRA: слить адаптер в веса --
  # vLLM не умеет адаптеры, поэтому бенчить можно только слитую модель.
  MODEL_HOST="$OUT_HOST"
  if [[ -f "$OUT_HOST/adapter_config.json" || "$DRY_RUN" == "1" ]]; then
    MODEL_HOST="$RESULT_DIR/$EID-merged"
    start=$SECONDS
    run env \
      DATA_DIR="$DATA_DIR" HF_CACHE="$HF_CACHE" OUT_DIR="$RESULT_DIR" GPUS="$GPUS" \
      "$REPO/SFT/run.sh" \
        python -m scripts.merge_lora "/out/$EID" "/out/$EID-merged" \
      > "$LOGS/$EID.merge.log" 2>&1
    rc=$?
    say "$EID merge_lora: rc=$rc, $(( (SECONDS-start)/60 )) мин"
    [[ $rc -ne 0 ]] && { say "$EID -> бенч пропущен (лог: $LOGS/$EID.merge.log)"; rule; continue; }
  fi

  [[ "${SKIP_BENCH:-0}" == "1" ]] && { say "$EID бенч пропущен (SKIP_BENCH=1)"; rule; continue; }

  # ----------------------------------------------------- бенч Design2Code --
  # --max-pixels ОБЯЗАН совпадать с обучением, иначе чекпоинт меряется вне
  # своего трейн-распределения (H1 из плана экспериментов).
  # HOST_MODEL_DIR монтирует веса в бенч-контейнер по тому же пути.
  start=$SECONDS
  run env \
    IMAGE_TAG="$BENCH_IMAGE" \
    HOST_OUTDIR="$RESULT_DIR/$EID-bench" \
    HOST_HF_CACHE="$HF_CACHE" \
    HOST_MODEL_DIR="$MODEL_HOST" \
    CONTAINER_NAME="bench-$EID" GPUS="$GPUS" \
    CLEARML_TAGS="$EID,$SID,pilot1k" \
    "$REPO/Evaluation/run.sh" \
      --model "$MODEL_HOST" \
      --hf-dataset "$BENCH_DATASET" --hf-config default --hf-split train \
      --n-samples "$BENCH_N" --batch-size "$BENCH_N" \
      --max-pixels "$PIXELS" \
      --tensor-parallel-size "$BENCH_TP" \
    > "$LOGS/$EID.bench.log" 2>&1
  say "$EID бенч: rc=$?, $(( (SECONDS-start)/60 )) мин"

  if [[ -f "$RESULT_DIR/$EID-bench/summary.json" ]]; then
    say "$EID итог: $(python3 -c "
import json;d=json.load(open('$RESULT_DIR/$EID-bench/summary.json'))
print(' '.join(f'{k}={d[k]:.4f}' for k in ('final_score','block_match','text','position','color','clip') if k in d))
print('  обрезано по токенам:', d.get('n_length_truncated'))" 2>/dev/null)"
  fi
  rule
done

say "очередь закончена"
say "сводки: $RESULT_DIR/*-bench/summary.json | в ClearML фильтр по тегу pilot1k"
if [[ "$DRY_RUN" != "1" ]]; then
  echo
  echo "СВОДНАЯ ТАБЛИЦА:" | tee -a "$REPORT"
  python3 "$REPO/SFT/scripts/collect_results.py" "$RESULT_DIR" 2>/dev/null | tee -a "$REPORT"
fi
