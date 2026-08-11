# Запуск из /workspace/SFT внутри контейнера:
#   bash scripts/throughput_matrix.sh
#   ONLY=nogc_8192_bs2 bash scripts/throughput_matrix.sh
#   STEPS=7 bash scripts/throughput_matrix.sh
set -uo pipefail

cd "$(dirname "$0")/.."

DATASET="${DATASET:-/data/webcode2m_1000_split}"
CONFIG="${CONFIG:-configs/full_ft_qwen3_5_4b.yaml}"
STEPS="${STEPS:-5}"
NPROC="${NPROC:-2}"
OUT="${OUT:-throughput-logs}"
ONLY="${ONLY:-}"
# добавляется в КАЖДЫЙ прогон — нужно, когда обход вроде
# --dataloader_num_workers 0 надо применить ко всей матрице
EXTRA_ARGS="${EXTRA_ARGS:-}"
# 1 — не перезапускать эксперименты, где лог уже содержит замеры
SKIP_EXISTING="${SKIP_EXISTING:-0}"

mkdir -p "$OUT"

# Упавший ранг оставляет живой CUDA-контекст, и следующий эксперимент падает по
# OOM не по своей вине. Добиваем своё и ждём, пока карты реально освободятся.
# pkill по имени модуля безопасен: у контейнера свой PID-namespace, чужие
# процессы на GPU 2/3 отсюда не видны.
free_gpus() {
  pkill -9 -f "scripts.throughput_run" 2>/dev/null
  pkill -9 -f "torch.distributed.run" 2>/dev/null
  local used limit=$((2000 * NPROC))
  for _ in $(seq 1 45); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
           | head -n "$NPROC" | awk '{s+=$1} END {print s+0}')
    if [[ "$used" -lt "$limit" ]]; then
      return 0
    fi
    sleep 2
  done
  echo "[matrix] ВНИМАНИЕ: карты не освободились за 90с (занято ${used} MiB, порог ${limit})"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv | sed 's/^/    /'
}

run_one() {
  local name="$1"; shift
  if [[ -n "$ONLY" && "$ONLY" != "$name" ]]; then
    return 0
  fi
  if [[ "$SKIP_EXISTING" == "1" && -s "$OUT/$name.log" ]] \
     && [[ $(grep -c "train_runtime" "$OUT/$name.log") -ge 3 ]]; then
    echo "[matrix] $name: уже замерен, пропускаю"
    return 0
  fi
  echo "=============================================================="
  echo "[matrix] $name  |  ${RUN_ENV:+[$RUN_ENV] }$*"
  local t0=$SECONDS
  env ${RUN_ENV:-} /opt/venv/bin/torchrun --nproc_per_node="$NPROC" --tee 3 \
    -m scripts.throughput_run \
    --config "$CONFIG" \
    --dataset_name "$DATASET" \
    --max_steps "$STEPS" \
    --logging_steps 1 \
    --logging_first_step true \
    --include_num_input_tokens_seen true \
    --save_strategy no \
    --eval_strategy no \
    --report_to none \
    --output_dir "$OUT/_scratch" \
    "$@" $EXTRA_ARGS > "$OUT/$name.log" 2>&1
  local rc=$?
  local dt=$((SECONDS - t0))
  if [[ $rc -eq 0 ]]; then
    echo "[matrix] $name: готово за ${dt}с"
  else
    echo "[matrix] $name: УПАЛ (код $rc, ${dt}с) — $OUT/$name.log"
    grep -m1 -E "OutOfMemoryError|SIGSEGV|Error" "$OUT/$name.log" | sed 's/^/    /'
  fi
  free_gpus
}

# ---- прогрев: первый эксперимент оплачивает компиляцию ядер fla за всех ------
# (кэши Triton разнесены по рангам, так что компилируют оба). Имя с подчёркивания
# — throughput_report такие не показывает.
run_one _warmup --per_device_train_batch_size 4 --gradient_accumulation_steps 1

# ---- база и реализация внимания ---------------------------------------------
run_one base        --per_device_train_batch_size 4 --gradient_accumulation_steps 2
run_one attn_sdpa   --per_device_train_batch_size 4 --gradient_accumulation_steps 2 --attn_implementation sdpa
run_one attn_eager  --per_device_train_batch_size 4 --gradient_accumulation_steps 2 --attn_implementation eager

# ---- размер микробатча при неизменных 16 сэмплах на шаг ----------------------
run_one bs2  --per_device_train_batch_size 2 --gradient_accumulation_steps 4
run_one bs8  --per_device_train_batch_size 8 --gradient_accumulation_steps 1

# ---- gradient checkpointing --------------------------------------------------
# len8192 — контроль: сколько даёт само укорачивание, без выключения чекпоинтинга
run_one len8192        --max_length 8192 --per_device_train_batch_size 4 --gradient_accumulation_steps 2
run_one nogc_8192_bs1  --max_length 8192 --per_device_train_batch_size 1 --gradient_accumulation_steps 8  --gradient_checkpointing false
run_one nogc_8192_bs2  --max_length 8192 --per_device_train_batch_size 2 --gradient_accumulation_steps 4  --gradient_checkpointing false
run_one nogc_16384_bs1 --max_length 16384 --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --gradient_checkpointing false

# ---- стадия ZeRO -------------------------------------------------------------
run_one zero1 --per_device_train_batch_size 4 --gradient_accumulation_steps 2 --deepspeed configs/deepspeed_zero1.json
run_one zero3 --per_device_train_batch_size 4 --gradient_accumulation_steps 2 --deepspeed configs/deepspeed_zero3.json

# ---- компиляция --------------------------------------------------------------
run_one compile --per_device_train_batch_size 4 --gradient_accumulation_steps 2 --torch_compile true

# ---- сколько стоит коммуникация: тут размер шага МЕНЯЕТСЯ намеренно ----------
# accum1 — 8 сэмплов на шаг (коммуникация вдвое чаще), accum8 — 64 (боевой конфиг,
# шаг вчетверо длиннее, поэтому меньше шагов).
run_one accum1 --per_device_train_batch_size 4 --gradient_accumulation_steps 1
STEPS=4 run_one accum8 --per_device_train_batch_size 4 --gradient_accumulation_steps 8

# ---- загрузка CUDA-модулей: профиль показал 230 с на Lazy Function Loading ----
# EAGER грузит ядра заранее. Сравнивать строго с accum8 — та же конфигурация.
STEPS=4 RUN_ENV="CUDA_MODULE_LOADING=EAGER" run_one modules_eager \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 8

# ---- батчинг по бюджету токенов ----------------------------------------------
# Бюджет 65536 = нынешний потолок bs4 x 16384, accum 5 -> ~655k токенов на шаг
# против ~750k у accum8. Сравнивать с accum8.
STEPS=4 run_one tokenbatch \
  --max_tokens_per_batch 65536 --gradient_accumulation_steps 5
STEPS=4 run_one tokenbatch_nobucket \
  --max_tokens_per_batch 65536 --gradient_accumulation_steps 5 --length_bucket 0

# ---- буферы DeepSpeed --------------------------------------------------------
# Снапшот памяти показал всплеск 12.7 ГиБ внутри step(): all-gather собирает
# полный набор весов в плоский буфер (8.46 ГиБ = 4.539 млрд x 2 Б) плюс шард
# (4.23 ГиБ). Меньшие бакеты режут это на куски ценой лишних раундов обмена.
run_one smallbucket --per_device_train_batch_size 4 --gradient_accumulation_steps 2 \
  --deepspeed configs/deepspeed_zero2_smallbucket.json
STEPS=4 run_one smallbucket_compile \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
  --deepspeed configs/deepspeed_zero2_smallbucket.json --torch_compile true

# ---- torch.compile: единственный рычаг против дробления на мелкие ядра -------
# 133 тыс. запусков ядер на шаг, очередь команд забита, ~5 синхронизаций на вызов
# fla. compile сливает цепочки мелких операций в одно ядро.
# Прошлый OOM был при bs=4 — пробуем с меньшим микробатчем.
run_one compile_bs2 --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
  --torch_compile true

# reduce-overhead включает CUDA-графы: они убирают именно накладные расходы на
# запуск ядер. Но графы требуют стабильных форм тензоров.
run_one compile_graphs --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
  --torch_compile true --torch_compile_mode reduce-overhead

# Формы стабилизируем округлением длин до корзины — то, ради чего писался
# length_bucket. По скорости он ничего не дал, но графам он нужен.
STEPS=4 run_one compile_graphs_bucket \
  --max_tokens_per_batch 32768 --gradient_accumulation_steps 10 --length_bucket 2048 \
  --torch_compile true --torch_compile_mode reduce-overhead

echo
echo "[matrix] готово. Сводка:"
echo "    /opt/venv/bin/python -m scripts.throughput_report $OUT"
