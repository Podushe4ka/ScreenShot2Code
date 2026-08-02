#!/usr/bin/env bash
# Всё подряд, без присмотра: факты о модели -> проверка стабильности ->
# матрица замеров -> сводка -> профиль. Результаты в один каталог.
#
# Запуск из /workspace/SFT внутри контейнера:
#   nohup bash scripts/run_all.sh > run_all.out 2>&1 &
#   tail -f run_all.out
#
# Ни один шаг не роняет прогон целиком: каждый отчитывается и передаёт дальше.
# Итог — $RESULT_DIR/REPORT.txt.
set -uo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${RESULT_DIR:-}" ]]; then
  # /out — смонтированный с хоста просторный диск (OUT_DIR в run.sh).
  # Без него пишем рядом с репозиторием, но там легко упереться в место.
  BASE=$([[ -d /out ]] && echo /out || echo .)
  RESULT_DIR="$BASE/experiments-$(date +%Y%m%d-%H%M%S)"
fi
DATASET="${DATASET:-/data/webcode2m_1000_split}"
CONFIG="${CONFIG:-configs/full_ft_qwen3_5_4b.yaml}"
NPROC="${NPROC:-2}"

LOGS="$RESULT_DIR/logs"
REPORT="$RESULT_DIR/REPORT.txt"
mkdir -p "$LOGS"

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$REPORT"; }
rule() { printf '%s\n' "----------------------------------------------------------" | tee -a "$REPORT"; }

matrix() {
  # запускает матрицу с нужным окружением; $1 — файл журнала прогона
  local journal="$1"; shift
  env OUT="$LOGS" DATASET="$DATASET" CONFIG="$CONFIG" NPROC="$NPROC" \
      EXTRA_ARGS="$EXTRA_ARGS" "$@" \
      bash scripts/throughput_matrix.sh >> "$journal" 2>&1
}

measured() { grep -qc "train_runtime" "$1" 2>/dev/null; }

say "каталог результатов: $RESULT_DIR"
say "датасет: $DATASET | конфиг: $CONFIG | GPU: $NPROC"
rule

# --- 0. конфиги из gen.py ------------------------------------------------------
say "шаг 0: перегенерация конфигов"
if /opt/venv/bin/python -m configs.gen > "$RESULT_DIR/configs_gen.txt" 2>&1; then
  say "  ок"
else
  say "  ОШИБКА, см. configs_gen.txt"
fi

# --- 1. факты об архитектуре ---------------------------------------------------
say "шаг 1: факты о модели (нужны для честного MFU)"
/opt/venv/bin/python -m scripts.model_facts --config "$CONFIG" --load-weights \
  > "$RESULT_DIR/model_facts.txt" 2>&1
PARAMS=$(grep -o '"params_billions": *[0-9.]*' "$RESULT_DIR/model_facts.txt" \
         | head -1 | grep -o '[0-9.]*$')
if [[ -z "$PARAMS" ]]; then
  PARAMS=4.0
  say "  размер модели не определился, беру $PARAMS млрд (см. model_facts.txt)"
else
  say "  параметров: $PARAMS млрд"
fi
rule

# --- 2. стабильность: один прогон, при падении — обход без воркеров ------------
say "шаг 2: проверка стабильности (эксперимент base)"
EXTRA_ARGS=""
if measured "$LOGS/base.log"; then
  say "  base уже замерен в этом каталоге, проверку пропускаю"
else
  matrix "$RESULT_DIR/stability.txt" ONLY=base
fi

if measured "$LOGS/base.log"; then
  say "  базовый прогон прошёл"
else
  say "  базовый прогон упал — пробую без воркеров даталоадера"
  rm -f "$LOGS/base.log"
  EXTRA_ARGS="--dataloader_num_workers 0"
  matrix "$RESULT_DIR/stability.txt" ONLY=base
  if measured "$LOGS/base.log"; then
    say "  прошло без воркеров => виноват даталоадер, матрица пойдёт с этим обходом"
  else
    say "  падает и без воркеров => дело в нативном коде модели, не в даталоадере"
    say "  матрицу всё равно запускаю: часть конфигураций может оказаться живой"
    EXTRA_ARGS=""
  fi
fi
rule

# --- 3. матрица ----------------------------------------------------------------
say "шаг 3: матрица замеров (~1.5 часа)"
matrix "$RESULT_DIR/matrix.txt" SKIP_EXISTING=1
say "  матрица завершена"
rule

# --- 4. сводка -----------------------------------------------------------------
say "шаг 4: сводка"
/opt/venv/bin/python -m scripts.throughput_report "$LOGS" \
  --params-billions "$PARAMS" --gpus "$NPROC" > "$RESULT_DIR/summary.txt" 2>&1
cat "$RESULT_DIR/summary.txt" | tee -a "$REPORT"
rule

# --- 5. профиль ----------------------------------------------------------------
say "шаг 5: профиль ядер"
if /opt/venv/bin/torchrun --nproc_per_node="$NPROC" --tee 3 \
     -m scripts.profile_steps \
     --config "$CONFIG" --dataset_name "$DATASET" $EXTRA_ARGS \
     > "$RESULT_DIR/profile.txt" 2>&1; then
  say "  профиль собран"
else
  say "  профиль не собрался, см. profile.txt"
fi
if grep -q "self CUDA time" "$RESULT_DIR/profile.txt" 2>/dev/null; then
  say "  топ ядер (CUDA и CPU) — полностью в profile.txt"
  sed -n '/self CUDA time/,$p' "$RESULT_DIR/profile.txt" | head -18 >> "$REPORT"
  sed -n '/self CPU time/,$p' "$RESULT_DIR/profile.txt" | head -18 >> "$REPORT"
fi
rule

say "ГОТОВО"
say "смотреть: $REPORT | $RESULT_DIR/summary.txt | $RESULT_DIR/profile.txt | $LOGS/"
