#!/usr/bin/env bash
# Прогон UI2Code^N (учителя) нашим харнессом на том же наборе, что и все свипы.
#
# UI2Code^N — GLM-4.1V-9B, дообученная под UI-to-code (SOTA на Design2Code).
# Нужна как верхняя точка отсчёта: столько выжимает специализированная модель
# на НАШИХ метриках, а не на числах из статьи.
#
# Два отличия от Qwen, из-за которых нельзя просто подставить --model:
#  1) промпт. Модель обучена на своей формулировке; мерить её нашим длинным
#     промптом — мерить рассогласование, а не качество. Берём родной через
#     --prompt-file. Для сравнимости гоняем ОБА варианта.
#  2) пиксель-бюджет. min_pixels/max_pixels — параметры Qwen2VL-процессора;
#     у Glm4vImageProcessor бюджет задаётся через size, и передача чужих
#     kwargs роняет загрузку. Передаём 0 = не передавать вовсе.
#
# Запуск: GPUS='"device=1"' ./bench_ui2code.sh
set -uo pipefail
cd "$(dirname "$0")/.."; REPO="$PWD"
REPO="$PWD"
source "$REPO/experiments/lib/common.sh"
BASE="$STORAGE/Screenshot2Code"
mkdir -p "$BASE/logs/ui2code" "$BASE/prompts"
MODEL="${MODEL:-zai-org/UI2Code_N}"
BENCH_DS_E="${BENCH_DS_E:-/root/.cache/huggingface/d2c_short_bench}"
OUT="${OUT:-$BASE/checkpoints_exps/ui2code}"
# N задаём явно для датасетов с хаба (там load_from_disk не сработает).
# Пусто = локальный набор, размер считаем сами.
N="${N:-}"
# Картинки полного Design2Code бывают очень высокими, а GLM берёт пиксель-бюджет
# из своего конфига (longest_edge ~9.6 Мп ≈ 12к визуальных токенов). Вместе с
# 16384 новыми токенами это не влезает в 24384 — держим запас.
MAXLEN="${MAXLEN:-40960}"
MAXNEW="${MAXNEW:-16384}"
TP="${TP:-1}"; GPUS="${GPUS:-\"device=1\"}"
# 9B + CLIP на одной карте. Коридор узкий: при 0.9 не поднимается clip_server
# («не поднялся за 120с»), при 0.6 не хватает уже самому vLLM на KV-кэш
# («No available memory for the cache blocks»). 0.7 держит обоих.
GPU_MEM="${GPU_MEM:-0.70}"
# Креды ClearML: run.sh пробрасывает CLEARML_* из окружения, tracking.py без них
# тихо no-op'ит. Раньше здесь стоял CLEARML_DISABLE=1 — скопирован из отладочных
# sanity-скриптов, для полноценного эксперимента это неверно.
[[ -f "$REPO/.env" ]] && set -a && . "$REPO/.env" && set +a
LOG="$BASE/logs/ui2code/UI2CODE.log"; RUN_LOG="$LOG"; SAY_TIME_FMT='%H:%M:%S'

# Родной промпт модели (из её карточки на HF).
NATIVE="$BASE/prompts/ui2code_native.txt"
[[ -f "$NATIVE" ]] || printf 'Please generate the corresponding html code for the given UI screenshot.' > "$NATIVE"

if [[ -n "$N" ]]; then
  NREAL="$N"
else
  NREAL=$(count_samples "${BENCH_DS_E/\/root\/.cache\/huggingface//storage/Screenshot2Code/hf_cache}")
fi
[[ -n "$NREAL" ]] || { say "датасет не читается: $BENCH_DS_E"; exit 1; }
say "UI2Code^N: $BENCH_DS_E, $NREAL сэмплов, greedy, TP=$TP, max_model_len=$MAXLEN"

# $1 — метка прогона, $2 — файл промпта или пусто (пусто = наш встроенный PROMPT)
run(){ local name="$1" pfile="${2:-}"; local bdir="$OUT/bench-$name"
  [[ -f "$bdir/summary.json" ]] && { say "$name: уже есть"; return; }
  mkchmod "$bdir"
  say "$name: промпт = ${pfile:-встроенный}"
  env IMAGE_TAG=design2code-bench:latest HOST_OUTDIR="$bdir" HOST_HF_CACHE="$BASE/hf_cache" \
      CONTAINER_NAME="ui2code-$name" GPUS="$GPUS" \
      "$REPO/Evaluation/run.sh" --no-resume --model "$MODEL" \
        --hf-dataset "$BENCH_DS_E" --hf-config default --hf-split train \
        --n-samples "$NREAL" --batch-size "$NREAL" --n-examples-per-batch "$NREAL" \
        --temperature 0 --min-pixels 0 --max-pixels 0 --materialize-dom \
        ${pfile:+--prompt-file "$pfile"} \
        --tensor-parallel-size "$TP" --gpu-memory-utilization "$GPU_MEM" \
        --max-new-tokens "$MAXNEW" --max-model-len "$MAXLEN" --num-workers 8 \
      > "$BASE/logs/ui2code/ui2code_$name.log" 2>&1
  docker rm "ui2code-$name" >/dev/null 2>&1
  sleep 60   # ждём, пока предыдущий процесс отпустит VRAM: иначе следующий
             # запуск видит меньше свободной памяти, чем есть на самом деле
  local sc; sc=$(python3 -c "import json;print(round(json.load(open('$bdir/summary.json'))['final_score'],3))" 2>/dev/null)
  say "$name: final_score = ${sc:-НЕТ}"
}

run native "$NATIVE"                       # честная оценка на её же формулировке
[[ "${ONLY_NATIVE:-0}" = "1" ]] || run ours ""   # наш промпт — для прямого сравнения с Qwen

say "=== СВОДКА UI2CODE (greedy; база Qwen3.5-4B на этом наборе 0.758) ==="
python3 - "$OUT" <<'PY' | tee -a "$LOG"
import json, os, sys, glob, csv
out = sys.argv[1]
for d in sorted(glob.glob(os.path.join(out, "bench-*"))):
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
    print("  %-8s final %.3f  block %.3f  text %.3f  pos %.3f  color %.3f  clip %.3f  нулей %s" % (
        os.path.basename(d).replace("bench-", ""), s["final_score"], s["block_match"],
        s["text"], s["position"], s["color"], s["clip"], zeros))
PY
