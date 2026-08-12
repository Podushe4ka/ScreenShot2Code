#!/usr/bin/env bash
# Перебенч свипа по эпохам в GREEDY-режиме (temperature 0).
#
# Зачем: старые числа свипа сняты с дефолтами vLLM — чистый сэмплинг при
# temperature 1.0, top_p 1.0, без seed (SamplingParams вызывался вообще без
# параметров). На странице в ~8к токенов шанс сорваться копится по всем
# токенам, поэтому выход получался «всё или ничего»: либо почти эталон
# (0.85-0.94), либо обрыв посреди <style> с пустым рендером и score ровно 0.
# Здесь меряем те же чекпоинты greedy и базу тем же режимом для сравнения.
#
# Веса не переобучаем — берём готовые чекпоинты прошлого свипа.
# Запуск: GPUS='"device=1"' TP=1 ./sweep_greedy.sh
set -uo pipefail
cd "$(dirname "$0")/.."; REPO="$PWD"   # скрипт лежит в experiments/, работаем от корня репо
REPO="$PWD"
source "$REPO/experiments/lib/common.sh"
BASE="$STORAGE/Screenshot2Code"
mkdir -p "$BASE/logs/greedy"
OUT="$BASE/checkpoints_exps/d2c-sweep"
RUN="$OUT/full_ft_qwen3_5_4b_s42_20260805-201901"
TP="${TP:-1}"; GPUS="${GPUS:-\"device=1\"}"
BENCH_DS_E=/root/.cache/huggingface/d2c_short_bench      # путь ВНУТРИ образа бенча
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-4B}"
LOG="$BASE/logs/greedy/GREEDY.log"; RUN_LOG="$LOG"; SAY_TIME_FMT='%H:%M:%S'

NREAL=$(count_samples /storage/Screenshot2Code/hf_cache/d2c_short_bench)
[[ -n "$NREAL" ]] || { say "не смог прочитать датасет"; exit 1; }
say "greedy-перебенч: $NREAL сэмплов, TP=$TP, GPUS=$GPUS"

# $1 — имя прогона, $2 — модель (путь к чекпоинту или id на HF)
# ⚠ bash раскрывает ВСЕ аргументы `local` до присваивания, поэтому
# `local a="$1" b="$OUT/$a"` под set -u падает на unbound variable — объявляем раздельно.
bench(){ local name="$1" model="$2"; local bdir="$OUT/bench-greedy-$name"
  [[ -f "$bdir/summary.json" ]] && { say "$name: уже есть"; return; }
  mkchmod "$bdir"
  say "$name: бенч $model ..."
  # HOST_MODEL_DIR нужен только для локальных весов; база тянется из HF-кэша.
  local mnt=""; [[ -d "$model" ]] && mnt="$model"
  env IMAGE_TAG=design2code-bench:latest HOST_OUTDIR="$bdir" HOST_HF_CACHE="$BASE/hf_cache" \
      ${mnt:+HOST_MODEL_DIR="$mnt"} CONTAINER_NAME="greedy-$name" GPUS="$GPUS" CLEARML_DISABLE=1 \
      "$REPO/Evaluation/run.sh" --no-resume --model "$model" \
        --hf-dataset "$BENCH_DS_E" --hf-config default --hf-split train \
        --n-samples "$NREAL" --batch-size "$NREAL" --n-examples-per-batch "$NREAL" \
        --temperature 0 --max-pixels 2097152 --tensor-parallel-size "$TP" \
        --gpu-memory-utilization 0.9 --max-new-tokens 16384 --max-model-len 24384 --num-workers 8 \
      > "$BASE/greedy_${name}.log" 2>&1
  docker rm "greedy-$name" >/dev/null 2>&1
  local sc; sc=$(python3 -c "import json;print(round(json.load(open('$bdir/summary.json'))['final_score'],3))" 2>/dev/null)
  say "$name: final_score = ${sc:-НЕТ}"
}

bench base "$BASE_MODEL"
for STEP in 5 10 15 20 25; do
  EP=$(( STEP / 5 ))
  [[ -f "$RUN/checkpoint-$STEP/config.json" ]] || { say "ep$EP: весов нет"; continue; }
  bench "ep$EP" "$RUN/checkpoint-$STEP"
done

say "=== СВОДКА GREEDY (в скобках — старое, сэмплинг T=1.0) ==="
python3 - "$OUT" <<'PY' | tee -a "$LOG"
import json, os, sys
out = sys.argv[1]
old = {"ep1": 0.059, "ep2": 0.297, "ep3": 0.138, "ep4": 0.527, "ep5": 0.539, "base": 0.835}
for name in ("base", "ep1", "ep2", "ep3", "ep4", "ep5"):
    f = os.path.join(out, "bench-greedy-%s" % name, "summary.json")
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    print("  %-5s final %.3f (было %.3f)  block %.3f  text %.3f  pos %.3f  color %.3f  clip %.3f" % (
        name, d["final_score"], old.get(name, 0), d["block_match"], d["text"],
        d["position"], d["color"], d["clip"]))
PY
