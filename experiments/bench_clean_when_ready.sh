#!/usr/bin/env bash
# Отбенчить чистую ветку A/B, как только её обучение закончится.
#
# Зачем отдельный проход: основная очередь прошла фазу бенча чистой ветки
# тогда, когда весов ещё не было (первый заход упал по OOM из-за соседа, и
# обучение перезапускалось следом). Повторно её звать нельзя, пока обучение
# идёт: `train_variant` не увидит готовых весов и начнёт учить заново.
#
# Признак завершённого прогона — config.json в КОРНЕ каталога рана, а не в
# checkpoint-*/: трейнер пишет его финальным сохранением. Промежуточные
# чекпоинты появляются задолго до конца и признаком служить не могут.
#
#   tmux new -s cleanbench
#   cd ~/ScreenShot2Code && source .env && ./experiments/bench_clean_when_ready.sh
set -uo pipefail

cd "$(dirname "$0")/.."
REPO="$PWD"
RUN_ROOT="${RUN_ROOT:-/mnt/storage-1/Screenshot2Code/checkpoints_exps/wc2m-15k-ab}"

waited=0
until compgen -G "$RUN_ROOT/clean/*/config.json" > /dev/null 2>&1; do
  (( waited % 10 == 0 )) && echo "[$(date '+%m-%d %H:%M:%S')] жду конца обучения чистой ветки, прошло $waited мин"
  sleep 60
  waited=$((waited+1))
  if (( waited > 480 )); then
    echo "[$(date '+%m-%d %H:%M:%S')] обучение не закончилось за 8 ч — выхожу, разбираться руками"
    exit 1
  fi
done

echo "[$(date '+%m-%d %H:%M:%S')] веса чистой ветки готовы — запускаю бенч всех её чекпоинтов"
# ONLY=clean: обучение будет пропущено (веса на месте), пойдёт сразу бенч.
exec env ONLY=clean "$REPO/experiments/run_wc2m_ab.sh"
