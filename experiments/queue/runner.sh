#!/usr/bin/env bash
# runner.sh — планировщик прогонов: сам находит свободные карты и берёт работу.
#
# По одному экземпляру на КАЖДОЙ машине. Координация — через общий диск, а не
# по ssh между хостами: агент ключей форвардится только пока жив мой ssh-сеанс,
# и оркестратор, ходящий с машины на машину, умрёт вместе с закрытым ноутбуком.
# Общий каталог на /mnt/storage-1 виден обеим машинам всегда.
#
#   очередь/jobs/<приоритет>-<имя>.job   задания (обычные bash-скрипты)
#   очередь/claims/<имя>/                кто взял (атомарный mkdir)
#   очередь/done/<имя>.log               выполненные
#
# Захват через `mkdir`: он атомарен даже на сетевой ФС, в отличие от
# «проверить файл, потом создать» —两 раннера иначе возьмут одно задание.
#
# Задание объявляет свои требования строками-заголовками:
#   # NEED_GPUS=2      сколько карт нужно
#   # FREE_MB=5000     какую карту считать свободной (обучению нужна пустая,
#                      бенчу хватает половины)
# Раннер экспортирует в задание GPU_IDS (через запятую) и GPU_COUNT.
#
#   HOST_TAG=a100-3 ./experiments/queue/runner.sh
set -uo pipefail

cd "$(dirname "$0")/../.."
REPO="$PWD"

Q="${Q:-/mnt/storage-1/Screenshot2Code/queue}"
JOBS="$Q/jobs"; CLAIMS="$Q/claims"; DONE="$Q/done"; LOGS="$Q/logs"
mkdir -p "$JOBS" "$CLAIMS" "$DONE" "$LOGS"
HOST_TAG="${HOST_TAG:-$(hostname)}"
POLL="${POLL:-60}"
MAX_IDLE_MIN="${MAX_IDLE_MIN:-720}"

say() { echo "[$(date '+%m-%d %H:%M:%S')] [$HOST_TAG] $*" | tee -a "$LOGS/runner-$HOST_TAG.log"; }

# ⚠ Пустая память НЕ значит свободную карту. RL-трек (verl/GRPO) идёт с
# param_offload + optimizer_offload + free_cache_engine: между фазами он
# выгружает параметры и оптимизатор в RAM и освобождает KV-кэш vLLM, поэтому
# nvidia-smi показывает почти ноль при живом прогоне. Занять карту в этот
# промежуток — значит убить чужое обучение, когда оно попросит память назад:
# умирает не заехавший, а тот, кто там был. Двенадцать часов чужого прогона
# так уже потеряны.
#
# Поэтому признак занятости — НАЛИЧИЕ ЛЮБОГО процесса на карте, а не объём.
# Контекст CUDA живёт всё время работы задачи и виден в compute-apps даже
# после выгрузки весов.
gpu_busy_by_process() {   # печатает индексы карт, где есть хоть один процесс
  local uuid pid idx
  declare -A byuuid=()
  while IFS=', ' read -r idx uuid; do
    [[ -n "${uuid:-}" ]] && byuuid["$uuid"]="$idx"
  done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null)
  while IFS=', ' read -r uuid pid; do
    [[ -n "${uuid:-}" && -n "${byuuid[$uuid]:-}" ]] && echo "${byuuid[$uuid]}"
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null)
}

free_gpus_once() {
  local limit="$1" ids=() id used busy
  busy=" $(gpu_busy_by_process | sort -u | tr '\n' ' ') "
  while IFS=', ' read -r id used; do
    [[ -z "${used:-}" ]] && continue
    [[ "$busy" == *" $id "* ]] && continue          # на карте живёт чужой процесс
    (( used < limit )) && ids+=("$id")
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)
  (IFS=,; echo "${ids[*]}")
}

# Карта должна выглядеть свободной НЕСКОЛЬКО раз подряд: одиночный замер
# попадает ровно в паузу между фазами чужого прогона.
CONFIRM_SAMPLES="${CONFIRM_SAMPLES:-3}"
CONFIRM_GAP="${CONFIRM_GAP:-60}"

free_gpus() {
  local limit="$1" acc first i cur
  first="$(free_gpus_once "$limit")"
  [[ -z "$first" ]] && { echo ""; return; }
  acc="$first"
  for (( i = 1; i < CONFIRM_SAMPLES; i++ )); do
    sleep "$CONFIRM_GAP"
    cur="$(free_gpus_once "$limit")"
    # пересечение: карта остаётся кандидатом, только если свободна во ВСЕХ пробах
    local keep=()
    for id in ${acc//,/ }; do
      [[ ",$cur," == *",$id,"* ]] && keep+=("$id")
    done
    acc="$(IFS=,; echo "${keep[*]}")"
    [[ -z "$acc" ]] && break
  done
  echo "$acc"
}

say "старт. очередь: $Q | опрос раз в ${POLL}с"
idle=0

while true; do
  took=0
  for job in $(ls "$JOBS"/*.job 2>/dev/null | sort); do
    name="$(basename "$job" .job)"
    [[ -d "$CLAIMS/$name" ]] && continue
    [[ -f "$DONE/$name.log" ]] && continue

    need=$(grep -m1 '^# NEED_GPUS=' "$job" | cut -d= -f2); need="${need:-1}"
    freemb=$(grep -m1 '^# FREE_MB=' "$job" | cut -d= -f2); freemb="${freemb:-5000}"

    ids="$(free_gpus "$freemb")"
    have=$([[ -z "$ids" ]] && echo 0 || awk -F, '{print NF}' <<< "$ids")
    (( have < need )) && continue

    # Атомарный захват. Если mkdir не удался — задание уже взял кто-то другой
    # (второй раннер на этой же или на соседней машине), просто идём дальше.
    mkdir "$CLAIMS/$name" 2>/dev/null || continue
    ids="$(cut -d, -f1-"$need" <<< "$ids")"
    echo "$HOST_TAG $ids $(date -Is)" > "$CLAIMS/$name/owner"

    say "беру $name: нужно $need карт, даю $ids"
    took=1; idle=0
    start=$SECONDS
    GPU_IDS="$ids" GPU_COUNT="$need" REPO="$REPO" HOST_TAG="$HOST_TAG" \
      bash "$job" > "$LOGS/$name.log" 2>&1
    rc=$?
    say "$name: rc=$rc, $(( (SECONDS-start)/60 )) мин"
    # Успех фиксируем в done (задание больше не берётся). Провал оставляем
    # незафиксированным, но claim не снимаем: перезапускать упавшее автоматом
    # опасно — если оно падает мгновенно, очередь уйдёт в цикл. Разбор руками.
    if (( rc == 0 )); then
      cp "$LOGS/$name.log" "$DONE/$name.log"
    else
      say "$name: ПРОВАЛ — оставляю в claims, автоперезапуска не делаю (лог: $LOGS/$name.log)"
    fi
    break            # после каждой работы заново оцениваем карты
  done

  if (( took == 0 )); then
    (( idle % 10 == 0 )) && say "работы нет или карт не хватает, жду (простой $idle мин)"
    sleep "$POLL"; idle=$((idle+1))
    (( idle > MAX_IDLE_MIN )) && { say "простой больше $MAX_IDLE_MIN мин — выхожу"; exit 0; }
  fi
done
