# common.sh — помощники, общие для оркестраторов experiments/.
#
# Подключение (после того как скрипт перешёл в корень репозитория):
#     cd "$(dirname "$0")/.."; REPO="$PWD"
#     source "$REPO/experiments/lib/common.sh"
#
# Файл только ОБЪЯВЛЯЕТ функции и дефолты, ничего не делает при подключении:
# оркестраторы запускаются под `set -uo pipefail`, и побочный эффект в общем
# файле ломал бы их непредсказуемо и в разных местах.
#
# say() читает RUN_LOG и SAY_TIME_FMT в момент ВЫЗОВА, а не подключения.
# Поэтому скрипт вправе задать их после source — лишь бы до первого say.

# Общий диск. Тот же дефолт, что был вбит в скрипты до вынесения.
: "${STORAGE:=/mnt/storage-1}"

# Куда дублировать вывод say(). Не задан — печатаем только в stdout.
# Значение НЕ трогаем здесь: у скриптов это либо REPORT, либо LOG.
: "${RUN_LOG:=}"

# Формат метки времени. Дефолт — тот, что стоял в большинстве скриптов;
# sanity_* исторически печатают без даты и задают формат сами.
: "${SAY_TIME_FMT:=%m-%d %H:%M:%S}"

say() {
    if [[ -n "${RUN_LOG:-}" ]]; then
        echo "[$(date "+$SAY_TIME_FMT")] $*" | tee -a "$RUN_LOG"
    else
        echo "[$(date "+$SAY_TIME_FMT")] $*"
    fi
}

# Заголовок фазы: пустая строка + шапка. Скрипты со своими проверками между
# фазами (run_night.sh зовёт ещё check_disk) переопределяют phase после source.
phase() {
    if [[ -n "${RUN_LOG:-}" ]]; then echo | tee -a "$RUN_LOG"; else echo; fi
    say "########## $* ##########"
}

# mkchmod <путь-на-хосте-под-$STORAGE>
#
# Создать каталог и открыть его на запись. Делается ИЗ КОНТЕЙНЕРА, а не хостовым
# mkdir: файлы, созданные docker'ом, принадлежат root, и хостовой пользователь
# не может ни дописать в них, ни удалить. Внутри образа sft диск смонтирован
# как /storage, поэтому префикс $STORAGE снимается.
mkchmod() {
    local rel="${1#"$STORAGE"/}"
    docker run --rm -v "$STORAGE:/storage" --entrypoint bash sft \
        -c "mkdir -p /storage/$rel && chmod -R 777 /storage/$rel" 2>/dev/null
}

# count_samples <путь-датасета-ВНУТРИ-контейнера>
#
# Сколько сэмплов в собранном датасете. Печатает число или ничего, если прочитать
# не удалось, — вызывающий обязан проверить результат на пустоту и упасть внятно.
# Считается в образе sft: на хосте нет ни datasets, ни прав на каталог.
count_samples() {
    docker run --rm -v "$STORAGE:/storage" --entrypoint /opt/venv/bin/python sft \
        -c "from datasets import load_from_disk;print(len(load_from_disk('$1')))" 2>/dev/null
}
