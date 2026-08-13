#!/usr/bin/env bash
# Локальные копии CDN-библиотек для офлайн-рендера react_cdn-страниц.
#
# Зачем. Рендер и метрика ходят в сеть только через перехват: `_materialize_dom_coro`
# в Evaluation/metrics_only/render.py подменяет запросы по подстроке URL на локальный файл, а всё
# остальное внешнее делает route.abort(). Имена файлов здесь обязаны совпадать со
# значениями словаря `_VENDOR_MAP` в Evaluation/metrics_only/render.py — иначе перехват молча
# не сработает и страница отрендерится пустой.
#
# Использование:
#   bash Data/generators/synth/fetch_vendor.sh [DEST]
#   export VENDOR_DIR=$(pwd)/Data/vendor
set -euo pipefail

DEST="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/Data/vendor}"
mkdir -p "$DEST"

# Версии пиннуем: смена мажора Tailwind/React меняет рендер, а значит и скриншоты,
# а значит и несравнимость датасетов, собранных в разное время.
REACT_VER="18.3.1"
BABEL_VER="7.26.4"
TAILWIND_VER="3.4.16"
FA_VER="6.5.2"

fetch() {  # fetch <url> <имя файла в DEST>
  local url="$1" out="$DEST/$2"
  if [[ -s "$out" ]]; then
    echo "  = $2 (уже есть, $(wc -c <"$out" | tr -d ' ') Б)"
    return
  fi
  echo "  ↓ $2 <- $url"
  curl -fsSL --retry 3 --max-time 120 "$url" -o "$out"
  echo "    $(wc -c <"$out" | tr -d ' ') Б"
}

echo "Вендорим в $DEST"
fetch "https://unpkg.com/react@${REACT_VER}/umd/react.development.js"             react.development.js
fetch "https://unpkg.com/react@${REACT_VER}/umd/react.production.min.js"         react.production.min.js
fetch "https://unpkg.com/react-dom@${REACT_VER}/umd/react-dom.development.js"     react-dom.development.js
fetch "https://unpkg.com/react-dom@${REACT_VER}/umd/react-dom.production.min.js" react-dom.production.min.js
fetch "https://unpkg.com/@babel/standalone@${BABEL_VER}/babel.min.js"            babel.js
fetch "https://cdn.tailwindcss.com/${TAILWIND_VER}"                            tailwind.js
fetch "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/${FA_VER}/css/all.min.css" fontawesome.css

cat > "$DEST/VERSIONS.txt" <<EOF
react            $REACT_VER
react-dom        $REACT_VER
@babel/standalone $BABEL_VER
tailwindcss(cdn) $TAILWIND_VER
font-awesome     $FA_VER
EOF

echo
echo "Готово. Дальше:  export VENDOR_DIR=$DEST"
echo "Внимание: fontawesome.css тянет шрифты по относительному ../webfonts/ — они"
echo "не вендорятся и будут отброшены route.abort(). Иконки в генерации делаем"
echo "инлайновым <svg>, а не классами fa-*."
