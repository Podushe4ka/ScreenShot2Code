#!/usr/bin/env python3
"""convert_lib.py (WebCode2M) — логика конвертера реального корпуса WebCode2M
(`xcodemind/webcode2m_purified`) в формат drafting-контракта (`SFT/DATA_FORMAT_CONTRACT.md`).

Отличия от WebSight-конвертера (`Data/converters/websight/convert_lib.py`):
  • источник — РЕАЛЬНЫЕ pruned-страницы, CSS уже лежит в `<style>`/`style=` (это НЕ Tailwind),
    поэтому `precompile_tailwind` НЕ применяется;
  • страницы могут тянуть внешние ресурсы (JS, `<link>` CSS, шрифты) — для детерминированного
    оффлайн-рендера вырезаем `<script>` и внешние `<link rel=stylesheet>` (инлайновый `<style>`
    остаётся), плюс де-блоб data-URI;
  • всё ОБЩЕЕ (плейсхолдеры, `render_full`, счётчик токенов, схема `FEATURES`) берётся из
    `../common/` — один источник правды, не дублируем.

⚠ Раньше общее ядро подгружалось из `../websight/convert_lib.py` через
`importlib.spec_from_file_location` + `exec_module`. Это работало, но давало модуль, который
не находится по имени: при `multiprocessing` со `spawn` (macOS) воркеры такой модуль не
восстанавливают, а пикл функций из него падает. Теперь обычный импорт пакета `common`.

Профили как у drafting: интерактив — `convert.ipynb`, батч — `convert_parallel.py`.
"""
import io
import os
import re
import sys

from bs4 import BeautifulSoup

# Пролог доступа к общему ядру — ОДИН И ТОТ ЖЕ во всех точках входа (см. common/__init__.py).
_CONV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CONV not in sys.path:
    sys.path.insert(0, _CONV)

from common.budget import (MAX_PIXELS, MIN_PIXELS, RENDER_WIDTH,  # noqa: E402
                           TOKENIZER_ID_DEFAULT, count_tokens, qwen_image_tokens)
from common.imaging import ahash, hamming  # noqa: E402
from common.placeholders import (replace_images_with_placeholder,  # noqa: E402
                                 strip_background_images)
from common.render import close_threaded, render_full, render_threaded  # noqa: E402
from common.schema import FEATURES  # noqa: E402

close_renderer = close_threaded   # историческое имя: так браузер закрывает convert.ipynb

DATASET_ID = "xcodemind/webcode2m_purified"   # реальные pruned-страницы, HTML+CSS слиты в `text`
SPLIT = "train"
HTML_FIELD = "text"       # поле с HTML(+CSS)
IMAGE_FIELD = "image"     # поле со скриншотом (для near-dup хэша)

_DATA_URI_RE = re.compile(r'data:[^;,\s"\')]+;base64,[A-Za-z0-9+/=]+', re.I)


def strip_data_uris(html):
    """Вырезать длинные base64 data-URI (в purified редки, но безопасно — чистим до токенизации)."""
    return _DATA_URI_RE.sub("data:,", html or "")


def sanitize_offline(html_text):
    """Убрать то, что ломает детерминированный оффлайн-рендер реальных страниц:
    `<script>` и внешние `<link rel=stylesheet>`. Инлайновый `<style>` (основная масса CSS
    в WebCode2M) остаётся; внешние шрифты рендерятся системным фолбэком."""
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup.find_all("script"):
        tag.decompose()
    for link in soup.find_all("link"):
        rel = link.get("rel")
        rel = " ".join(rel).lower() if isinstance(rel, list) else (rel or "").lower()
        if "stylesheet" in rel:
            link.decompose()
    return str(soup)


def process_one(html_text):
    """WebCode2M: sanitize -> де-блоб -> плейсхолдеры -> рендер. БЕЗ precompile_tailwind.
    Возвращает ("ok", target_html, png_bytes) | ("err", msg, traceback). Ошибка страницы
    не роняет пул (та же конвенция, что в drafting)."""
    try:
        html = sanitize_offline(html_text)
        html = strip_data_uris(html)
        html, _ = replace_images_with_placeholder(html)      # <img> и CSS bg-url -> серый блок
        img = render_full(html, RENDER_WIDTH)                # full_page, ширина фикс, высота по контенту
        buf = io.BytesIO(); img.save(buf, "PNG")
        return ("ok", html, buf.getvalue())
    except Exception as e:
        import traceback
        return ("err", f"{type(e).__name__}: {e}", traceback.format_exc())
