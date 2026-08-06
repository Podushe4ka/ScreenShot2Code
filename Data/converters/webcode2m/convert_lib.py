#!/usr/bin/env python3
"""convert_lib.py (WebCode2M) — логика конвертера реального корпуса WebCode2M
(`xcodemind/webcode2m_purified`) в формат drafting-контракта (`SFT/DATA_FORMAT_CONTRACT.md`).

Отличия от WebSight-конвертера (`Data/converters/websight/convert_lib.py`):
  • источник — РЕАЛЬНЫЕ pruned-страницы, CSS уже лежит в `<style>`/`style=` (это НЕ Tailwind),
    поэтому `precompile_tailwind` НЕ применяется;
  • страницы могут тянуть внешние ресурсы (JS, `<link>` CSS, шрифты) — для детерминированного
    оффлайн-рендера вырезаем `<script>` и внешние `<link rel=stylesheet>` (инлайновый `<style>`
    остаётся), плюс де-блоб data-URI;
  • всё ОБЩЕЕ (плейсхолдеры, `render_full`, счётчик токенов, схема `FEATURES`) переиспользуется
    из `Data/converters/websight/convert_lib.py` — один источник правды, не дублируем.

Профили как у drafting: интерактив — `convert.ipynb`, батч — `convert_parallel.py`.
"""
import importlib.util
import io
import os
import re

from bs4 import BeautifulSoup

# ---- переиспользуем протестированное ядро из drafting-конвертера (один источник правды) ----
_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "drafting", "convert_lib.py")
_spec = importlib.util.spec_from_file_location("drafting_convert_lib", _BASE)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

replace_images_with_placeholder = _base.replace_images_with_placeholder
strip_background_images = _base.strip_background_images
render_full = _base.render_full
render_threaded = _base.render_threaded
close_renderer = _base.close_renderer
ahash = _base.ahash
hamming = _base.hamming
qwen_image_tokens = _base.qwen_image_tokens
count_tokens = _base.count_tokens
FEATURES = _base.FEATURES
RENDER_WIDTH = _base.RENDER_WIDTH
MIN_PIXELS = _base.MIN_PIXELS
MAX_PIXELS = _base.MAX_PIXELS
TOKENIZER_ID_DEFAULT = _base.TOKENIZER_ID_DEFAULT

# ---- источник ----
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
