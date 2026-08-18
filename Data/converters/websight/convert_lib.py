"""
convert_lib.py — логика конвертера WebSight → формат контракта (drafting).

Импортируется и интерактивным ноутбуком (convert.ipynb), и батч-скриптом
(convert_parallel.py) — чтобы правки/фиксы жили в ОДНОМ месте.

⚠ Что здесь ЕСТЬ, а чего нет. Всё общее для конвертеров — бюджет токенов, рендер,
плейсхолдеры, схема контракта, хэши near-dup — переехало в `../common/` (см. его
docstring: до этого было три копии рендера с разными флагами Chromium и две реализации
плейсхолдера). Здесь остаётся только специфика WebSight:
  * Tailwind precompile (CDN -> инлайновый <style>, v4 через pytailwindcss);
  * `page_domains` — домены страницы для декотаминации;
  * `process_one` — воркер батча (плейсхолдеры + precompile + рендер).
Имена общего ядра ре-экспортируются ниже, чтобы `from convert_lib import ...` в
convert_parallel.py и в ноутбуке продолжал работать без переписывания.
"""
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile

from urllib.parse import urlparse

from bs4 import BeautifulSoup

# Пролог доступа к общему ядру — ОДИН И ТОТ ЖЕ во всех точках входа (см. common/__init__.py).
_CONV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CONV not in sys.path:
    sys.path.insert(0, _CONV)

from common.budget import (MAX_PIXELS, MIN_PIXELS, RENDER_WIDTH,  # noqa: E402
                           TOKENIZER_ID_DEFAULT, count_tokens, qwen_image_tokens,
                           recommend_max_length)
from common.imaging import ahash, fit_to_size, hamming  # noqa: E402
from common.placeholders import (PLACEHOLDER_CLASSES, PLACEHOLDER_STYLE,  # noqa: E402
                                 replace_images_with_placeholder, strip_background_images)
from common.render import close_threaded, render_full, render_threaded  # noqa: E402
from common.schema import FEATURES  # noqa: E402

# Историческое имя: так браузер закрывают ноутбуки (convert.ipynb обоих конвертеров) и
# view_arrow.py. Именно `close_threaded`, а не `close`: ноутбук рендерит через поток, и
# закрывать браузер надо из того же потока, где он создан.
close_renderer = close_threaded

_TW_CDN_RE = re.compile(r'<script\b[^>]*tailwind[^>]*>\s*</script>|<link\b[^>]*tailwind[^>]*>', re.I)


def page_domains(html_text):
    """Домены, на которые ссылается страница (для декотаминации)."""
    doms = set()
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup.find_all(["a", "img", "link", "script", "source"]):
        url = (tag.get("href") or tag.get("src") or "").strip()
        if not url or url.startswith(("#", "mailto:", "tel:", "javascript:", "data:")):
            continue
        t = "http:" + url if url.startswith("//") else url
        net = urlparse(t).netloc.lower()
        if net:
            doms.add(net)
    return doms


def precompile_tailwind(html_text):
    """CDN-Tailwind -> инлайновый <style> (только используемые классы). Tailwind v4 через
    standalone `pytailwindcss` (без Node): input '@import "tailwindcss"' + '@source' на HTML."""
    tw = shutil.which("tailwindcss")
    if tw is None:
        raise RuntimeError("нужен tailwindcss CLI: pip install pytailwindcss")
    clean = _TW_CDN_RE.sub("", html_text)
    with tempfile.TemporaryDirectory() as t:
        with open(os.path.join(t, "page.html"), "w", encoding="utf-8") as f:
            f.write(clean)
        with open(os.path.join(t, "in.css"), "w") as f:
            f.write('@import "tailwindcss";\n@source "./page.html";\n')
        subprocess.run([tw, "-i", "in.css", "-o", "out.css", "--minify"],
                       cwd=t, check=True, capture_output=True, text=True)
        with open(os.path.join(t, "out.css"), encoding="utf-8") as f:
            css = f.read()
    style = f"<style>{css}</style>"
    low = clean.lower()
    if "</head>" in low:
        i = low.index("</head>")
        return clean[:i] + style + clean[i:]
    return "<!doctype html><html><head><meta charset='utf-8'>" + style + "</head><body>" + clean + "</body></html>"


def process_one(html_text):
    """Плейсхолдеры + precompile + рендер. Возвращает ("ok", target_html, png_bytes)
    или ("err", msg, traceback). Ошибка на странице не роняет пул."""
    try:
        html, _ = replace_images_with_placeholder(html_text)
        html = precompile_tailwind(html)
        img = render_full(html, RENDER_WIDTH)
        buf = io.BytesIO(); img.save(buf, "PNG")
        return ("ok", html, buf.getvalue())
    except Exception as e:
        import traceback
        return ("err", f"{type(e).__name__}: {e}", traceback.format_exc())
