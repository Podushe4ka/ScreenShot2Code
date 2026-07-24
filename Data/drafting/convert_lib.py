"""
convert_lib.py — общая логика конвертера WebSight → формат контракта (drafting).

Единый источник правды. Импортируется и интерактивным ноутбуком (convert.ipynb),
и батч-скриптом (convert_parallel.py) — чтобы правки/фиксы жили в ОДНОМ месте.

Слои:
  * константы схемы/рендера/токенов;
  * гигиена HTML (плейсхолдеры вместо <img> и CSS background-image, дедуп, near-dup, decontam);
  * Tailwind precompile (CDN -> инлайновый <style>, v4 через pytailwindcss);
  * рендер (Playwright): render_full — прямой sync (скрипт/воркер); render_threaded — в потоке (ноутбук в Jupyter);
  * process_one — воркер батча (плейсхолдеры + precompile + рендер);
  * оценка токенов (кода и картинки).
"""
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from bs4 import BeautifulSoup
from datasets import Features, Image, Sequence, Value
from PIL import Image as PILImage
from urllib.parse import urlparse

# --- count_tokens / recommend_max_length из ../analysis/token_len.py (без дублирования) ---
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
from token_len import count_tokens, recommend_max_length  # noqa: E402

# ------------------------------------------------------------------ константы
RENDER_WIDTH = 1280                    # ширина вьюпорта ре-рендера; высота — по контенту
MIN_PIXELS = 256 * 32 * 32             # оценка визуальных токенов (как в SFT/train/formatting.py)
MAX_PIXELS = 1280 * 32 * 32
TOKENIZER_ID_DEFAULT = "Qwen/Qwen3-VL-8B-Instruct"

# серые плейсхолдеры — РОВНО как в Evaluation/Experiments.ipynb (менять синхронно с eval)
PLACEHOLDER_CLASSES = ["bg-gray-300", "w-full", "h-48", "rounded"]
PLACEHOLDER_STYLE = ("background-color:#d1d5db;width:100%;height:12rem;"
                     "border-radius:0.5rem;display:block;")

FEATURES = Features({                   # схема сэмпла (контракт §2)
    "task_type":    Value("string"),
    "images":       Sequence(Image()),
    "current_html": Value("string"),
    "target_html":  Value("string"),
    "instruction":  Value("string"),
})

# ------------------------------------------------------------ гигиена HTML
_BG_RE = re.compile(r'background(-image)?\s*:\s*[^;{}"\']*url\([^)]*\)[^;{}"\']*', re.I)
_TW_CDN_RE = re.compile(r'<script\b[^>]*tailwind[^>]*>\s*</script>|<link\b[^>]*tailwind[^>]*>', re.I)


def strip_background_images(html_text):
    """CSS background-image: url(...) -> серый фон (hero-фото — тоже картинка)."""
    return _BG_RE.sub("background-color:#d1d5db", html_text)


def replace_images_with_placeholder(html_text):
    """<img> -> серый <div>; плюс background-image -> серый фон. Возвращает (html, n_img)."""
    soup = BeautifulSoup(html_text, "html.parser")
    n = 0
    for img in soup.find_all("img"):
        div = soup.new_tag("div")
        div["class"] = PLACEHOLDER_CLASSES
        div["style"] = PLACEHOLDER_STYLE
        img.replace_with(div)
        n += 1
    return strip_background_images(str(soup)), n


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


def ahash(img, size=8):
    """Average-hash картинки (near-dup) — чистый PIL, без зависимостей."""
    g = img.convert("L").resize((size, size))
    px = list(g.getdata()); avg = sum(px) / len(px)
    return sum(1 << i for i, p in enumerate(px) if p > avg)


def hamming(a, b):
    return bin(a ^ b).count("1")


def fit_to_size(img, size):
    """Привести скриншот к (W,H): паддинг белым + обрезка. Только для смоука SIZE_MODE='pad'."""
    tw, th = size
    img = img.convert("RGB")
    canvas = PILImage.new("RGB", size, (255, 255, 255))
    canvas.paste(img.crop((0, 0, min(img.width, tw), min(img.height, th))), (0, 0))
    return canvas


# ------------------------------------------------- Tailwind precompile (v4)
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


# ------------------------------------------- рендер (Playwright, свой браузер на процесс)
_PW = {"pw": None, "browser": None}
_RENDER_EXEC = ThreadPoolExecutor(max_workers=1)   # для ноутбука: sync-API в потоке (в Jupyter asyncio-loop)


def _browser():
    if _PW["browser"] is None:
        from playwright.sync_api import sync_playwright
        _PW["pw"] = sync_playwright().start()
        _PW["browser"] = _PW["pw"].chromium.launch()
    return _PW["browser"]


def render_full(html_text, width=RENDER_WIDTH):
    """Прямой sync-рендер ВСЕЙ страницы: ширина фикс, высота — по контенту (body.scrollHeight).
    Годится для скрипта/воркера (отдельный процесс, нет asyncio-loop)."""
    page = _browser().new_page(viewport={"width": width, "height": 1024}, device_scale_factor=1)
    try:
        page.set_content(html_text, wait_until="networkidle")
        height = int(page.evaluate(
            "() => Math.max(document.body ? document.body.scrollHeight : "
            "document.documentElement.scrollHeight, 1)"))
        png = page.screenshot(clip={"x": 0, "y": 0, "width": width, "height": max(1, height)})
    finally:
        page.close()
    return PILImage.open(io.BytesIO(png)).convert("RGB")


def render_threaded(html_text, width=RENDER_WIDTH):
    """То же, но через поток — для ноутбука в Jupyter (sync-API не работает в asyncio-loop)."""
    return _RENDER_EXEC.submit(render_full, html_text, width).result()


def close_renderer():
    def _close():
        if _PW["browser"] is not None:
            _PW["browser"].close(); _PW["pw"].stop(); _PW["browser"] = _PW["pw"] = None
    _RENDER_EXEC.submit(_close).result()


# ------------------------------------------------------ воркер батча (для пула)
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


# ------------------------------------------------------------- оценка токенов
def qwen_image_tokens(w, h, patch=28):
    """Приближённо: процессор Qwen зажимает площадь в [MIN,MAX] пикселей; 1 токен ≈ 28x28 px."""
    px = min(max(w * h, MIN_PIXELS), MAX_PIXELS)
    return round(px / (patch * patch))
