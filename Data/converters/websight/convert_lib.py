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

# Подсчёт токенов вынесен в ЛЕНИВЫЕ обёртки: token_len -> transformers -> torch тянутся
# ТОЛЬКО при реальном вызове count_tokens/recommend_max_length, а не при import convert_lib.
# Так основной путь (сборка датасета, воркеры пула) не грузит transformers/torch.
_EDA_TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "eda", "tools")


def count_tokens(text, tokenizer):
    """Длина текста в токенах. Ленивая обёртка над ../../eda/tools/token_len.py."""
    if _EDA_TOOLS not in sys.path:
        sys.path.append(_EDA_TOOLS)
    from token_len import count_tokens as _ct
    return _ct(text, tokenizer)


def recommend_max_length(*args, **kwargs):
    """Ориентир max_length. Ленивая обёртка над token_len.recommend_max_length."""
    if _EDA_TOOLS not in sys.path:
        sys.path.append(_EDA_TOOLS)
    from token_len import recommend_max_length as _rml
    return _rml(*args, **kwargs)


RENDER_WIDTH = 1280                    # ширина вьюпорта ре-рендера; высота — по контенту
MIN_PIXELS = 256 * 32 * 32             # 262144 — совпадает с SFT/train/formatting.py
# ⚠ MAX_PIXELS здесь 1.31 Мп, а обучение и бенч идут на 2.10 Мп
# (SFT/train/formatting.py, SFT_MAX_PIXELS). Расхождение осталось с доTier-A времён:
# конвертер оценивает визуальный бюджет строже, чем он будет на самом деле.
# Значение НЕ трогать не глядя — от него зависит отбраковка в filter_by_height.py,
# то есть состав уже собранных наборов. Разбор — docs/experiments/DIVERGENCES.md, K1.
MAX_PIXELS = 1280 * 32 * 32
TOKENIZER_ID_DEFAULT = "Qwen/Qwen3-VL-8B-Instruct"

# Серые плейсхолдеры обязаны посимвольно совпадать с Evaluation/metrics_only/render.py — бенч
# подменяет <img> и в предсказании, и в эталоне, и расхождение конвенции сделало бы
# обучающие таргеты непохожими на то, что метрика видит на бенче.
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


_PW = {"pw": None, "browser": None}
_RENDER_EXEC = ThreadPoolExecutor(max_workers=1)   # для ноутбука: sync-API в потоке (в Jupyter asyncio-loop)


def _browser():
    if _PW["browser"] is None:
        from playwright.sync_api import sync_playwright
        _PW["pw"] = sync_playwright().start()
        _PW["browser"] = _PW["pw"].chromium.launch()
    return _PW["browser"]


def render_full(html_text, width=RENDER_WIDTH):
    """Прямой sync-рендер ВСЕЙ страницы: ширина фикс = width, высота = вся прокрутка.
    Годится для скрипта/воркера (отдельный процесс, нет asyncio-loop).

    ВАЖНО (не регрессировать) — почему именно full_page=True, а не ресайз+clip:
    Playwright при full_page привязывает `vh`/`min-h-screen`/`h-screen` к высоте ВЬЮПОРТА,
    а не к полной высоте контента, и сам захватывает всю прокрутку. Прошлые два подхода оба
    были неверны:
      • clip без ресайза на вьюпорте 1024 — обрезал всё выше 1024;
      • ресайз вьюпорта в scrollHeight + clip — тогда `100vh` героя = вся высота страницы,
        герой растягивался на весь кадр и выпихивал контент за обрез (скрин = сплошной фон
        героя, реального контента нет). Проверено: см. тест в истории коммита.
    Скроллбар мог красть 16px ширины (1296 вместо 1280) — страхуемся кропом до width."""
    page = _browser().new_page(viewport={"width": width, "height": 1024}, device_scale_factor=1)
    try:
        # wait_until="load", НЕ "networkidle": после precompile+плейсхолдеров страница
        # self-contained (внешних запросов нет), а networkidle всё равно ждёт 500мс "тишины"
        # на каждой странице (~574мс vs ~67мс на пустой сети — замер в истории коммита).
        page.set_content(html_text, wait_until="load")
        # Ретрай на первом захвате. Chromium сразу после старта браузера иногда отвечает
        # `Protocol error (Page.captureScreenshot): Unable to capture screenshot` — гонка
        # инициализации, а не дефект страницы: ТОТ ЖЕ html вторым в очереди снимается
        # штатно. Воспроизведено на пустой странице. Без ретрая случайно бракуется первая
        # страница каждого прогона, и брак выглядит как «страница не рендерится».
        for attempt in range(3):
            try:
                png = page.screenshot(full_page=True)
                break
            except Exception:
                if attempt == 2:
                    raise
                page.wait_for_timeout(400)
    finally:
        page.close()
    img = PILImage.open(io.BytesIO(png)).convert("RGB")
    if img.width != width:            # остаточная ширина скроллбара/горизонт. оверфлоу
        img = img.crop((0, 0, width, img.height))
    return img


def render_threaded(html_text, width=RENDER_WIDTH):
    """То же, но через поток — для ноутбука в Jupyter (sync-API не работает в asyncio-loop)."""
    return _RENDER_EXEC.submit(render_full, html_text, width).result()


def close_renderer():
    def _close():
        if _PW["browser"] is not None:
            _PW["browser"].close(); _PW["pw"].stop(); _PW["browser"] = _PW["pw"] = None
    _RENDER_EXEC.submit(_close).result()


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


def qwen_image_tokens(w, h, patch=32):
    """Приближённо: процессор Qwen зажимает площадь в [MIN,MAX] пикселей; 1 токен ≈ 32x32 px.

    patch=32 = patch_size*merge_size у Qwen3.5 (модель зафиксирована PLAN §4). Раньше стояло 28
    (Qwen2.5-VL) — это рассинхрон с MIN/MAX_PIXELS выше, которые уже считаны через 32: при 28
    делитель 784 давал ~1672 токена на потолке 1.31 Мп вместо фактических 1280. Теперь
    MAX_PIXELS/(32*32) = 1280 — сходится с бюджетом в filter_by_height.py."""
    px = min(max(w * h, MIN_PIXELS), MAX_PIXELS)
    return round(px / (patch * patch))
