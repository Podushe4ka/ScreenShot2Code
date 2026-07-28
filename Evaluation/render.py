"""
render.py — рендер HTML в PNG + единая конвенция серых плейсхолдеров вместо <img>.

Самостоятельный файл: не импортирует metrics.py (там CLIP-модель грузится прямо
при импорте — незачем тянуть torch/clip только ради рендера и замены плейсхолдеров).
take_screenshot скопирована сюда же — тот же самый рендерер (Playwright, full-page
screenshot), что использует metrics.py, картинки будут 1-в-1.

Использование как модуля:
    from render import replace_images_with_placeholder, render_html_to_png
"""

import atexit
import os
from pathlib import Path

from bs4 import BeautifulSoup
from PIL import Image
from playwright.sync_api import sync_playwright

# --- единая конвенция плейсхолдера (см. обсуждение по контракту данных §4a) ---
PLACEHOLDER_CLASSES = ["bg-gray-300", "w-full", "h-48", "rounded"]
PLACEHOLDER_STYLE = "background-color:#d1d5db;width:100%;height:12rem;border-radius:0.5rem;display:block;"

# --- переиспользуемый Playwright/Chromium на весь процесс -------------------
# Раньше take_screenshot открывала sync_playwright()+launch() на КАЖДЫЙ вызов
# (а на один сэмпл их 4+: ref, pred, и две перекрашенные копии в get_blocks_ocr_free
# внутри metrics.py). Запуск браузера — это сотни мс - секунды накладных
# расходов на вызов; при батче сэмплов они складываются в заметную долю
# общего времени рендера. Держим один браузер живым на весь прогон и просто
# открываем/закрываем страницу (page) на каждый скриншот — страница дешёвая,
# процесс браузера — нет.
_playwright_ctx = None
_browser = None


def _get_browser():
    global _playwright_ctx, _browser
    if _browser is None:
        _playwright_ctx = sync_playwright().start()
        # --disable-gpu и связанные флаги: контейнер пробрасывает GPU для
        # CUDA (vLLM/PyTorch), но не DRM/VAAPI-устройства, нужные Chromium
        # для аппаратного композитинга. Без этих флагов GPU-процесс Chromium
        # падает (crash loop -> "GPU process isn't usable. Goodbye."), после
        # чего браузер закрывается целиком и ВСЕ дальнейшие скриншоты в этом
        # процессе валятся с "Target page, context or browser has been
        # closed" - это резко бьёт при нескольких параллельных воркерах
        # (--num-workers), так как конкуренция за software-рендеринг растёт.
        # Скриншотим статичный HTML/Tailwind без WebGL/canvas-анимаций,
        # так что аппаратный композитинг не нужен - программный растеризатор
        # (--disable-gpu) медленнее на сложных canvas-сценах, но здесь не
        # играет роли и, что важнее, не падает.
        _browser = _playwright_ctx.chromium.launch(args=[
            "--disable-gpu",
            "--disable-gpu-compositing",
            "--disable-software-rasterizer",
            "--disable-dev-shm-usage",
        ])
    return _browser


def close_browser():
    """Закрыть общий браузер (например, в конце run_benchmark.py)."""
    global _playwright_ctx, _browser
    if _browser is not None:
        try:
            _browser.close()
        finally:
            _browser = None
    if _playwright_ctx is not None:
        try:
            _playwright_ctx.stop()
        finally:
            _playwright_ctx = None


atexit.register(close_browser)


def take_screenshot(url, output_file="screenshot.png", do_it_again=False):
    """Тот же рендерер, что в metrics.py (скопировано, не импортировано — см. docstring выше).
    Использует общий процесс браузера (см. _get_browser) вместо запуска нового
    на каждый вызов."""
    if os.path.exists(url):
        url = "file://" + os.path.abspath(url)
    if os.path.exists(output_file) and not do_it_again:
        return
    try:
        browser = _get_browser()
        page = browser.new_page()
        try:
            page.goto(url, timeout=60000)
            page.screenshot(path=output_file, full_page=True, animations="disabled", timeout=60000)
        finally:
            page.close()
    except Exception as e:
        print(f"Failed to take screenshot due to: {e}. Generating a blank image.")
        Image.new('RGB', (1280, 960), color='white').save(output_file)


def replace_images_with_placeholder(html_text: str) -> tuple[str, int]:
    """Заменяет каждый <img> на div-плейсхолдер. Возвращает (новый_html, число_замен)."""
    soup = BeautifulSoup(html_text, "html.parser")
    n_replaced = 0
    for img_tag in soup.find_all("img"):
        new_div = soup.new_tag("div")
        new_div["class"] = PLACEHOLDER_CLASSES
        new_div["style"] = PLACEHOLDER_STYLE
        img_tag.replace_with(new_div)
        n_replaced += 1
    return str(soup), n_replaced


def render_html_to_png(html_path: str, png_path: str, overwrite: bool = True) -> bool:
    """Рендерит HTML в PNG тем же движком, что и официальные метрики (Playwright,
    full-page screenshot). При ошибке take_screenshot сам создаёт пустую белую
    картинку-заглушку, чтобы пайплайн не падал на битом рендере."""
    html_path = str(Path(html_path).resolve())
    png_path = str(Path(png_path).resolve())
    take_screenshot(html_path, output_file=png_path, do_it_again=overwrite)
    return Path(png_path).exists()


def prepare_and_render(html_text: str, html_out_path: str, png_out_path: str) -> dict:
    """Шорткат для run_benchmark.py: заменяет плейсхолдеры, сохраняет HTML, рендерит в PNG."""
    clean_html, n_replaced = replace_images_with_placeholder(html_text)
    Path(html_out_path).write_text(clean_html, encoding="utf-8")
    ok = render_html_to_png(html_out_path, png_out_path)
    return {
        "html_path": html_out_path,
        "png_path": png_out_path,
        "n_images_replaced": n_replaced,
        "render_ok": ok,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Рендер HTML в PNG с заменой <img> на серый плейсхолдер.")
    parser.add_argument("--html", required=True)
    parser.add_argument("--png", required=True)
    parser.add_argument("--no-placeholder", action="store_true", help="Не заменять <img>, рендерить как есть")
    args = parser.parse_args()

    if args.no_placeholder:
        ok = render_html_to_png(args.html, args.png)
        print(f"Рендер {'OK' if ok else 'FAILED'} -> {args.png}")
    else:
        html_text = Path(args.html).read_text(encoding="utf-8")
        clean_path = str(Path(args.html).with_suffix(".placeholder.html"))
        clean_html, n = replace_images_with_placeholder(html_text)
        Path(clean_path).write_text(clean_html, encoding="utf-8")
        ok = render_html_to_png(clean_path, args.png)
        print(f"Заменено <img>: {n}. Рендер {'OK' if ok else 'FAILED'} -> {args.png}")
