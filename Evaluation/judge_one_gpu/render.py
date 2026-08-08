"""
render.py — рендер HTML в PNG + единая конвенция серых плейсхолдеров вместо <img>.

Самостоятельный файл: не импортирует metrics.py (там CLIP-модель грузится прямо
при импорте — незачем тянуть torch/clip только ради рендера и замены плейсхолдеров).

Один постоянный браузер (Chromium) на процесс, на общем фоновом event loop'е
в отдельном daemon-потоке — так вызывающий код может оставаться синхронным
(take_screenshot/render_html_to_png/prepare_and_render), а несколько
скриншотов одного сэмпла (pred.png + OCR-free рендеры в metrics.py) идут
конкурентно через render_many, без накладных расходов на запуск браузера
на каждый вызов.

Использование как модуля:
    from render import replace_images_with_placeholder, render_html_to_png
"""

import asyncio
import atexit
import os
from pathlib import Path

from bs4 import BeautifulSoup
from PIL import Image

# --- единая конвенция плейсхолдера ---
PLACEHOLDER_CLASSES = ["bg-gray-300", "w-full", "h-48", "rounded"]
PLACEHOLDER_STYLE = "background-color:#d1d5db;width:100%;height:12rem;border-radius:0.5rem;display:block;"

# Таймаут одного page.goto/page.screenshot, мс. Настраиваемый — на этапе
# рендера+метрик чекпоинта на сэмпл приходится несколько скриншотов подряд
# (pred + OCR-free блоки, см. metrics.get_blocks_ocr_free), и при высоком
# --render-workers процессы конкурируют за CPU/браузер; можно поднять через
# D2C_RENDER_TIMEOUT_MS или явный параметр timeout_ms в вызовах ниже.
DEFAULT_RENDER_TIMEOUT_MS = int(os.environ.get("D2C_RENDER_TIMEOUT_MS", "90000"))

# =============================================================================
# Единственный движок рендера: постоянный Chromium на фоновом event loop'е
# =============================================================================
_loop_thread = None
_loop: "asyncio.AbstractEventLoop | None" = None
_playwright_ctx = None
_browser = None
_browser_lock: "asyncio.Lock | None" = None


def _ensure_loop_thread():
    global _loop_thread, _loop
    if _loop is not None:
        return
    import threading

    ready = threading.Event()

    def _run():
        global _loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _loop = loop
        ready.set()
        loop.run_forever()

    _loop_thread = threading.Thread(target=_run, daemon=True, name="render-async-loop")
    _loop_thread.start()
    ready.wait(timeout=10.0)


def _run_coro(coro):
    _ensure_loop_thread()
    fut = asyncio.run_coroutine_threadsafe(coro, _loop)
    return fut.result()


async def _get_browser():
    global _playwright_ctx, _browser, _browser_lock
    if _browser_lock is None:
        _browser_lock = asyncio.Lock()
    if _browser is not None:
        return _browser
    async with _browser_lock:
        if _browser is None:
            from playwright.async_api import async_playwright
            # --disable-gpu и связанные флаги: контейнер пробрасывает GPU для
            # CUDA (vLLM/PyTorch), но не DRM/VAAPI-устройства, нужные Chromium
            # для аппаратного композитинга — без них GPU-процесс Chromium
            # падает в crash loop и закрывает весь браузер. Скриншотим
            # статичный HTML/Tailwind без WebGL/canvas-анимаций, так что
            # программный растеризатор достаточен и, что важнее, не падает.
            _playwright_ctx = await async_playwright().start()
            _browser = await _playwright_ctx.chromium.launch(args=[
                "--disable-gpu",
                "--disable-gpu-compositing",
                "--disable-software-rasterizer",
                "--disable-dev-shm-usage",
            ])
    return _browser


async def _recreate_browser():
    """Форсированно закрывает и заново поднимает браузер — используется как
    восстановление после подозрения на мёртвый браузер (страница отказалась
    открываться/скриншотиться и после этого браузер, скорее всего, мёртв
    целиком — единичный retry на том же мёртвом браузере не поможет)."""
    global _browser, _playwright_ctx
    async with _browser_lock:
        if _browser is not None:
            try:
                await _browser.close()
            except Exception:
                pass
            _browser = None
        if _playwright_ctx is not None:
            try:
                await _playwright_ctx.stop()
            except Exception:
                pass
            _playwright_ctx = None
    return await _get_browser()


async def _close_browser_coro():
    global _playwright_ctx, _browser
    if _browser is not None:
        try:
            await _browser.close()
        finally:
            _browser = None
    if _playwright_ctx is not None:
        try:
            await _playwright_ctx.stop()
        finally:
            _playwright_ctx = None


def close_browser():
    """Закрывает общий браузер и останавливает фоновый loop-поток. Синхронная
    функция — безопасно вызывать из atexit / основного потока. Вызывается
    перед запуском ProcessPoolExecutor, чтобы не тащить объект браузера в
    форкнутые/заспавненные дочерние процессы."""
    global _loop, _loop_thread
    if _loop is not None:
        try:
            fut = asyncio.run_coroutine_threadsafe(_close_browser_coro(), _loop)
            fut.result(timeout=30.0)
        except Exception:
            pass
        _loop.call_soon_threadsafe(_loop.stop)
        if _loop_thread is not None:
            _loop_thread.join(timeout=10.0)
        _loop = None
        _loop_thread = None


class RenderResult:
    """Итог одного скриншота: ok, причина сбоя (для диагностики — раньше
    любой сбой молча превращался в белую заглушку без различения причины) и
    сколько попыток потребовалось."""

    __slots__ = ("ok", "reason", "attempts")

    def __init__(self, ok: bool, reason: str = None, attempts: int = 1):
        self.ok = ok
        self.reason = reason  # None | "timeout" | "other"
        self.attempts = attempts


async def take_screenshot_async(url, output_file="screenshot.png", do_it_again=False,
                                 timeout_ms: int = None, retries: int = 1) -> RenderResult:
    """Рендерит один URL/файл в PNG. При провале различает timeout от прочих
    ошибок (RenderResult.reason) вместо того чтобы просто печатать в stdout —
    так вызывающий код (и агрегированная статистика батча) видит, что именно
    произошло, а не только финальный render_ok=False.

    retries: сколько ДОПОЛНИТЕЛЬНЫХ попыток сделать после первой неудачи (по
    умолчанию 1 — то есть максимум 2 попытки всего). Первая неудача часто
    означает, что браузер уже мёртв (см. _recreate_browser) — простой повтор
    на том же браузере обычно бесполезен, поэтому retry идёт через
    пересоздание браузера, а не просто "попробовать ещё раз с тем же page"."""
    if timeout_ms is None:
        timeout_ms = DEFAULT_RENDER_TIMEOUT_MS
    if os.path.exists(url):
        url = "file://" + os.path.abspath(url)
    if os.path.exists(output_file) and not do_it_again:
        return RenderResult(ok=True)

    from playwright.async_api import TimeoutError as PlaywrightTimeoutError

    last_reason = None
    for attempt in range(1, retries + 2):  # 1 первая попытка + retries повторов
        try:
            browser = await _get_browser()
            page = await browser.new_page()
            try:
                await page.goto(url, timeout=timeout_ms)
                await page.screenshot(path=output_file, full_page=True,
                                       animations="disabled", timeout=timeout_ms)
            finally:
                await page.close()
            return RenderResult(ok=True, attempts=attempt)
        except PlaywrightTimeoutError as e:
            last_reason = "timeout"
            print(f"[render] Таймаут ({timeout_ms}мс) на попытке {attempt} для {url}: {e}")
        except Exception as e:
            last_reason = "other"
            print(f"[render] Ошибка рендера на попытке {attempt} для {url}: {e}")

        if attempt <= retries:
            # Пересоздаём браузер перед повтором — сбой page.goto/screenshot
            # часто значит, что общий браузер процесса уже в нерабочем
            # состоянии (см. docstring _recreate_browser); без пересоздания
            # повтор почти всегда падает той же ошибкой.
            try:
                await _recreate_browser()
            except Exception as e:
                print(f"[render] Не удалось пересоздать браузер перед повтором: {e}")

    Image.new('RGB', (1280, 960), color='white').save(output_file)
    return RenderResult(ok=False, reason=last_reason)


def take_screenshot(url, output_file="screenshot.png", do_it_again=False,
                     timeout_ms: int = None, retries: int = 1) -> RenderResult:
    """Синхронная точка входа для одиночного рендера."""
    return _run_coro(take_screenshot_async(
        url, output_file=output_file, do_it_again=do_it_again,
        timeout_ms=timeout_ms, retries=retries,
    ))


async def _render_many_coro(jobs: list[dict], max_concurrency: int,
                             timeout_ms: int, retries: int) -> list[RenderResult]:
    if not jobs:
        return []
    sem = asyncio.Semaphore(max_concurrency)

    async def _one(job):
        async with sem:
            return await take_screenshot_async(
                job["html"], output_file=job["png"], do_it_again=job.get("overwrite", True),
                timeout_ms=timeout_ms, retries=retries,
            )

    return await asyncio.gather(*(_one(job) for job in jobs))


def render_many(jobs: list[dict], max_concurrency: int = 6,
                 timeout_ms: int = None, retries: int = 1) -> list[RenderResult]:
    """Рендерит несколько (html, png) задач конкурентно на общем браузере.

    jobs: список {"html": путь_к_html, "png": путь_к_png, "overwrite": bool}.
    max_concurrency: верхний предел одновременно открытых вкладок.
    Возвращает список RenderResult в том же порядке, что jobs."""
    if not jobs:
        return []
    return _run_coro(_render_many_coro(jobs, max_concurrency,
                                        timeout_ms or DEFAULT_RENDER_TIMEOUT_MS, retries))


atexit.register(close_browser)


def replace_images_with_placeholder(html_text: str) -> tuple[str, int]:
    """Заменяет каждый <img> на div-плейсхолдер. Возвращает (новый_html, число_замен).

    Заменяет только <img> — если модель рисует "картинки" другими средствами
    (например div с background-color через инлайн-стиль или CSS-класс), эта
    функция их не увидит и не тронет; это не баг постобработки, а следствие
    того, что модель не выразила изображение тегом <img>. Такие случаи стоит
    ловить на уровне промпта/данных, не здесь."""
    soup = BeautifulSoup(html_text, "html.parser")
    n_replaced = 0
    for img_tag in soup.find_all("img"):
        new_div = soup.new_tag("div")
        new_div["class"] = PLACEHOLDER_CLASSES
        new_div["style"] = PLACEHOLDER_STYLE
        img_tag.replace_with(new_div)
        n_replaced += 1
    return str(soup), n_replaced


def render_html_to_png(html_path: str, png_path: str, overwrite: bool = True,
                        timeout_ms: int = None, retries: int = 1) -> RenderResult:
    """Рендерит HTML в PNG тем же движком, что и официальные метрики
    (Playwright, full-page screenshot)."""
    html_path = str(Path(html_path).resolve())
    png_path = str(Path(png_path).resolve())
    return take_screenshot(html_path, output_file=png_path, do_it_again=overwrite,
                            timeout_ms=timeout_ms, retries=retries)


def prepare_and_render(html_text: str, html_out_path: str, png_out_path: str,
                        timeout_ms: int = None, retries: int = 1) -> dict:
    """Заменяет плейсхолдеры, сохраняет HTML, рендерит в PNG."""
    clean_html, n_replaced = replace_images_with_placeholder(html_text)
    Path(html_out_path).write_text(clean_html, encoding="utf-8")
    result = render_html_to_png(html_out_path, png_out_path, timeout_ms=timeout_ms, retries=retries)
    return {
        "html_path": html_out_path,
        "png_path": png_out_path,
        "n_images_replaced": n_replaced,
        "render_ok": result.ok,
        "render_fail_reason": result.reason,
    }


def prepare_and_render_many(items: list[tuple[str, str, str]], max_concurrency: int = 6,
                             timeout_ms: int = None, retries: int = 1) -> list[dict]:
    """Batched-версия prepare_and_render: заменяет плейсхолдеры и сохраняет
    HTML для каждого элемента, затем рендерит все PNG одним render_many
    (конкурентно на общем браузере). Возвращает список dict в том же
    порядке и с тем же контрактом полей, что prepare_and_render."""
    prepped = []
    for html_text, html_out_path, png_out_path in items:
        clean_html, n_replaced = replace_images_with_placeholder(html_text)
        Path(html_out_path).write_text(clean_html, encoding="utf-8")
        prepped.append({
            "html_path": html_out_path,
            "png_path": png_out_path,
            "n_images_replaced": n_replaced,
        })

    jobs = [{"html": p["html_path"], "png": p["png_path"], "overwrite": True} for p in prepped]
    results = render_many(jobs, max_concurrency=max_concurrency, timeout_ms=timeout_ms, retries=retries)

    for p, result in zip(prepped, results):
        p["render_ok"] = result.ok
        p["render_fail_reason"] = result.reason
    return prepped


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Рендер HTML в PNG с заменой <img> на серый плейсхолдер.")
    parser.add_argument("--html", required=True)
    parser.add_argument("--png", required=True)
    parser.add_argument("--no-placeholder", action="store_true", help="Не заменять <img>, рендерить как есть")
    parser.add_argument("--timeout-ms", type=int, default=None)
    args = parser.parse_args()

    if args.no_placeholder:
        result = render_html_to_png(args.html, args.png, timeout_ms=args.timeout_ms)
        print(f"Рендер {'OK' if result.ok else f'FAILED ({result.reason})'} -> {args.png}")
    else:
        html_text = Path(args.html).read_text(encoding="utf-8")
        clean_path = str(Path(args.html).with_suffix(".placeholder.html"))
        clean_html, n = replace_images_with_placeholder(html_text)
        Path(clean_path).write_text(clean_html, encoding="utf-8")
        result = render_html_to_png(clean_path, args.png, timeout_ms=args.timeout_ms)
        print(f"Заменено <img>: {n}. Рендер {'OK' if result.ok else f'FAILED ({result.reason})'} -> {args.png}")
