"""
render.py — рендер HTML в PNG + единая конвенция серых плейсхолдеров вместо <img>.

Самостоятельный файл: не импортирует metrics.py (там CLIP-модель грузится прямо
при импорте — незачем тянуть torch/clip только ради рендера и замены плейсхолдеров).
take_screenshot скопирована сюда же — тот же самый рендерер (Playwright, full-page
screenshot), что использует metrics.py, картинки будут 1-в-1.

Использование как модуля:
    from render import replace_images_with_placeholder, render_html_to_png
"""

import asyncio
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


# =============================================================================
# АСИНХРОННЫЙ ДВИЖОК — рендер N скриншотов ОДНОГО сэмпла конкурентно
# =============================================================================
# Почему это нужно: на сэмпл сейчас 3 независимых скриншота — pred.png,
# и внутри get_blocks_ocr_free ещё pred_p.png + pred_p_1.png (плюс те же три
# для ref, если кэш эталона ещё не посчитан). Раньше все три шли СТРОГО
# последовательно внутри одного CPU-воркера через один и тот же браузер —
# хотя между ними нет зависимости по данным, а сама задача целиком I/O-bound
# (весь "вес" - это ожидание page.goto/page.screenshot внутри Chromium,
# питон в это время ничего не считает). Последовательные await'ы экономят
# ноль, конкурентные (asyncio.gather) - экономят почти весь простой.
#
# ВАЖНО про event loop: Playwright's async API привязывает свой internal
# driver-процесс к тому event loop'у, в котором был вызван .start(). Если
# каждый вызов render_many() создавал бы свой asyncio.run(...) (свой НОВЫЙ
# loop), а браузер при этом переиспользовался бы как модульная глобальная
# переменная между вызовами - второй вызов подсовывал бы объекту, созданному
# в loop #1, операции из loop #2, и это виснет (проверено эмпирически: без
# этого фонового потока процесс не завершался даже после явного close).
# Решение: один com постоянный event loop, живущий в отдельном
# daemon-потоке на весь процесс - и браузер, и все скриншоты всегда
# работают внутри ОДНОГО и того же loop'а, независимо от того, что
# render_many() вызывается синхронно из разных мест кода много раз подряд.
_loop_thread = None
_loop: "asyncio.AbstractEventLoop | None" = None
_async_playwright_ctx = None
_async_browser = None
_async_lock: "asyncio.Lock | None" = None


def _ensure_loop_thread():
    """Стартует фоновый поток с постоянным event loop, если ещё не запущен.
    Идемпотентно и потокобезопасно (простая проверка + порядок операций
    достаточны, т.к. вызывается только из главного потока каждого
    процесса — воркеры ProcessPoolExecutor не расшаривают этот модуль
    между потоками)."""
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
    """Выполняет coroutine на общем фоновом loop'е и блокирующе ждёт
    результат — синхронный вход в асинхронный мир для вызывающего кода."""
    _ensure_loop_thread()
    fut = asyncio.run_coroutine_threadsafe(coro, _loop)
    return fut.result()


async def _get_async_browser():
    global _async_playwright_ctx, _async_browser, _async_lock
    if _async_lock is None:
        _async_lock = asyncio.Lock()
    if _async_browser is not None:
        return _async_browser
    async with _async_lock:
        if _async_browser is None:  # двойная проверка — конкурентные вызовы могли ждать лок
            from playwright.async_api import async_playwright
            _async_playwright_ctx = await async_playwright().start()
            _async_browser = await _async_playwright_ctx.chromium.launch(args=[
                "--disable-gpu",
                "--disable-gpu-compositing",
                "--disable-software-rasterizer",
                "--disable-dev-shm-usage",
            ])
    return _async_browser


async def _close_async_browser_coro():
    global _async_playwright_ctx, _async_browser
    if _async_browser is not None:
        try:
            await _async_browser.close()
        finally:
            _async_browser = None
    if _async_playwright_ctx is not None:
        try:
            await _async_playwright_ctx.stop()
        finally:
            _async_playwright_ctx = None


def close_async_browser():
    """Закрывает общий асинхронный браузер и останавливает фоновый loop-поток.
    Синхронная функция — безопасно вызывать из atexit / основного потока,
    без ручного asyncio.run() на стороне вызывающего кода."""
    global _loop, _loop_thread
    if _loop is not None:
        try:
            fut = asyncio.run_coroutine_threadsafe(_close_async_browser_coro(), _loop)
            fut.result(timeout=30.0)
        except Exception:
            pass
        _loop.call_soon_threadsafe(_loop.stop)
        if _loop_thread is not None:
            _loop_thread.join(timeout=10.0)
        _loop = None
        _loop_thread = None


async def take_screenshot_async(url, output_file="screenshot.png", do_it_again=False):
    """Асинхронный аналог take_screenshot — идентичная логика/аргументы,
    но не блокирует поток на время page.goto/page.screenshot, так что
    несколько вызовов можно исполнять конкурентно через asyncio.gather.
    Выполняется на общем фоновом loop'е (см. _ensure_loop_thread) —
    вызывать напрямую только из кода, уже работающего на этом loop'е
    (используйте render_many/render_many_sync снаружи)."""
    if os.path.exists(url):
        url = "file://" + os.path.abspath(url)
    if os.path.exists(output_file) and not do_it_again:
        return
    try:
        browser = await _get_async_browser()
        page = await browser.new_page()
        try:
            await page.goto(url, timeout=60000)
            await page.screenshot(path=output_file, full_page=True, animations="disabled", timeout=60000)
        finally:
            await page.close()
    except Exception as e:
        print(f"Failed to take screenshot due to: {e}. Generating a blank image.")
        Image.new('RGB', (1280, 960), color='white').save(output_file)


async def _render_many_coro(jobs: list[dict], max_concurrency: int) -> None:
    if not jobs:
        return
    sem = asyncio.Semaphore(max_concurrency)

    async def _one(job):
        async with sem:
            await take_screenshot_async(
                job["html"], output_file=job["png"], do_it_again=job.get("overwrite", True)
            )

    await asyncio.gather(*(_one(job) for job in jobs))


def render_many(jobs: list[dict], max_concurrency: int = 6) -> None:
    """Рендерит несколько (html, png) задач КОНКУРЕНТНО на общем браузере,
    через постоянный фоновый event loop (см. _ensure_loop_thread) — так
    браузер переживает между вызовами render_many без loop-mismatch.

    jobs: список {"html": путь_к_html, "png": путь_к_png, "overwrite": bool}.
    max_concurrency: верхний предел одновременно открытых вкладок — не даём
    asyncio.gather открыть неограниченное число page сразу (при большом
    --num-workers это и так I/O-bound задача, но каждая вкладка — это
    процесс/память Chromium; предел защищает от скачка RAM/FD при редком
    сэмпле с необычно большим числом задач). 3 скриншота на сэмпл — обычный
    случай, так что предел почти никогда не активируется на практике.

    Ошибки НЕ пробрасываются наружу на уровень отдельной задачи — как и в
    take_screenshot, при сбое рендера конкретный png получит белую заглушку,
    остальные задачи батча всё равно завершатся."""
    if not jobs:
        return
    _run_coro(_render_many_coro(jobs, max_concurrency))


atexit.register(close_async_browser)


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


def prepare_and_render_many(items: list[tuple[str, str, str]], max_concurrency: int = 6) -> list[dict]:
    """Batched-версия prepare_and_render: принимает список
    (html_text, html_out_path, png_out_path), заменяет плейсхолдеры и
    сохраняет HTML для КАЖДОГО (дёшево, CPU-bound, без выгоды от asyncio),
    затем рендерит ВСЕ PNG одним render_many — конкурентно на общем
    браузере, а не по одному через prepare_and_render в цикле.

    Это основная точка входа для ускорения: там, где раньше вызывающий код
    делал `for x in items: prepare_and_render(x)` (N последовательных
    Chromium-рендеров), теперь один вызов рендерит их все параллельно.
    Возвращает список dict в том же порядке и с тем же контрактом полей,
    что prepare_and_render (html_path, png_path, n_images_replaced, render_ok)."""
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
    render_many(jobs, max_concurrency=max_concurrency)

    for p in prepped:
        p["render_ok"] = Path(p["png_path"]).exists()
    return prepped


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
