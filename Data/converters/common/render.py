"""Рендер страницы через Playwright — один браузер на процесс, одни флаги.

Копий этого кода было три (websight, webui, complexity/features), и флаги запуска
Chromium в них РАЗЛИЧАЛИСЬ: у двух стоял набор `--disable-gpu ...` с пометкой «без них
GPU-процесс уходит в crash loop и утаскивает браузер целиком», у websight — ничего.
То есть один и тот же прогон в контейнере падал или не падал в зависимости от того, чей
конвертер его запустил. Флаги здесь — объединение, то есть более безопасный вариант.
"""
import io
import threading
from concurrent.futures import ThreadPoolExecutor

from PIL import Image as PILImage

from common.budget import RENDER_WIDTH

# Те же флаги, что в Evaluation/metrics_only/render.py: без DRM-устройств GPU-процесс
# Chromium уходит в crash loop и утаскивает браузер целиком.
CHROMIUM_ARGS = [
    "--disable-gpu", "--disable-gpu-compositing",
    "--disable-software-rasterizer", "--disable-dev-shm-usage",
]

# `owner` — поток, в котором создан браузер. Sync-API Playwright привязан к греенлету
# своего потока: закрыть браузер из ЧУЖОГО потока нельзя, будет
# `greenlet.error: Cannot switch to a different thread`. Раньше эта пара (создать в main
# через render_full, закрыть через close_renderer) роняла закрытие — воспроизводится на
# конвейере WebCode2M. Теперь close() сам разбирается, откуда звать.
_PW = {"pw": None, "browser": None, "owner": None}
# Для ноутбука: sync-API Playwright не работает внутри asyncio-loop Jupyter, поэтому
# render_threaded уводит вызов в отдельный поток. Ровно один воркер — браузер не потокобезопасен.
_RENDER_EXEC = ThreadPoolExecutor(max_workers=1)


def browser():
    if _PW["browser"] is None:
        from playwright.sync_api import sync_playwright
        _PW["pw"] = sync_playwright().start()
        _PW["browser"] = _PW["pw"].chromium.launch(args=CHROMIUM_ARGS)
        _PW["owner"] = threading.get_ident()
    return _PW["browser"]


def _close_here():
    if _PW["browser"] is not None:
        try:
            _PW["browser"].close()
        finally:
            _PW["browser"] = None
    if _PW["pw"] is not None:
        try:
            _PW["pw"].stop()
        finally:
            _PW["pw"] = None
    _PW["owner"] = None


def close():
    """Закрыть браузер и остановить Playwright. Идемпотентна и безопасна из любого потока:
    если браузер создан не здесь, закрытие уезжает в поток-владелец."""
    owner = _PW["owner"]
    if owner is None:                       # браузера нет — закрывать нечего
        _close_here()
        return
    if owner == threading.get_ident():
        _close_here()
        return
    # Владелец — рендер-поток (единственный, кроме текущего, кто мог создать браузер).
    _RENDER_EXEC.submit(_close_here).result()


# Историческое имя. Раньше оно означало «закрой из рендер-потока»; теперь close() сам
# выбирает поток, так что это просто алиас — оба вызова корректны.
close_threaded = close


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
    page = browser().new_page(viewport={"width": width, "height": 1024}, device_scale_factor=1)
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
