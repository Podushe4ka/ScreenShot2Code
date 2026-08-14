#!/usr/bin/env python3
"""convert_lib.py (WebUI) — WebUI -> self-contained HTML с инлайновым CSS, точный путь
через браузер.

Чем WebUI отличается от двух уже сделанных источников и почему нужен отдельный конвертер:

  • WebSight  — Tailwind-синтетика, стиль в utility-классах, лечится `precompile_tailwind`;
  • WebCode2M — реальные страницы, CSS уже внутри `<style>`/`style=`, лечится санитайзом;
  • WebUI     — реальные страницы, но CSS лежит ОТДЕЛЬНОЙ КОЛОНКОЙ `css`, и там весь
    стайлшит сайта целиком. У дизайн-систем это 170–470 КБ на страницу в сотню узлов.
    Отсюда p99 = 171 851 токен (`Data/eda/datasets_overview.md`), из-за которого WebUI и
    не взяли в MVP. Ни фильтр по длине, ни обрезка тут не помогают: обрезанный стайлшит
    ломает страницу. Помогает только tree-shaking — оставить правила, которые к этой
    странице реально применились.

Порядок операций (нумерация — по плану задачи):
  1. дедуп по `sample_id`, одна строка на страницу (desktop) — фаза 0, `fetch_columns.py`;
  2. `assemble_page`  — html + `<style>css</style>`, снос `<script>`, внешних шрифтов и
                        любых внешних URL, де-блоб data-URI;
  3. `treeshake_css`  — soupsieve-префильтр (`cssprune.prefilter`) + CDP-трекинг
                        применённых правил при рендере на 1280 (`cssprune.rebuild_from_usage`);
  4. `<img>`          -> канонический серый плейсхолдер (тот же, что в бенче и в промпте
                        генерации, `Data/generators/synth/prompts/01_page_generation.md`);
  5. ПЕРЕРЕНДЕР       — скриншот свой, через `Evaluation/metrics_only/render.py`. Картинка
                        из WebUI не годится: она снята при 1280×720, то есть только первый
                        экран, а `html` содержит страницу целиком — пара была бы
                        рассогласована (скриншот шапки, а в таргете вся страница);
  6. `accept_page`    — рендер ДО и ПОСЛЕ tree-shaking сравнивается НАШЕЙ ЖЕ метрикой
                        (`Evaluation/metrics_only/metrics.score_pair`); ниже порога —
                        страницу выбрасываем. Самопроверяющийся шаг: если шейкер сломал
                        вёрстку, это видно тем же числом, которым меряют модель;
  7. фильтр по токенам под `max_length` — в `convert_parallel.py`.

Общее ядро (схема `FEATURES`, плейсхолдеры, счётчик токенов, near-dup) берём из
`../websight/convert_lib.py` — один источник правды, второй рендерер/схему не плодим.
"""
import importlib.util
import io
import os
import re
import sys

from bs4 import BeautifulSoup

import cssprune

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))

# Переиспользуем протестированное ядро drafting-конвертера.
_BASE = os.path.join(_HERE, "..", "websight", "convert_lib.py")
_spec = importlib.util.spec_from_file_location("websight_convert_lib", _BASE)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

FEATURES = _base.FEATURES
RENDER_WIDTH = _base.RENDER_WIDTH
MIN_PIXELS, MAX_PIXELS = _base.MIN_PIXELS, _base.MAX_PIXELS
TOKENIZER_ID_DEFAULT = _base.TOKENIZER_ID_DEFAULT
PLACEHOLDER_CLASSES = _base.PLACEHOLDER_CLASSES
PLACEHOLDER_STYLE = _base.PLACEHOLDER_STYLE
ahash, hamming = _base.ahash, _base.hamming
count_tokens = _base.count_tokens
qwen_image_tokens = _base.qwen_image_tokens

# Рендер и метрика приёмки — из eval-трека, чтобы обучающий скриншот снимался ровно тем же
# движком и с той же конвенцией, которой потом меряют модель на бенче.
_EVAL = os.path.join(_REPO, "Evaluation", "metrics_only")
if _EVAL not in sys.path:
    sys.path.insert(0, _EVAL)

DEFAULT_ACCEPT_THRESHOLD = 0.95     # порог приёмки конвертации (шаг 6), стартовое значение
DEFAULT_PIXEL_THRESHOLD = 0.995     # запасной сигнал приёмки — попиксельное совпадение рендеров
DATA_URI_MAX = 1024                 # base64-полезная нагрузка длиннее — режем (де-блоб)

# ── HTML: парсер ──────────────────────────────────────────────────────────────
# Тот же порядок парсеров, что в Evaluation/metrics_only/render.py: html.parser на странице
# со `<script type="text/babel">` выбрасывает остаток документа при пересборке через
# str(soup). Здесь скрипты и так сносятся, но у WebUI встречается сырой JSX в разметке.
_PARSERS = ("html5lib", "html.parser")


def make_soup(html_text):
    for parser in _PARSERS:
        try:
            return BeautifulSoup(html_text, parser)
        except Exception:
            continue
    return BeautifulSoup(html_text, "html.parser")


def style_text(tag):
    """Содержимое `<style>`. НЕ `get_text()`.

    ⚠ Под html5lib текст внутри `<style>` — это узел `Stylesheet` (подкласс
    NavigableString), а bs4 считает его НЕ текстом: `get_text()` возвращает пустую строку,
    хотя `.string` отдаёт все 4221 символа. Под `html.parser` и `lxml` такого нет — то есть
    поведение молча зависит от того, какой парсер отработал первым.

    Цена ошибки — тихая потеря стилей: извлекли пустоту, тег удалили, страница поехала, и
    никакого исключения. На самом WebUI это не выстрелило (там CSS вынесен в отдельную
    колонку, страниц со `<style>` внутри `html` — 0 из 3 001 проверенной), но на любом
    источнике с инлайновыми стилями выстрелит.
    """
    return tag.string or "".join(tag.strings) or ""


# ── шаг 2: сборка self-contained страницы ─────────────────────────────────────

_DATA_URI_RE = re.compile(r"data:([\w.+/-]+)?;base64,([A-Za-z0-9+/=]+)", re.I)
# Внешним считаем всё, что не data:/#fragment: и http(s), и протокол-относительное `//cdn`,
# и корневое `/assets/x.png`, и относительное `./a.woff2` — файла рядом всё равно нет.
_EXTERNAL_LINK_RELS = {"stylesheet", "preload", "prefetch", "preconnect", "dns-prefetch",
                       "icon", "shortcut icon", "apple-touch-icon", "manifest", "modulepreload"}
# Теги, которые тянут сеть и/или рендерятся недетерминированно. iframe/video/embed/object
# занимают место в потоке — меняем на тот же серый плейсхолдер, чтобы не поехала вёрстка;
# остальное сносим.
_BOX_MEDIA = ("iframe", "video", "embed", "object", "canvas")
_DROP_TAGS = ("script", "noscript", "audio", "source", "track", "base", "meta")


def deblob_data_uris(text, max_len=DATA_URI_MAX):
    """Длинные base64 data-URI -> `data:,`. Короткие (мелкие inline-SVG иконки) оставляем:
    они дают реальную картинку почти бесплатно, а вот встроенный шрифт или фото на
    сотню килобайт — это чистый вес таргета."""
    def sub(m):
        return m.group(0) if len(m.group(2)) <= max_len else "data:,"
    return _DATA_URI_RE.sub(sub, text or "")


def _is_external(url):
    u = (url or "").strip()
    if not u or u.startswith(("#", "data:", "mailto:", "tel:", "javascript:", "about:")):
        return False
    return True


def sanitize_html(soup):
    """Снести всё сетевое и недетерминированное. Мутирует soup, возвращает счётчики."""
    stat = {"scripts": 0, "links": 0, "media_boxed": 0, "attrs": 0, "inline_style_urls": 0}

    for tag in soup.find_all(_DROP_TAGS):
        if tag.name == "meta":
            # meta нужен только charset/viewport; http-equiv refresh и og:image — мусор/сеть.
            if tag.get("charset") or (tag.get("name") or "").lower() == "viewport":
                continue
        if tag.name == "script":
            stat["scripts"] += 1
        tag.decompose()

    for link in soup.find_all("link"):
        rel = link.get("rel")
        rel = " ".join(rel).lower() if isinstance(rel, list) else (rel or "").lower()
        if rel in _EXTERNAL_LINK_RELS or _is_external(link.get("href")):
            stat["links"] += 1
            link.decompose()

    for tag in soup.find_all(_BOX_MEDIA):
        div = soup.new_tag("div")
        div["class"] = list(PLACEHOLDER_CLASSES)
        div["style"] = PLACEHOLDER_STYLE
        tag.replace_with(div)
        stat["media_boxed"] += 1

    # Атрибуты, ведущие наружу. `href` у `<a>` оставляем: клика при рендере нет, а текст
    # ссылки и её наличие — часть вёрстки, которую модель должна воспроизвести.
    for el in soup.find_all(True):
        for attr in ("src", "srcset", "poster", "data-src", "data-srcset", "background",
                     "xlink:href", "action", "formaction"):
            if attr in el.attrs and _is_external(el.get(attr)):
                del el[attr]
                stat["attrs"] += 1
        style = el.get("style")
        if style:
            new, n = cssprune.strip_external_urls(style)
            if n:
                el["style"] = new
                stat["inline_style_urls"] += n
    return stat


def assemble_page(html_text, css_text):
    """Шаг 2: одна self-contained страница из колонок `html` + `css`.

    Возвращает (soup, css, stat). CSS ещё НЕ шейкнут — это шаг 3; здесь он только
    очищен от внешних `url(...)` и слит с теми `<style>`, что остались в разметке.
    """
    html_text = deblob_data_uris(html_text)
    soup = make_soup(html_text)

    # Инлайновые <style> из разметки сливаем в общий стайлшит: у WebUI основная масса CSS
    # уже вынесена в колонку, но у части страниц в теле остаются свои блоки. Порядок важен —
    # разметочные идут ПОСЛЕ колоночных, как это было бы в документе.
    inline_css = []
    for st in soup.find_all("style"):
        inline_css.append(style_text(st))
        st.decompose()

    stat = sanitize_html(soup)
    css_all = "\n".join([deblob_data_uris(css_text or "")] + [deblob_data_uris(c) for c in inline_css])
    css_all, n_urls = cssprune.strip_external_urls(css_all)
    stat["css_urls"] = n_urls
    stat["css_len_raw"] = len(css_all)
    return soup, css_all, stat


def replace_images_with_placeholder(soup):
    """Шаг 4. `<img>` -> серый `<div>` ровно той строкой, что в бенче и в промпте генерации.
    Работает по soup (а не по тексту), потому что до рендера страница живёт как дерево."""
    n = 0
    for img in soup.find_all("img"):
        div = soup.new_tag("div")
        div["class"] = list(PLACEHOLDER_CLASSES)
        div["style"] = PLACEHOLDER_STYLE
        img.replace_with(div)
        n += 1
    return n


def build_html(soup, css_text):
    """Собрать финальный документ: стайлшит одним `<style>` в `<head>`."""
    doc = make_soup(str(soup))
    head = doc.find("head")
    if head is None:
        head = doc.new_tag("head")
        html_el = doc.find("html")
        if html_el is None:
            return f"<!doctype html><html><head><style>{css_text}</style></head><body>{soup}</body></html>"
        html_el.insert(0, head)
    if css_text.strip():
        style = doc.new_tag("style")
        style.string = css_text
        head.append(style)
    out = str(doc)
    if not out.lstrip().lower().startswith("<!doctype"):
        out = "<!doctype html>\n" + out
    return out


# ── шаг 3: tree-shaking через CDP ─────────────────────────────────────────────

_PW = {"pw": None, "browser": None}


def _browser():
    if _PW["browser"] is None:
        from playwright.sync_api import sync_playwright
        _PW["pw"] = sync_playwright().start()
        # Те же флаги, что в Evaluation/metrics_only/render.py: без DRM-устройств GPU-процесс
        # Chromium уходит в crash loop и утаскивает браузер целиком.
        _PW["browser"] = _PW["pw"].chromium.launch(args=[
            "--disable-gpu", "--disable-gpu-compositing",
            "--disable-software-rasterizer", "--disable-dev-shm-usage",
        ])
    return _PW["browser"]


def close_browser():
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


def used_css_ranges(html_text, width=RENDER_WIDTH, height=1024, timeout_ms=30000):
    """Диапазоны правил, которые Chromium реально применил при рендере на `width`.

    Возвращает (ranges, sheet_text). `sheet_text` — текст стайлшита ГЛАЗАМИ БРАУЗЕРА:
    пересобирать надо именно по нему, потому что оффсеты в `ruleUsage` указывают в него,
    а не в нашу строку (Chromium иначе трактует пробелы и комментарии перед селектором).

    Трекинг включается ДО загрузки контента — иначе первый разбор стилей происходит
    раньше подписки, и `ruleUsage` приходит пустым (страница выглядит как «ни одно
    правило не применилось», и вся вёрстка была бы съедена шейкером).
    """
    ctx = _browser().new_context(viewport={"width": width, "height": height},
                                 device_scale_factor=1)
    try:
        page = ctx.new_page()
        headers = []
        cdp = ctx.new_cdp_session(page)
        cdp.on("CSS.styleSheetAdded", lambda ev: headers.append(ev["header"]))
        cdp.send("DOM.enable")
        cdp.send("CSS.enable")
        cdp.send("CSS.startRuleUsageTracking")
        # wait_until="load", не "networkidle": страница после санитайза self-contained,
        # внешних запросов нет, а networkidle ждал бы 500 мс тишины на каждой странице.
        page.set_content(html_text, wait_until="load")
        usage = cdp.send("CSS.stopRuleUsageTracking").get("ruleUsage", [])
        ours = [h for h in headers if h.get("origin") == "regular" and not h.get("sourceURL")]
        if not ours:
            return [], ""
        sheet = max(ours, key=lambda h: h.get("length", 0))     # наш общий <style> — самый длинный
        sid = sheet["styleSheetId"]
        text = cdp.send("CSS.getStyleSheetText", {"styleSheetId": sid}).get("text", "")
        ranges = sorted((u["startOffset"], u["endOffset"])
                        for u in usage if u.get("styleSheetId") == sid and u.get("used"))
        return ranges, text
    finally:
        ctx.close()


def treeshake_css(html_no_style, css_text, soup_for_prefilter, width=RENDER_WIDTH,
                  keep_dead_fontface=False):
    """Шаг 3 целиком: префильтр -> браузер -> минимальный стайлшит.

    `html_no_style` — документ БЕЗ стайлшита (шаблон, куда подставляется CSS);
    `soup_for_prefilter` — его же дерево для soupsieve.
    Возвращает (css_min, stat).
    """
    pre_css, pre_stat = cssprune.prefilter(css_text, soup_for_prefilter)
    probe_html = build_html(soup_for_prefilter, pre_css)
    ranges, sheet_text = used_css_ranges(probe_html, width=width)
    if not sheet_text:
        # Стайлшит браузеру не достался (пустой CSS или страница не разобралась) —
        # честно отдаём префильтрованный вариант, а не пустоту.
        return pre_css, {**pre_stat, "cdp": "no-stylesheet", "css_len_pre": len(pre_css),
                         "css_len_min": len(pre_css)}
    css_min, shake_stat = cssprune.rebuild_from_usage(
        sheet_text, ranges, keep_dead_fontface=keep_dead_fontface)
    # Ступень 3b — по правилам шейкинг слеп внутри одного огромного `:root{}` с токенами
    # темы; переменные, которых никто не зовёт, режем отдельно (см. cssprune).
    css_min, var_stat = cssprune.prune_custom_props(css_min, extra_text=str(soup_for_prefilter))
    stat = {**pre_stat, **shake_stat, **var_stat, "cdp": "ok",
            "css_len_pre": len(pre_css), "css_len_min": len(css_min)}
    return css_min, stat


# ── шаги 5–6: ре-рендер и приёмка ────────────────────────────────────────────

def render_png(html_text, out_dir, stem):
    """Шаг 5. Скриншот снимается `Evaluation/metrics_only/render.py` — тем же рендерером,
    которым меряется бенч (ширина 1280 = дефолтный вьюпорт, высота = вся страница).

    ⚠ Через `render_many` (АСИНХРОННЫЙ движок), а не `render_html_to_png` (синхронный).
    Причина не косметическая: шаг 6 уже поднял в процессе фоновый asyncio-loop (metrics ->
    render_many -> _ensure_loop_thread), и следующий за ним синхронный Playwright падает с
    «Sync API inside the asyncio loop». `take_screenshot` это исключение ГЛОТАЕТ и молча
    подсовывает белую картинку 1280×960 — то есть в датасет уезжала бы пара
    «пустой белый скриншот -> нормальный HTML». Проверено: с sync-путём все три принятые
    страницы получили белые PNG. Тот же грабель описан в `Evaluation/metrics_only/RUNNING.md`.
    """
    import render as eval_render
    html_path = os.path.join(out_dir, f"{stem}.html")
    png_path = os.path.join(out_dir, f"{stem}.png")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_text)
    eval_render.render_many([{"html": html_path, "png": png_path, "overwrite": True}])
    return html_path, png_path


def accept_page(html_before, html_after, work_dir, stem="acc",
                pixel_threshold=DEFAULT_PIXEL_THRESHOLD, force_metric=False):
    """Шаг 6. Рендер ДО и ПОСЛЕ tree-shaking, сравнение нашей же метрикой Design2Code.

    ДО = страница с ПОЛНЫМ (только очищенным) стайлшитом, ПОСЛЕ = с минимальным.
    Если шейкер срезал лишнее — вёрстка поедет, и это видно тем же `final_score`,
    которым меряют модель. Порог задаётся снаружи (стартовый — 0.95).

    Возвращает dict метрик или {"error": ...}. Ошибку наверх не пробрасываем: одна
    непрожёванная страница не должна валить батч.
    """
    import metrics
    import render as eval_render
    os.makedirs(work_dir, exist_ok=True)
    ref_html = os.path.join(work_dir, f"{stem}_before.html")
    pred_html = os.path.join(work_dir, f"{stem}_after.html")
    with open(ref_html, "w", encoding="utf-8") as f:
        f.write(html_before)
    with open(pred_html, "w", encoding="utf-8") as f:
        f.write(html_after)

    out = {}
    ref_png = ref_html.replace(".html", ".png")
    pred_png = pred_html.replace(".html", ".png")
    eval_render.render_many([{"html": ref_html, "png": ref_png, "overwrite": True},
                             {"html": pred_html, "png": pred_png, "overwrite": True}])
    out["pixel_sim"] = pixel_similarity(ref_png, pred_png)

    # Короткое замыкание. Если два рендера совпали попиксельно, метрика Design2Code уже
    # ничего не решает: приёмка пройдена по любому из двух сигналов. А стоит она дорого —
    # это CLIP плюс ещё четыре рендера на страницу, и на замере она съедала ~80% времени
    # конвейера (0.46 стр/с на 4 воркерах = ~6 часов на корпус). Метрику всё равно считаем
    # там, где она РЕШАЕТ (пиксели разошлись), и на каждой `metric_every`-й странице —
    # чтобы в отчёте осталось её распределение, а не только пиксельное.
    if force_metric is False and (out["pixel_sim"] or 0) >= pixel_threshold:
        return out
    try:
        # use_ref_cache=False: эталон здесь свой у каждой страницы, кэш блоков только
        # пухнет на диске и ни разу не переиспользуется.
        out.update(metrics.score_pair(pred_html, ref_html, use_ref_cache=False))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def pixel_similarity(png_a, png_b):
    """Насколько два рендера совпадают попиксельно, [0..1].

    Прямое основание для приёмки: tree-shaking обязан оставить картинку прежней, и
    расхождение размера или пикселей — это ровно та поломка, которую шаг 6 ищет.
    Разная высота считается поломкой пропорционально: страница «схлопнулась» — значит
    шейкер срезал правило, державшее лейаут.
    """
    try:
        import numpy as np
        from PIL import Image
        a = Image.open(png_a).convert("RGB")
        b = Image.open(png_b).convert("RGB")
        if a.size != b.size:
            # Сравниваем по общей области, а расхождение габаритов штрафуем отдельно:
            # обрезать до пересечения и отрапортовать 1.0 было бы самообманом.
            w, h = min(a.width, b.width), min(a.height, b.height)
            if w == 0 or h == 0:
                return 0.0
            scale = (w * h) / max(a.width * a.height, b.width * b.height)
            a, b = a.crop((0, 0, w, h)), b.crop((0, 0, w, h))
        else:
            scale = 1.0
        na = np.asarray(a, dtype=np.int16)
        nb = np.asarray(b, dtype=np.int16)
        return float(scale * (1.0 - np.abs(na - nb).mean() / 255.0))
    except Exception:
        return None


# ── полный конвейер по одной странице ────────────────────────────────────────

_CSS_CLASS_RE = re.compile(r"\.(-?[_a-zA-Z]+[\w-]*)")


def styling_coverage(css_min, soup):
    """Доля классов разметки, у которых в итоговом CSS есть хоть одно правило.

    ЗАЧЕМ ИМЕННО ПОКРЫТИЕ КЛАССОВ, А НЕ «МАЛО ОБЪЯВЛЕНИЙ». У части источников WebUI колонка
    `css` НЕ содержит стайлшита компонентов. Проверено на `sap-fundamental-styles`: в 172 КБ
    лежат только служебные стили Storybook (`.sb-*`) и 58 блоков `:root` с токенами темы, а
    классов разметки (`fd-navigation`, `fd-navigation__container`) в CSS нет ВООБЩЕ —
    страница физически не восстанавливается из того, что дал источник. Скриншот в самом
    WebUI при этом стилизованный: он снят с живого сайта, где стайлшит был.

    Порог «мало объявлений» здесь не годится: он одинаково рубит и такую страницу, и честную
    простую вёрстку без CSS (туториалы w3schools) — а та вполне валидна, наш ре-рендер ей
    соответствует. Разделяет их именно покрытие: страница на 40 классов, из которых
    оформлен ноль, — это потерянный стайлшит; страница вообще без классов — просто простая.

    Шаг 6 (рендер до/после tree-shaking) такую страницу не ловит и не может: до и после она
    одинаково неоформленная, метрика честно даёт ~1.0. Поэтому нужен отдельный признак.
    """
    tree = cssprune.scan_css(css_min)
    total = cssprune.count_declarations(css_min, tree)
    custom = len(re.findall(r"(?<![\w-])--[\w-]+\s*:", css_min))

    used = set()
    for el in soup.find_all(True):
        c = el.get("class")
        if c:
            used.update(x.lower() for x in (c if isinstance(c, list) else [c]))
    in_css = {m.lower() for m in _CSS_CLASS_RE.findall(css_min)}
    # Инлайновый `style=` оформляет элемент без всякого класса — засчитываем его как
    # покрытие, иначе страница на инлайновых стилях выглядела бы «потерявшей стайлшит».
    inline_styled = sum(1 for el in soup.find_all(style=True))
    covered = used & in_css
    return {"decls_total": total, "decls_custom": custom,
            "decls_style": max(0, total - custom),
            "dom_nodes": len(soup.find_all(True)),
            "classes_used": len(used), "classes_covered": len(covered),
            "inline_styled": inline_styled,
            "class_coverage": (len(covered) / len(used)) if used else 1.0}


def convert_one(row, work_dir, accept_threshold=DEFAULT_ACCEPT_THRESHOLD,
                keep_dead_fontface=False, do_accept=True,
                min_class_coverage=0.10, unstyled_min_classes=10,
                pixel_threshold=DEFAULT_PIXEL_THRESHOLD, force_metric=False):
    """Все шаги 2–6 для одной строки WebUI.

    Возвращает dict: `status` in {ok, rejected, error}, плюс `target_html`, `png` (байты),
    `metrics`, `stat`. Исключение внутри становится status=error — страница выпадает из
    набора, но пул не падает (та же конвенция, что в двух других конвертерах).
    """
    out = {"sample_id": row.get("sample_id"), "status": "error", "stat": {}, "metrics": {}}
    try:
        soup, css_all, stat = assemble_page(row.get("html") or "", row.get("css") or "")
        stat["img_placeholders"] = replace_images_with_placeholder(soup)

        html_full = build_html(soup, css_all)                    # «ДО» для приёмки
        css_min, shake_stat = treeshake_css(html_full, css_all, soup,
                                            keep_dead_fontface=keep_dead_fontface)
        stat.update(shake_stat)
        html_min = build_html(soup, css_min)                     # «ПОСЛЕ» — кандидат в таргет

        cov = styling_coverage(css_min, soup)
        stat.update(cov)
        if (cov["classes_used"] >= unstyled_min_classes
                and cov["class_coverage"] < min_class_coverage
                and cov["inline_styled"] < unstyled_min_classes):
            out["status"] = "rejected"
            out["reason"] = (f"unstyled: оформлено {cov['classes_covered']}/{cov['classes_used']} "
                             f"классов — стайлшит компонентов в источнике отсутствует")
            out["stat"] = stat
            return out

        if do_accept:
            m = accept_page(html_full, html_min, work_dir, stem=str(out["sample_id"]),
                            pixel_threshold=pixel_threshold, force_metric=force_metric)
            out["metrics"] = m
            score, pix = m.get("final_score"), m.get("pixel_sim")
            if score is None and pix is None:
                out["status"] = "error"
                out["error"] = m.get("error", "метрика не посчиталась")
                return out
            # Метрика Design2Code на почти пустой странице (одна кнопка на белом фоне)
            # не находит ни одного блока и возвращает РОВНО 0 — это артефакт измерения,
            # а не сломанная конвертация. Замерено: 8 из 40 страниц смоука, все —
            # витрины отдельных компонентов. Поэтому решение принимается по двум сигналам:
            # метрика — основной, попиксельное сравнение тех же двух рендеров — запасной.
            # Реальную поломку шейкера пиксели ловят строже метрики (вёрстка поехала —
            # картинка поехала), так что запасной путь ничего не пропускает.
            ok_metric = score is not None and score >= accept_threshold
            ok_pixel = pix is not None and pix >= pixel_threshold
            if not (ok_metric or ok_pixel):
                out["status"] = "rejected"
                out["reason"] = (f"accept score={score if score is None else round(score,3)} "
                                 f"pixel={pix if pix is None else round(pix,4)} "
                                 f"(пороги {accept_threshold}/{pixel_threshold})")
                out["stat"] = stat
                return out
            out["accept_via"] = "metric" if ok_metric else "pixel"

        html_path, png_path = render_png(html_min, work_dir, f"{out['sample_id']}_final")
        with open(png_path, "rb") as f:
            png = f.read()
        out.update(status="ok", target_html=html_min, png=png, stat=stat)
        return out
    except Exception as e:
        import traceback
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
        return out
