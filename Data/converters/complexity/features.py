#!/usr/bin/env python3
"""features.py — признаки сложности страницы. Общий модуль для WebUI и WebCode2M.

ЗАЧЕМ ЭТО ОТДЕЛЬНО ОТ ДЛИНЫ. «Длинный таргет» и «сложная страница» коррелируют, но это
разные вещи, и наивный отбор по длине набирает не сложные страницы, а РАЗДУТЫЕ — ровно тот
мусор, на котором модель выучилась писать `<head>` на 25 тысяч символов вместо вёрстки
(разбор bimodal-коллапса, `docs/ROADMAP.md`). Поэтому ключевой признак здесь —
**ПЛОТНОСТЬ**: сколько видимых блоков приходится на токен кода. У честно сложной страницы
блоков много на единицу кода, у раздутой — мало.

ДВА УРОВНЯ, И ЭТО НЕ ЛЕНЬ, А АРИФМЕТИКА.
  • `static_features(html)` — без браузера, ~1 мс на страницу. Нужен, чтобы просеять
    кандидатов WebCode2M: их ~200 тысяч, и рендерить их все нельзя ни на каком железе.
  • `rendered_features(html_path)` — через Playwright, ~0.3–1 с на страницу. Даёт то, чего
    в исходнике нет в принципе: сколько узлов РЕАЛЬНО видно, в сколько колонок легла
    страница, какая доля блоков перекрывается. План требует считать сложность по
    ОТРЕНДЕРЕННОЙ странице — так и делаем, но только для шортлиста, прошедшего статику.

Обе функции возвращают dict с ОДИНАКОВЫМИ ключами там, где признак определён в обоих
режимах, плюс `render_ok` — чтобы отчёт всегда мог сказать, на чём именно посчитан отбор.
"""
import math
import os
import re
import sys

from bs4 import BeautifulSoup

_HERE = os.path.dirname(os.path.abspath(__file__))
_WEBUI = os.path.join(_HERE, "..", "webui")
if _WEBUI not in sys.path:
    sys.path.insert(0, _WEBUI)

import cssprune  # noqa: E402  — свой разбор CSS с оффсетами, см. webui/cssprune.py

# Признаки, по которым строится сводная сложность. Веса — не подгонка под метрику, а
# расстановка приоритетов: структура страницы (узлы/глубина/разнообразие тегов) весит
# больше, чем богатство отдельных элементов (таблицы/формы/svg).
FEATURE_WEIGHTS = {
    "nodes":          0.30,   # сколько элементов на странице
    "depth":          0.15,   # насколько глубоко вложена вёрстка
    "distinct_tags":  0.12,   # разнообразие тегов — плоский div-суп штрафуется
    "css_decls":      0.15,   # объём оформления
    "unique_classes": 0.08,
    "richness":       0.12,   # таблицы + поля форм + svg
    "columns":        0.08,   # многоколоночность = нетривиальный лейаут
}

_FORM_FIELDS = ("input", "select", "textarea", "button", "label", "option")


def _depth(soup):
    """Максимальная глубина дерева элементов (без текстовых узлов)."""
    best = 0
    for el in soup.find_all(True):
        d = 0
        p = el.parent
        while p is not None and getattr(p, "name", None):
            d += 1
            p = p.parent
        best = max(best, d)
    return best


def _collect_css(soup):
    """Весь CSS страницы: блоки `<style>` + инлайновые `style=`.

    ⚠ `.string`, а не `get_text()`: под html5lib содержимое `<style>` — узел `Stylesheet`,
    и bs4 не считает его текстом, поэтому `get_text()` молча отдаёт пустоту. Здесь разбор
    идёт через `html.parser`, где этого нет, но признак `css_decls` входит в сводную
    сложность — обнулись он от смены парсера, отбор поехал бы незаметно.
    """
    parts = [(st.string or "".join(st.strings) or "") for st in soup.find_all("style")]
    inline = [el.get("style") or "" for el in soup.find_all(style=True)]
    return "\n".join(parts), inline


def static_features(html_text, tokens_code=None):
    """Дешёвые признаки прямо из исходника. Без браузера."""
    soup = BeautifulSoup(html_text, "html.parser")
    els = soup.find_all(True)
    tags = [e.name.lower() for e in els]
    classes = set()
    for e in els:
        c = e.get("class")
        if c:
            classes.update(c if isinstance(c, list) else [c])

    style_css, inline_styles = _collect_css(soup)
    decls = cssprune.count_declarations(style_css)
    # Инлайновые `style=` — тоже объявления, и на реальных страницах их много.
    decls += sum(s.count(":") for s in inline_styles)

    tables = len(soup.find_all("table"))
    fields = sum(1 for t in tags if t in _FORM_FIELDS)
    svgs = len(soup.find_all("svg"))

    f = {
        "nodes": len(els),
        "depth": _depth(soup),
        "distinct_tags": len(set(tags)),
        "css_decls": decls,
        "unique_classes": len(classes),
        "tables": tables,
        "form_fields": fields,
        "svgs": svgs,
        "richness": tables * 3 + fields + svgs,
        "columns": 0,          # без рендера колонок не видно — заполнит rendered_features
        "overlap_frac": 0.0,
        "blocks_visible": 0,
        "render_ok": False,
    }
    f["text_len"] = len(soup.get_text(" ", strip=True))
    if tokens_code:
        f["tokens_code"] = tokens_code
        f["density"] = f["nodes"] / max(1, tokens_code)
    return f


# ── рендер-признаки ───────────────────────────────────────────────────────────
# Один проход по DOM в браузере: геометрия, видимость, колонки, перекрытия. Всё считается
# внутри страницы и возвращается одним объектом — гонять сотни вызовов page.evaluate на
# страницу было бы на порядок дороже самого рендера.
_PROBE_JS = r"""
() => {
  const MIN_AREA = 64;              // меньше 8x8 px — иконка/разделитель, не блок
  const OVERLAP_CAP = 600;          // перекрытия считаем попарно; без предела это O(n^2)
  const els = Array.from(document.querySelectorAll('*'));
  let nodes = 0, maxDepth = 0;
  const tags = new Set(), classes = new Set();
  const boxes = [];
  const pageW = Math.max(document.documentElement.scrollWidth, 1);

  for (const el of els) {
    const tag = el.tagName.toLowerCase();
    if (tag === 'script' || tag === 'style' || tag === 'head' || tag === 'meta'
        || tag === 'link' || tag === 'title') continue;
    nodes++;
    tags.add(tag);
    for (const c of el.classList) classes.add(c);
    let d = 0; let p = el.parentElement;
    while (p) { d++; p = p.parentElement; }
    if (d > maxDepth) maxDepth = d;

    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || parseFloat(cs.opacity) === 0) continue;
    const r = el.getBoundingClientRect();
    const area = r.width * r.height;
    if (area < MIN_AREA) continue;
    // Есть ли у элемента СВОЙ прямой текст — так отличаем блок с содержимым от обёртки.
    let ownText = false;
    for (const n of el.childNodes) {
      if (n.nodeType === 3 && n.textContent.trim().length > 0) { ownText = true; break; }
    }
    boxes.push({x: r.left + window.scrollX, y: r.top + window.scrollY,
                w: r.width, h: r.height, text: ownText,
                leaf: el.children.length === 0});
  }

  // Колонки: кластеризуем ЛЕВЫЕ КРАЯ блоков, которые не тянутся на всю ширину (те —
  // контейнеры, а не колонки). Кластер засчитывается, если в нём хотя бы 3 блока —
  // иначе за колонку сойдёт любая случайно смещённая пара.
  const cand = boxes.filter(b => b.w > 40 && b.w < 0.9 * pageW);
  const xs = cand.map(b => Math.round(b.x)).sort((a, b) => a - b);
  const clusters = [];
  for (const x of xs) {
    if (clusters.length && x - clusters[clusters.length - 1].last <= 24) {
      const c = clusters[clusters.length - 1]; c.n++; c.last = x;
    } else clusters.push({start: x, last: x, n: 1});
  }
  const columns = clusters.filter(c => c.n >= 3).length;

  // Перекрытия: доля ЛИСТОВЫХ блоков, пересекающихся с другим листовым. Обёртки не в
  // счёт — они перекрывают своих детей по определению, и это не признак плотности.
  const leaves = boxes.filter(b => b.leaf).slice(0, OVERLAP_CAP);
  let overlapped = 0;
  for (let i = 0; i < leaves.length; i++) {
    const a = leaves[i];
    for (let j = i + 1; j < leaves.length; j++) {
      const b = leaves[j];
      if (a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h) {
        overlapped++; break;
      }
    }
  }

  return {
    nodes, depth: maxDepth,
    distinct_tags: tags.size, unique_classes: classes.size,
    blocks_visible: boxes.length,
    blocks_text: boxes.filter(b => b.text).length,
    blocks_leaf: leaves.length,
    columns,
    overlap_frac: leaves.length ? overlapped / leaves.length : 0,
    page_w: pageW, page_h: document.documentElement.scrollHeight,
  };
}
"""

_PW = {"pw": None, "browser": None}


def _browser():
    if _PW["browser"] is None:
        from playwright.sync_api import sync_playwright
        _PW["pw"] = sync_playwright().start()
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


def rendered_features(html_text, tokens_code=None, width=1280, height=1024, timeout_ms=30000):
    """Признаки по ОТРЕНДЕРЕННОЙ странице. При сбое рендера падает обратно на статику,
    помечая это `render_ok=False` — страница не выпадает из отбора молча."""
    base = static_features(html_text, tokens_code=tokens_code)
    ctx = _browser().new_context(viewport={"width": width, "height": height},
                                 device_scale_factor=1)
    try:
        page = ctx.new_page()
        page.set_default_timeout(timeout_ms)
        page.set_content(html_text, wait_until="load")
        probe = page.evaluate(_PROBE_JS)
    except Exception as e:
        base["render_error"] = f"{type(e).__name__}: {e}"
        return base
    finally:
        ctx.close()

    base.update({k: probe[k] for k in
                 ("nodes", "depth", "distinct_tags", "unique_classes", "blocks_visible",
                  "blocks_text", "blocks_leaf", "columns", "overlap_frac", "page_w", "page_h")})
    base["render_ok"] = True
    base["richness"] = base["tables"] * 3 + base["form_fields"] + base["svgs"]
    if tokens_code:
        # ПЛОТНОСТЬ — видимых блоков на токен кода. Считается по рендеру, иначе теряет
        # смысл: в исходнике «блок» неотличим от обёртки, которая ничего не рисует.
        base["density"] = base["blocks_visible"] / max(1, tokens_code)
        base["tokens_code"] = tokens_code
    return base


# ── сводная сложность ────────────────────────────────────────────────────────

def percentile_ranks(values):
    """Ранг каждого значения в [0,1]. Ранги, а не сырые числа: признаки в разных единицах
    (узлы — сотни, колонки — единицы), и складывать их напрямую бессмысленно."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    n = len(values)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        r = (i + j) / 2.0 / max(1, n - 1)      # средний ранг для связок
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def composite_scores(feature_dicts, weights=None):
    """Сводная сложность = взвешенное среднее перцентильных рангов признаков.

    Считается ПО НАБОРУ (ранг внутри корпуса), поэтому число сравнимо только внутри
    одного прогона — что и нужно для отбора по перцентилям.
    """
    weights = weights or FEATURE_WEIGHTS
    cols = {}
    for name in weights:
        cols[name] = percentile_ranks([float(f.get(name) or 0) for f in feature_dicts])
    total_w = sum(weights.values())
    out = []
    for i in range(len(feature_dicts)):
        s = sum(weights[name] * cols[name][i] for name in weights) / total_w
        out.append(s)
    return out
