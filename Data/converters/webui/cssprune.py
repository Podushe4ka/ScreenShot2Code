#!/usr/bin/env python3
"""cssprune.py — разбор CSS в дерево правил с оффсетами + tree-shaking.

Зачем это вообще нужно. В WebUI стили лежат ОТДЕЛЬНОЙ колонкой `css`, и там не стили
страницы, а весь стайлшит сайта: у дизайн-систем (SAP Fundamental, Orbit, Carbon…) это
170–470 КБ на страницу в 100 DOM-узлов. Отсюда хвост длины кода до 172k токенов, из-за
которого WebUI не взяли в MVP (`Data/eda/datasets_overview.md`, §1). Обучать на таком
таргете нельзя и не нужно: 99% правил к этой странице отношения не имеют — ровно тот
случай, где модель учится писать гигантский `<head>` вместо вёрстки
(`docs/ROADMAP.md`, разбор bimodal-коллапса).

Три ступени, от дешёвой к дорогой:
  1. `strip_external_urls` — `url(...)` наружу превращается в `none` (страница обязана быть
     self-contained и рендериться оффлайн; `data:` и `url(#fragment)` остаются);
  2. `prefilter` — матчинг селекторов по DOM БЕЗ браузера: сначала копеечная проверка по
     множествам тегов/классов/id страницы, потом soupsieve на выживших. Снимает основную
     массу (у дизайн-систем это 95–99% правил), чтобы браузеру досталось меньше;
  3. `rebuild_from_usage` — правила, которые Chromium через CDP
     (`CSS.startRuleUsageTracking`) реально применил при рендере на 1280.

Ступень 2 обязана быть КОНСЕРВАТИВНОЙ: она не имеет права выбросить то, что оставила бы
ступень 3. Поэтому любой селектор, который не удалось разобрать, остаётся.

Модуль не тянет ни playwright, ни datasets — только bs4/soupsieve. Сам CDP-вызов живёт в
`convert_lib.py`, сюда приходит уже готовый список использованных диапазонов.
"""
import re

import soupsieve

# ── at-правила по способу разбора ──────────────────────────────────────────────
# внутри лежат ВЛОЖЕННЫЕ правила — рекурсируем и шейкаем внутренность
NESTED_AT = {"media", "supports", "layer", "container", "scope", "document",
             "-webkit-media", "-moz-document"}
# внутри лежат кадры анимации — блок атомарен, режется целиком по имени
KEYFRAMES_AT = {"keyframes", "-webkit-keyframes", "-moz-keyframes", "-o-keyframes"}
# внутри лежат объявления — блок атомарен (@font-face, @page, @property, @counter-style…)


class Rule:
    """Узел CSS с абсолютными оффсетами в исходном тексте.

    Оффсеты — то, чем этот разбор отличается от tinycss2 и прочих парсеров: диапазоны
    из CDP (`ruleUsage.startOffset/endOffset`) указывают в текст стайлшита, и без своих
    оффсетов их не с чем сопоставить.
    """
    __slots__ = ("kind", "start", "end", "prelude", "body_start", "body_end", "children")

    def __init__(self, kind, start, end, prelude, body_start=None, body_end=None, children=()):
        self.kind = kind                  # style | at_nested | at_keyframes | at_decl | at_stmt
        self.start, self.end = start, end
        self.prelude = prelude
        self.body_start, self.body_end = body_start, body_end
        self.children = list(children)

    def __repr__(self):
        return f"<{self.kind} {self.prelude[:40]!r} [{self.start}:{self.end}]>"


# ── низкоуровневый сканер: строки, комментарии, скобки ────────────────────────

def _skip_string(t, i):
    q = t[i]; i += 1; n = len(t)
    while i < n:
        if t[i] == "\\":
            i += 2; continue
        if t[i] == q:
            return i + 1
        i += 1
    return n


def _skip_ws_comments(t, i, end):
    while i < end:
        if t[i].isspace():
            i += 1
        elif t.startswith("/*", i):
            j = t.find("*/", i + 2)
            i = end if j < 0 else min(j + 2, end)
        else:
            break
    return i


def _scan_prelude(t, i, end):
    """До неэкранированного `{` или `;` на нулевой глубине скобок. -> (индекс, терминатор)."""
    depth = 0
    while i < end:
        c = t[i]
        if c in "\"'":
            i = _skip_string(t, i); continue
        if t.startswith("/*", i):
            j = t.find("*/", i + 2)
            i = end if j < 0 else j + 2
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and c == "{":
            return i, "{"
        elif depth == 0 and c == ";":
            return i, ";"
        i += 1
    return end, None


def _match_brace(t, i, end):
    """t[i] == '{' -> индекс СРАЗУ ПОСЛЕ парной '}'. Незакрытый блок тянется до end."""
    depth = 0
    while i < end:
        c = t[i]
        if c in "\"'":
            i = _skip_string(t, i); continue
        if t.startswith("/*", i):
            j = t.find("*/", i + 2)
            i = end if j < 0 else j + 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return end


_AT_NAME_RE = re.compile(r"@([\w-]+)")


def scan_css(text, start=0, end=None):
    """Плоский разбор стайлшита в дерево `Rule` с абсолютными оффсетами."""
    end = len(text) if end is None else end
    nodes, i = [], start
    while True:
        i = _skip_ws_comments(text, i, end)
        if i >= end:
            break
        node_start = i
        pe, term = _scan_prelude(text, i, end)
        if term is None:
            break                                     # обрезанный хвост — молча бросаем
        prelude = text[node_start:pe].strip()
        if term == ";":
            if prelude:
                nodes.append(Rule("at_stmt", node_start, pe + 1, prelude))
            i = pe + 1
            continue
        after = _match_brace(text, pe, end)
        body_start, body_end = pe + 1, max(pe + 1, after - 1)
        if prelude.startswith("@"):
            m = _AT_NAME_RE.match(prelude)
            name = (m.group(1).lower() if m else "")
            if name in NESTED_AT:
                kind, children = "at_nested", scan_css(text, body_start, body_end)
            elif name in KEYFRAMES_AT:
                kind, children = "at_keyframes", []
            else:
                kind, children = "at_decl", []
        else:
            kind, children = ("style", []) if prelude else ("at_decl", [])
        nodes.append(Rule(kind, node_start, after, prelude, body_start, body_end, children))
        i = after
    return nodes


# ── очистка от внешних ресурсов ───────────────────────────────────────────────
# `data:` и `url(#id)` (ссылка на inline-SVG фильтр/градиент) остаются: они не ходят в сеть.
_URL_RE = re.compile(r"url\(\s*(['\"]?)([^)'\"]*)\1\s*\)", re.I)


def strip_external_urls(css):
    """Внешние `url(...)` -> `none`. Возвращает (css, сколько_срезано)."""
    n = 0

    def sub(m):
        nonlocal n
        target = m.group(2).strip()
        if target.lower().startswith("data:") or target.startswith("#"):
            return m.group(0)
        n += 1
        return "none"

    return _URL_RE.sub(sub, css or ""), n


# ── ступень 2: префильтр по DOM без браузера ──────────────────────────────────

def dom_index(soup):
    """Множества тегов / классов / id, реально присутствующих в документе."""
    tags, classes, ids = set(), set(), set()
    for el in soup.find_all(True):
        tags.add(el.name.lower())
        cls = el.get("class")
        if cls:
            classes.update(c.lower() for c in (cls if isinstance(cls, list) else [cls]))
        eid = el.get("id")
        if eid:
            ids.add(str(eid).lower())
    return {"tags": tags, "classes": classes, "ids": ids}


def split_selector_list(sel):
    """Разбить список селекторов по запятым ВЕРХНЕГО уровня (внутри :is(...)/[..] — не режем)."""
    out, depth, buf, i, n = [], 0, [], 0, len(sel)
    while i < n:
        c = sel[i]
        if c in "\"'":
            j = _skip_string(sel, i); buf.append(sel[i:j]); i = j; continue
        if c in "([":
            depth += 1
        elif c in ")]":
            depth = max(0, depth - 1)
        elif c == "," and depth == 0:
            out.append("".join(buf).strip()); buf = []; i += 1; continue
        buf.append(c); i += 1
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


# Правый компаунд селектора — то, чем правило «цепляется» за элемент. Если нужного ему
# класса/id/тега в документе нет, правило не сматчится ни при каком раскладе. Комбинаторы
# внутри :is()/:not()/:has() в расчёт не берём — этот разбор нарочно грубый и односторонний
# (может оставить лишнее, но не выбросить нужное).
_COMBINATOR_RE = re.compile(r"[\s>+~]+(?![^(\[]*[)\]])")
_CLASS_RE = re.compile(r"\.(-?[_a-zA-Z]+[\w-]*)")
_ID_RE = re.compile(r"#(-?[_a-zA-Z]+[\w-]*)")
_TAG_RE = re.compile(r"^([a-zA-Z][\w-]*)")
# Псевдоклассы состояния/структуры, которые soupsieve либо не знает, либо трактует иначе
# браузера. Для ГРУБОГО отбора их просто снимаем — селектор становится ШИРЕ, а значит
# префильтр остаётся консервативным.
_PSEUDO_RE = re.compile(r"::?[\w-]+(\([^()]*\))?")


def _rightmost_compound(sel):
    parts = _COMBINATOR_RE.split(sel.strip())
    parts = [p for p in parts if p]
    return parts[-1] if parts else sel.strip()


def selector_plausible(sel, idx):
    """Грубая проверка «может ли этот селектор сматчиться в этом документе»."""
    comp = _rightmost_compound(sel)
    if not comp or comp.startswith("@"):
        return True
    bare = _PSEUDO_RE.sub("", comp)
    if not bare.strip():
        return True                        # чистая псевдо-конструкция (`::selection`, `:root`)
    for cid in _ID_RE.findall(bare):
        if cid.lower() not in idx["ids"]:
            return False
    for cls in _CLASS_RE.findall(bare):
        if cls.lower() not in idx["classes"]:
            return False
    m = _TAG_RE.match(bare)
    if m:
        tag = m.group(1).lower()
        if tag not in ("html", "body", "*") and tag not in idx["tags"]:
            return False
    return True


def selector_matches_dom(sel, soup):
    """Точная проверка через soupsieve. Неразобранный селектор -> True (не выбрасываем)."""
    probe = _PSEUDO_RE.sub("", sel).strip()
    # `.btn:hover` -> `.btn`: смотрим, есть ли вообще носитель стиля. Само :hover при
    # статическом рендере не сработает и правило отсеет уже CDP на ступени 3.
    probe = re.sub(r"^\s*[>+~]\s*", "", probe)
    if not probe or probe in ("*", "html", "body", ":root"):
        return True
    try:
        return soupsieve.select_one(probe, soup) is not None
    except Exception:
        return True


def _keep_style_rule(node, text, soup, idx):
    for sel in split_selector_list(node.prelude):
        if not selector_plausible(sel, idx):
            continue
        if selector_matches_dom(sel, soup):
            return True
    return False


def prefilter(css_text, soup, tree=None):
    """Ступень 2. Возвращает (css, статистика). Собирает НОВЫЙ текст стайлшита —
    именно он потом уезжает в браузер, и именно в него будут указывать оффсеты CDP."""
    tree = scan_css(css_text) if tree is None else tree
    idx = dom_index(soup)
    # Ключи namespace'ятся префиксом `pre_`: статистику ступени 2 и ступени 3 сливают в
    # один dict, а у обеих есть «сколько правил на входе». Без префикса вторая молча
    # затирала первую, и в отчёте ступень 2 выглядела как «ничего не отфильтровала».
    stat = {"pre_rules_in": 0, "pre_rules_out": 0}

    def walk(nodes):
        out = []
        for nd in nodes:
            if nd.kind == "style":
                stat["pre_rules_in"] += 1
                if _keep_style_rule(nd, css_text, soup, idx):
                    stat["pre_rules_out"] += 1
                    out.append(css_text[nd.start:nd.end])
            elif nd.kind == "at_nested":
                inner = walk(nd.children)
                if inner:
                    out.append(f"{nd.prelude}{{{''.join(inner)}}}")
            elif nd.kind == "at_stmt":
                if not nd.prelude.lower().startswith("@import"):   # @import = сетевой запрос
                    out.append(css_text[nd.start:nd.end])
            else:                                                  # at_decl / at_keyframes
                out.append(css_text[nd.start:nd.end])
        return out

    return "\n".join(walk(tree)), stat


# ── ступень 3: пересборка по факту применения (CDP) ───────────────────────────

def _covered(node, used):
    """Правило считаем применённым, если хоть один использованный диапазон его накрывает.

    Сравнение по ПЕРЕСЕЧЕНИЮ, а не по равенству оффсетов: CDP отдаёт диапазон правила так,
    как его разобрал сам Chromium (иначе трактует комментарии и пробелы перед селектором),
    и требовать посимвольного совпадения границ значило бы терять живые правила.
    """
    for s, e in used:
        if s < node.end and e > node.start:
            return True
    return False


_ANIM_NAME_RE = re.compile(r"@(?:-\w+-)?keyframes\s+(['\"]?)([\w-]+)\1", re.I)
_FONT_SRC_RE = re.compile(r"\bsrc\s*:", re.I)


def rebuild_from_usage(css_text, used_ranges, tree=None, keep_dead_fontface=False,
                       compact=True):
    """Ступень 3: минимальный стайлшит из правил, которые Chromium применил при рендере.

    Что сохраняется ПОМИМО применённых style-правил (CDP про них usage не отдаёт вовсе):
      • `@font-face` — но только с живым `src` (после `strip_external_urls` внешние шрифты
        стали `none`; блок без источника — мёртвый код в таргете, модель училась бы писать
        `@font-face`, который ничего не грузит. Вернуть их можно `keep_dead_fontface=True`);
      • `@keyframes` — только те, чьё имя реально упомянуто в оставшихся объявлениях;
      • `@media` / `@supports` — обёртка сохраняется, если внутри уцелело хоть одно правило.
        Медиазапрос, не попавший во вьюпорт 1280, внутри пуст (CDP не метит его правила
        применёнными) и выбрасывается вместе с обёрткой — на скриншоте его и не видно.
      • `@property`, `@counter-style`, `@page` и прочие at_decl — сохраняются как есть.
    """
    tree = scan_css(css_text) if tree is None else tree
    stat = {"shake_rules_in": 0, "shake_rules_kept": 0, "keyframes_in": 0, "keyframes_kept": 0,
            "fontface_in": 0, "fontface_kept": 0}
    deferred_keyframes = []

    def walk(nodes):
        out = []
        for nd in nodes:
            if nd.kind == "style":
                stat["shake_rules_in"] += 1
                if _covered(nd, used_ranges):
                    stat["shake_rules_kept"] += 1
                    out.append(_emit(css_text, nd, compact))
            elif nd.kind == "at_nested":
                inner = walk(nd.children)
                if inner:
                    out.append(f"{_compact_text(nd.prelude, compact)}{{{''.join(inner)}}}")
            elif nd.kind == "at_keyframes":
                stat["keyframes_in"] += 1
                deferred_keyframes.append(nd)      # решение — после сборки остального
            elif nd.kind == "at_decl":
                low = nd.prelude.lower()
                if low.startswith("@font-face"):
                    stat["fontface_in"] += 1
                    body = css_text[nd.body_start:nd.body_end]
                    alive = keep_dead_fontface or (
                        _FONT_SRC_RE.search(body) and "none" not in _srcs(body))
                    if not alive:
                        continue
                    stat["fontface_kept"] += 1
                out.append(_emit(css_text, nd, compact))
            elif nd.kind == "at_stmt":
                if not nd.prelude.lower().startswith("@import"):
                    out.append(_compact_text(nd.prelude, compact) + ";")
        return out

    body_parts = walk(tree)
    kept_css = "".join(body_parts)
    for nd in deferred_keyframes:
        m = _ANIM_NAME_RE.match(nd.prelude)
        name = m.group(2) if m else None
        if name and re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", kept_css):
            stat["keyframes_kept"] += 1
            body_parts.append(_emit(css_text, nd, compact))
    return "".join(body_parts), stat


def _srcs(body):
    """Значения всех `src:` в блоке @font-face — чтобы отличить живой шрифт от выпотрошенного."""
    return " ".join(re.findall(r"\bsrc\s*:([^;}]*)", body, re.I))


_WS_RUN_RE = re.compile(r"\s+")


def _compact_text(s, compact):
    if not compact:
        return s
    out, i, n = [], 0, len(s)
    while i < n:                                   # сжимаем пробелы, НЕ трогая строки
        c = s[i]
        if c in "\"'":
            j = _skip_string(s, i); out.append(s[i:j]); i = j; continue
        if s.startswith("/*", i):
            j = s.find("*/", i + 2)
            i = n if j < 0 else j + 2
            out.append(" ")
            continue
        out.append(c); i += 1
    return _WS_RUN_RE.sub(" ", "".join(out)).strip()


def _emit(text, node, compact):
    if node.body_start is None:
        return text[node.start:node.end]
    prelude = _compact_text(node.prelude, compact)
    body = _compact_text(text[node.body_start:node.body_end], compact)
    if not body.strip():
        return ""
    return f"{prelude}{{{body}}}"


# ── ступень 3b: неиспользуемые CSS-переменные ────────────────────────────────
# Ступень 3 режет ПРАВИЛАМИ, и на одном правиле она слепа. У дизайн-систем (SAP Fundamental,
# Carbon) весь токен-набор темы лежит в ОДНОМ `:root{}` на тысячи объявлений: правило
# матчится на html, CDP честно метит его применённым, и 146 КБ переменных едут в таргет
# при 58 живых правилах. Замерено: sap-fundamental-styles 172 КБ -> 146 КБ после ступени 3,
# то есть шейкинг там не сработал вообще. Ниже — та же логика «выброси неиспользуемое»,
# но на уровне объявления.
_VAR_REF_RE = re.compile(r"var\(\s*(--[\w-]+)")
_CUSTOM_DECL_RE = re.compile(r"^\s*(--[\w-]+)\s*:", re.S)


def prune_custom_props(css_text, extra_text="", tree=None):
    """Выбросить `--переменные`, на которые никто не ссылается (транзитивно).

    `extra_text` — разметка страницы: переменную могут звать из инлайнового `style=`,
    и без этого она бы считалась мёртвой. Скриптов на странице к этому моменту нет,
    так что других способов дотянуться до переменной не остаётся.
    """
    tree = scan_css(css_text) if tree is None else tree
    custom = {}                                    # имя -> список значений (могут переопределяться)
    referenced = set(_VAR_REF_RE.findall(extra_text or ""))

    def collect(nodes):
        for nd in nodes:
            if nd.children:
                collect(nd.children); continue
            if nd.body_start is None:
                continue
            for decl in _split_decls(css_text[nd.body_start:nd.body_end]):
                m = _CUSTOM_DECL_RE.match(decl)
                if m:
                    custom.setdefault(m.group(1), []).append(decl)
                else:
                    referenced.update(_VAR_REF_RE.findall(decl))

    collect(tree)

    # Замыкание: переменная может ссылаться на переменную (`--btn-bg: var(--brand)`).
    frontier = set(referenced)
    while frontier:
        nxt = set()
        for name in frontier:
            for decl in custom.get(name, ()):
                for ref in _VAR_REF_RE.findall(decl):
                    if ref not in referenced:
                        referenced.add(ref); nxt.add(ref)
        frontier = nxt

    stat = {"vars_in": len(custom),
            "vars_kept": sum(1 for k in custom if k in referenced)}

    def rewrite(nodes):
        out = []
        for nd in nodes:
            if nd.children:
                inner = rewrite(nd.children)
                if inner:
                    out.append(f"{nd.prelude}{{{''.join(inner)}}}")
                continue
            if nd.body_start is None:
                out.append(css_text[nd.start:nd.end]); continue
            if nd.kind == "at_keyframes":
                out.append(css_text[nd.start:nd.end]); continue
            kept = []
            for decl in _split_decls(css_text[nd.body_start:nd.body_end]):
                m = _CUSTOM_DECL_RE.match(decl)
                if m and m.group(1) not in referenced:
                    continue
                if decl.strip():
                    kept.append(decl.strip())
            if kept:
                out.append(f"{nd.prelude}{{{';'.join(kept)}}}")
        return out

    return "".join(rewrite(tree)), stat


def count_declarations(css_text, tree=None):
    """Число CSS-объявлений — признак сложности (см. `Data/converters/complexity/`)."""
    tree = scan_css(css_text) if tree is None else tree
    total = 0

    def walk(nodes):
        nonlocal total
        for nd in nodes:
            if nd.children:
                walk(nd.children)
            elif nd.body_start is not None:
                body = css_text[nd.body_start:nd.body_end]
                total += sum(1 for part in _split_decls(body) if ":" in part)

    walk(tree)
    return total


def _split_decls(body):
    out, buf, i, n, depth = [], [], 0, len(body), 0
    while i < n:
        c = body[i]
        if c in "\"'":
            j = _skip_string(body, i); buf.append(body[i:j]); i = j; continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth = max(0, depth - 1)
        elif c == ";" and depth == 0:
            out.append("".join(buf)); buf = []; i += 1; continue
        buf.append(c); i += 1
    if "".join(buf).strip():
        out.append("".join(buf))
    return out
