"""Строит сэмплы задачи `editing` из чистых страниц.

Схема (из UI2Code^N §3.2.2): покрываем четыре операции — add / delete / replace / adjust.
Трудная операция «добавить компонент» получается РАЗВОРОТОМ пары удаления: вырезав блок,
мы бесплатно получаем два сэмпла — «удали этот блок» (полная -> урезанная) и «добавь такой
блок» (урезанная -> полная). Инструкция при этом точна по построению, потому что описывает
ровно ту правку, которую применил скрипт, а не то, что почудилось разметчику.

Семейства правок заземлены на darknoon/tailwind-edits (509 реальных inline-правок).
Замер по корпусу: 61% правок — цвет фона, 21% — цвет текста, 13% — насыщенность шрифта,
структурных правок там нет вовсе. Отсюда деление:

* `adjust` / `replace` — текстовая подмена классов. Работает и для static_inline
  (`class="…"`), и для react_cdn (`className="…"` внутри JSX), потому что идёт по тексту
  исходника, а не по разобранному дереву.
* `delete` / `add` — структурные, только для static_inline: вырезать JSX-поддерево из
  строки внутри `<script>` надёжно нельзя, а притворяться, что можно, — значит класть
  в датасет битые таргеты.

Использование:
    .venv/bin/python Data/converters/synth/mutate.py --limit 2
"""

import argparse
import json
import random
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "Evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import render as ev_render  # noqa: E402

SEED = 42

# Порог намеренно НИЗКИЙ, и это не небрежность, а разница природы двух задач.
# В polishing модель сама СРАВНИВАЕТ эталонный скриншот с текущим рендером — там правка
# обязана быть хорошо заметной, иначе сравнивать нечего (в degrade.py порог 0.06).
# В editing модели СКАЗАНО инструкцией, что менять; картинка лишь контекст. Поэтому
# здесь достаточно убедиться, что правка вообще не пустая: рендер изменился хоть
# сколько-нибудь. Требовать заметности значило бы выбрасывать законные точечные правки
# вроде перекраски бейджей — на плотной странице они дают доли процента площади.
MIN_VISUAL_DIFF = 0.0005

# Палитры Tailwind, между которыми осмысленно переключать. Оттенки берём из тех же
# семейств, что встречаются в корпусе правок.
_HUES = ["slate", "gray", "zinc", "red", "orange", "amber", "yellow", "lime",
         "green", "emerald", "teal", "cyan", "sky", "blue", "indigo", "violet",
         "purple", "fuchsia", "pink", "rose"]
_WEIGHTS = ["thin", "light", "normal", "medium", "semibold", "bold", "extrabold"]

_BG_RE = re.compile(r"\bbg-(" + "|".join(_HUES) + r")-(\d{2,3})\b")
_TEXT_RE = re.compile(r"\btext-(" + "|".join(_HUES) + r")-(\d{2,3})\b")
_FONT_RE = re.compile(r"\bfont-(" + "|".join(_WEIGHTS) + r")\b")

# Плейсхолдер картинки трогать НЕЛЬЗЯ по двум причинам сразу: он обязан совпадать
# побайтово с тем, что подставляет replace_images_with_placeholder, и его серый цвет
# всё равно прибит инлайновым style, так что перекраска класса bg-gray-300 внутри него
# не меняет ни пикселя. Вырезаем его из текста перед поиском и возвращаем на место после.
_PLACEHOLDER_RE = re.compile(
    r'<div class="bg-gray-300 w-full h-48 rounded" style="background-color:#d1d5db;'
    r'width:100%;height:12rem;border-radius:0\.5rem;display:block;"></div>')
_PH_TOKEN = "\x00PLACEHOLDER\x00"


def _mask_placeholders(html: str) -> tuple[str, int]:
    return _PLACEHOLDER_RE.sub(_PH_TOKEN, html), len(_PLACEHOLDER_RE.findall(html))


def _unmask_placeholders(html: str) -> str:
    ph = ('<div class="bg-gray-300 w-full h-48 rounded" style="background-color:#d1d5db;'
          'width:100%;height:12rem;border-radius:0.5rem;display:block;"></div>')
    return html.replace(_PH_TOKEN, ph)

# Человекочитаемые имена оттенков для инструкции.
_HUE_EN = {h: h for h in _HUES}
_HUE_EN.update({"slate": "slate gray", "zinc": "zinc gray", "sky": "sky blue"})

# Что считаем «компонентом», который можно вырезать целиком.
_BLOCK_TAGS = ("section", "article", "aside", "nav", "table", "form", "figure", "ul", "ol")
_MIN_SUB, _MAX_SUB = 4, 120


def _visible_label(tag, limit: int = 60) -> str:
    """Короткая подпись блока для инструкции — по видимому тексту, как сказал бы человек."""
    for h in ("h1", "h2", "h3", "h4", "caption", "legend", "th", "summary"):
        node = tag.find(h)
        if node and node.get_text(strip=True):
            return node.get_text(strip=True)[:limit]
    text = tag.get_text(" ", strip=True)
    return text[:limit] if text else tag.name


# --------------------------------------------------------------- adjust / replace


# Инструкция НЕ хранится в готовом виде: она собирается из meta под нужное направление.
# Иначе обратное направление легко получает текст прямого — то есть указание сделать ровно
# противоположное тому, что требуется, а таргет при этом валиден и ошибку ничем не поймать.
_TEMPLATES = {
    "bg_color": ("Change every element with a {a} background to {b} instead, keeping the "
                 "same shade level. Leave all other colors, text and layout untouched."),
    "text_color": ("Recolor the text that is currently {a} to {b}, keeping the same shade "
                   "level. Do not change backgrounds or layout."),
    "font_weight": ("Make the text that currently uses the {a} font weight {b} instead. "
                    "Keep colors, spacing and structure exactly as they are."),
}


def render_instruction(meta: dict, direction: str) -> str:
    """Текст инструкции для направления `apply` (эталон -> изменённая) или
    `restore` (изменённая -> эталон). Во втором случае «откуда» и «куда» меняются местами."""
    a, b = meta["a_human"], meta["b_human"]
    if direction == "restore":
        a, b = b, a
    return _TEMPLATES[meta["family"]].format(a=a, b=b)


def op_recolor_bg(html: str, rng: random.Random):
    hits = list(_BG_RE.finditer(html))
    if not hits:
        return None
    m = rng.choice(hits)
    old_hue, shade = m.group(1), m.group(2)
    new_hue = rng.choice([h for h in _HUES if h != old_hue])
    out = html.replace(f"bg-{old_hue}-{shade}", f"bg-{new_hue}-{shade}")
    return out, {"op": "replace", "family": "bg_color",
                 "from": f"bg-{old_hue}-{shade}", "to": f"bg-{new_hue}-{shade}",
                 "a_human": f"{_HUE_EN[old_hue]} {shade}",
                 "b_human": f"{_HUE_EN[new_hue]} {shade}",
                 "occurrences": len(_BG_RE.findall(html))}


def op_recolor_text(html: str, rng: random.Random):
    hits = list(_TEXT_RE.finditer(html))
    if not hits:
        return None
    m = rng.choice(hits)
    old_hue, shade = m.group(1), m.group(2)
    new_hue = rng.choice([h for h in _HUES if h != old_hue])
    out = html.replace(f"text-{old_hue}-{shade}", f"text-{new_hue}-{shade}")
    return out, {"op": "replace", "family": "text_color",
                 "from": f"text-{old_hue}-{shade}", "to": f"text-{new_hue}-{shade}",
                 "a_human": f"{_HUE_EN[old_hue]} {shade}",
                 "b_human": f"{_HUE_EN[new_hue]} {shade}",
                 "occurrences": len(_TEXT_RE.findall(html))}


def op_font_weight(html: str, rng: random.Random):
    hits = list(_FONT_RE.finditer(html))
    if not hits:
        return None
    old = rng.choice(hits).group(1)
    idx = _WEIGHTS.index(old)
    choices = [w for i, w in enumerate(_WEIGHTS) if abs(i - idx) >= 2]
    if not choices:
        return None
    new = rng.choice(choices)
    out = html.replace(f"font-{old}", f"font-{new}")
    return out, {"op": "adjust", "family": "font_weight",
                 "from": f"font-{old}", "to": f"font-{new}",
                 "a_human": old, "b_human": new,
                 "occurrences": len(_FONT_RE.findall(html))}


# --------------------------------------------------------------- delete / add

def op_delete_block(html: str, rng: random.Random):
    """Вырезает целый блок. Возвращает ДВА сэмпла — прямой и обратный.

    Обратное направление и есть трюк UI2Code^N: «добавь такой-то блок» — операция,
    которую иначе пришлось бы размечать вручную, здесь получается даром и с точной
    инструкцией, потому что мы знаем, что именно удалили.
    """
    soup = ev_render._make_soup(html)
    body = soup.body
    if body is None:
        return None
    cands = [t for t in body.find_all(_BLOCK_TAGS)
             if _MIN_SUB <= len(t.find_all(True)) <= _MAX_SUB
             and t.get_text(strip=True)]
    if not cands:
        return None
    victim = rng.choice(cands)
    label = _visible_label(victim)
    kind = victim.name
    victim.extract()
    reduced = str(soup)
    if len(reduced) < len(html) * 0.4:      # вырезали половину страницы — не правка, а разгром
        return None
    return reduced, label, kind


# --------------------------------------------------------------------- сборка

STYLE_OPS = [op_recolor_bg, op_recolor_text, op_font_weight]


def build_for_page(pid: str, raw: str, impl: str, rng: random.Random) -> list[dict]:
    """Список сэмплов editing для одной страницы."""
    out = []

    # Правки классов — ТОЛЬКО для react_cdn. В static_inline классы вида bg-*-* мертвы
    # (Tailwind не подключён), и единственное место, где они там встречаются, — это
    # плейсхолдер картинки. Перекраска такого класса не меняет на рендере ни пикселя,
    # то есть сэмпл учил бы «выполни инструкцию, правильный ответ — та же картинка».
    # Ровно такой брак и вышел на калибровке: p0004 (static) получил замену
    # bg-gray-300 -> bg-slate-300 в 8 плейсхолдерах при нулевом визуальном эффекте.
    masked, n_ph = _mask_placeholders(raw)
    for op in (STYLE_OPS if impl == "react_cdn" else []):
        res = op(masked, rng)
        if res is None:
            continue
        mutated, meta = res
        if mutated == masked:
            continue
        mutated = _unmask_placeholders(mutated)
        meta["placeholders_protected"] = n_ph
        # Прямое: из эталона сделать изменённую версию.
        out.append({"task_type": "editing", "page_id": pid, "impl": impl,
                    "current_html": raw, "target_html": mutated,
                    "instruction": render_instruction(meta, "apply"),
                    "meta": meta | {"direction": "apply"}})
        # Обратное: привести изменённую версию обратно к эталону.
        out.append({"task_type": "editing", "page_id": pid, "impl": impl,
                    "current_html": mutated, "target_html": raw,
                    "instruction": render_instruction(meta, "restore"),
                    "meta": meta | {"direction": "restore"}})

    # Структурные — только static_inline, см. докстринг модуля.
    if impl == "static_inline":
        res = op_delete_block(raw, rng)
        if res is not None:
            reduced, label, kind = res
            out.append({"task_type": "editing", "page_id": pid, "impl": impl,
                        "current_html": raw, "target_html": reduced,
                        "instruction": (f'Remove the "{label}" {kind} block from the page '
                                        f"entirely. Keep everything else unchanged."),
                        "meta": {"op": "delete", "family": "block", "label": label,
                                 "tag": kind, "direction": "apply"}})
            # Разворот пары удаления -> операция «добавить».
            out.append({"task_type": "editing", "page_id": pid, "impl": impl,
                        "current_html": reduced, "target_html": raw,
                        "instruction": (f'Add back a "{label}" {kind} block in the place '
                                        f"where it belongs, matching the surrounding "
                                        f"styling. Keep everything else unchanged."),
                        "meta": {"op": "add", "family": "block", "label": label,
                                 "tag": kind, "direction": "reverse"}})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="Data/synth_pilot/build/manifest.jsonl")
    ap.add_argument("--raw", default="Data/synth_pilot/raw")
    ap.add_argument("--out", default="Data/synth_pilot/editing.jsonl")
    ap.add_argument("--png-dir", default="Data/synth_pilot/build/edited")
    ap.add_argument("--limit", type=int, default=0, help="максимум сэмплов на страницу")
    args = ap.parse_args()

    from renderlib import render_page, visual_diff

    rng = random.Random(SEED)
    png_dir = Path(args.png_dir).resolve()
    png_dir.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in Path(args.manifest).open(encoding="utf-8")]
    ok = [r for r in recs if r["status"] == "ok"]

    total, dropped = 0, 0
    with tempfile.TemporaryDirectory(prefix="synth_edit_") as tmp, \
            Path(args.out).open("w", encoding="utf-8") as w:
        work = Path(tmp)
        for r in ok:
            raw = (Path(args.raw) / f"{r['id']}.html").read_text(encoding="utf-8")
            samples = build_for_page(r["id"], raw, r["impl"], rng)
            if args.limit:
                rng.shuffle(samples)
                samples = samples[:args.limit]

            kept = []
            for k, s in enumerate(samples):
                # Контракт §2: у editing одна картинка — рендер ТЕКУЩЕЙ страницы,
                # то есть той, которую модели предстоит править, а не эталона.
                try:
                    cur_img, _, _ = render_page(s["current_html"], r["impl"], work,
                                                f"{r['id']}_e{k}_cur")
                    tgt_img, _, _ = render_page(s["target_html"], r["impl"], work,
                                                f"{r['id']}_e{k}_tgt")
                except Exception:
                    dropped += 1
                    continue
                d = visual_diff(cur_img, tgt_img)
                if d < MIN_VISUAL_DIFF:
                    dropped += 1
                    continue
                png = png_dir / f"{r['id']}__{s['meta']['op']}_{k}.png"
                cur_img.save(png)
                s["current_png"] = str(png)
                s["meta"]["visual_diff"] = round(d, 3)
                w.write(json.dumps(s, ensure_ascii=False) + "\n")
                kept.append(s)

            total += len(kept)
            print(f"  {r['id']} {r['impl']:<14} -> {len(kept)}/{len(samples)} сэмплов "
                  f"({', '.join(sorted({x['meta']['op'] for x in kept})) or '—'})")

    print(f"\nвсего {total} сэмплов editing из {len(ok)} страниц -> {args.out}")
    if dropped:
        print(f"отброшено как невидимые на рендере: {dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
