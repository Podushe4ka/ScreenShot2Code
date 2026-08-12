"""Строит сэмплы задачи `polishing` из чистых страниц.

Задача polishing (UI2Code^N §3.1): на вход эталонный скриншот, текущий код и рендер этого
кода; на выход исправленный код. Значит нужна пара «испорченная версия -> эталон», в которой
порча ВИДНА на рендере.

Порча делается скриптом по управляемой библиотеке, а не берётся из выхлопа модели.
Честное ограничение: у UI2Code^N входы-рендеры диверсифицированы выхлопом нескольких VLM
(их модель, GLM-4.5V, Claude-4-Sonnet), и это ближе к распределению, которое модель увидит
на инференсе. Скриптовая порча даёт проверяемую истину и нулевую стоимость, но её ошибки
систематичны — модель может выучить «искать флип flex-direction» вместо «сравнивать
картинки». В v2 половину слайса надо заменить реальным выхлопом базовой модели с бенча.

Каждая деградация ПРОВЕРЯЕТСЯ рендером: если картинка изменилась меньше порога — правка
невидима и учить на ней нечему; если больше — страница разрушена, и это уже не polishing.

Использование:
    VENDOR_DIR=$(pwd)/Data/vendor .venv/bin/python Data/converters/synth/degrade.py
"""

import argparse
import json
import random
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mutate import _mask_placeholders, _unmask_placeholders  # noqa: E402
from renderlib import render_page, visual_diff  # noqa: E402

SEED = 42

# Полоса «видимо, но не разрушительно». Ниже нижней границы правка незаметна на скриншоте
# и сэмпл учит модель менять код без визуального повода; выше верхней — страница
# перестала быть той же самой, и задача превращается из polishing в переверстку с нуля.
MIN_DIFF, MAX_DIFF = 0.06, 0.55


# деградации
# Каждая возвращает (испорченный_html, описание) или None, если применить некуда.

def deg_flex_direction(html: str, rng: random.Random):
    if "flex-row" in html:
        return html.replace("flex-row", "flex-col"), "flex-row -> flex-col"
    if re.search(r"flex-direction\s*:\s*row", html):
        return (re.sub(r"flex-direction\s*:\s*row", "flex-direction: column", html),
                "flex-direction row -> column")
    return None


def deg_drop_max_width(html: str, rng: random.Random):
    out = re.sub(r"\bmax-w-(?:xs|sm|md|lg|xl|\dxl|screen-\w+|\[[^\]]+\])\b", "", html)
    if out == html:
        out = re.sub(r"max-width\s*:[^;}\"']+;?", "", html)
    return (out, "снят max-width") if out != html else None


def deg_drop_gap(html: str, rng: random.Random):
    out = re.sub(r"\bgap-(?:x-|y-)?\d+(?:\.\d+)?\b", "", html)
    if out == html:
        out = re.sub(r"\bgap\s*:[^;}\"']+;?", "", html)
    return (out, "убраны отступы между элементами (gap)") if out != html else None


def deg_grid_columns(html: str, rng: random.Random):
    if re.search(r"\bgrid-cols-[2-9]\b", html):
        return re.sub(r"\bgrid-cols-[2-9]\b", "grid-cols-1", html), "сетка схлопнута в одну колонку"
    m = re.search(r"grid-template-columns\s*:[^;}]+", html)
    if m:
        return (html.replace(m.group(0), "grid-template-columns: 1fr"),
                "сетка схлопнута в одну колонку")
    return None


def deg_palette_shift(html: str, rng: random.Random):
    """Сдвигает ВСЕ оттенки фона в одно семейство — страница «теряет» палитру."""
    hues = ("slate|gray|zinc|red|orange|amber|yellow|lime|green|emerald|teal|cyan|"
            "sky|blue|indigo|violet|purple|fuchsia|pink|rose")
    out, n = re.subn(rf"\bbg-(?:{hues})-(\d{{2,3}})\b", r"bg-slate-\1", html)
    return (out, f"палитра фонов сведена к серому ({n} мест)") if n >= 3 else None


def deg_font_scale(html: str, rng: random.Random):
    """Сбивает типографскую шкалу: заголовки мельчают, иерархия пропадает."""
    pairs = [("text-5xl", "text-xl"), ("text-4xl", "text-lg"), ("text-3xl", "text-base"),
             ("text-2xl", "text-sm")]
    out = html
    for a, b in pairs:
        out = out.replace(a, b)
    if out != html:
        return out, "сбита шкала заголовков"
    out = re.sub(r"font-size\s*:\s*([\d.]+)rem", lambda m: f"font-size: {float(m.group(1))*0.6:.2f}rem", html)
    return (out, "сбита шкала заголовков") if out != html else None


def deg_drop_padding(html: str, rng: random.Random):
    # Направление отступа сохраняем: pt-6 -> pt-0, а не p-0. Иначе «убрали верхний
    # отступ» на деле схлопывает отступы со всех четырёх сторон, и описание правки
    # расходится с тем, что реально сделано.
    out = re.sub(r"\bp([trblxy])?-(?:[4-9]|1[0-6])\b",
                 lambda m: f"p{m.group(1) or ''}-0", html)
    return (out, "убраны внутренние отступы") if out != html else None


DEGRADATIONS = [deg_flex_direction, deg_drop_max_width, deg_drop_gap, deg_grid_columns,
                deg_palette_shift, deg_font_scale, deg_drop_padding]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="Data/synth_pilot/build/manifest.jsonl")
    ap.add_argument("--raw", default="Data/synth_pilot/raw")
    ap.add_argument("--out", default="Data/synth_pilot/polishing.jsonl")
    ap.add_argument("--png-dir", default="Data/synth_pilot/build/degraded")
    ap.add_argument("--per-page", type=int, default=2, help="сколько деградаций на страницу")
    args = ap.parse_args()

    rng = random.Random(SEED)
    png_dir = Path(args.png_dir).resolve()
    png_dir.mkdir(parents=True, exist_ok=True)

    recs = [json.loads(l) for l in Path(args.manifest).open(encoding="utf-8")]
    ok = [r for r in recs if r["status"] == "ok"]

    total, tried, rejected = 0, 0, {"невидимо": 0, "разрушено": 0, "не применимо": 0,
                                    "рендер упал": 0}
    with tempfile.TemporaryDirectory(prefix="synth_degrade_") as tmp, \
            Path(args.out).open("w", encoding="utf-8") as w:
        work = Path(tmp)
        for r in ok:
            raw = (Path(args.raw) / f"{r['id']}.html").read_text(encoding="utf-8")
            ref_img, _, _ = render_page(raw, r["impl"], work, f"{r['id']}_ref")

            order = DEGRADATIONS[:]
            rng.shuffle(order)
            made = []
            for deg in order:
                if len(made) >= args.per_page:
                    break
                # Плейсхолдеры прячем на время правки — их разметка обязана остаться
                # побайтово той же, что подставляет replace_images_with_placeholder,
                # а серый цвет всё равно прибит инлайновым style (см. mutate.py).
                masked, _ = _mask_placeholders(raw)
                res = deg(masked, rng)
                if res is None:
                    rejected["не применимо"] += 1
                    continue
                bad_html, what = res
                bad_html = _unmask_placeholders(bad_html)
                tried += 1
                try:
                    bad_img, _, _ = render_page(bad_html, r["impl"], work,
                                                f"{r['id']}_{deg.__name__}")
                except Exception:
                    rejected["рендер упал"] += 1
                    continue
                d = visual_diff(ref_img, bad_img)
                if d < MIN_DIFF:
                    rejected["невидимо"] += 1
                    continue
                if d > MAX_DIFF:
                    rejected["разрушено"] += 1
                    continue

                png = png_dir / f"{r['id']}__{deg.__name__}.png"
                bad_img.save(png)
                w.write(json.dumps({
                    "task_type": "polishing", "page_id": r["id"], "impl": r["impl"],
                    "current_html": bad_html, "target_html": raw, "instruction": "",
                    "current_png": str(png), "meta": {"degradation": deg.__name__,
                                                      "what": what, "visual_diff": round(d, 3)},
                }, ensure_ascii=False) + "\n")
                made.append((deg.__name__, round(d, 3)))
                total += 1
            print(f"  {r['id']} {r['impl']:<14} -> {len(made)}: "
                  + ", ".join(f"{n} (diff {d})" for n, d in made))

    print(f"\nвсего {total} сэмплов polishing из {len(ok)} страниц (проб {tried}) -> {args.out}")
    print("отсев:", ", ".join(f"{k}={v}" for k, v in rejected.items() if v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
