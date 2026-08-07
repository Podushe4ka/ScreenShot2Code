"""Контактный лист по синтетическому набору — для проверки глазами.

Нужен потому, что автоматика ловит не всё. Линт видит запрещённый тег, метрики видят
пустой скриншот, но «страница отрендерилась, только вёрстка поехала и блоки налезли друг
на друга» проходит все пороги. На пилоте это самая частая причина брака, и находится она
только взглядом.

Два режима:
  pages     — сетка миниатюр всех принятых страниц с подписями;
  polishing — пары «эталон | испорченная» для задачи polishing, чтобы убедиться,
              что деградация действительно видна и действительно похожа на то,
              что модель будет чинить.

Использование:
    .venv/bin/python Data/converters/synth/contact_sheet.py pages
    .venv/bin/python Data/converters/synth/contact_sheet.py polishing
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

THUMB_W = 300
PAD = 10
LABEL_H = 34
BG = (24, 24, 27)
FG = (240, 240, 245)


def _font(size: int = 13):
    """Шрифт с кириллицей и CJK. Дефолтный битмап-шрифт PIL их не содержит и рисует
    подписи квадратами — а подписи здесь несут смысл (что за деградация, какой язык)."""
    for path in ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                 "/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _thumb(path: str, width: int = THUMB_W, max_h: int = 460) -> Image.Image:
    img = Image.open(path).convert("RGB")
    h = round(img.height * width / img.width)
    img = img.resize((width, h))
    return img.crop((0, 0, width, min(h, max_h)))


def _sheet(items: list[tuple[Image.Image, str]], cols: int, out: Path) -> None:
    if not items:
        print("нечего рисовать")
        return
    cell_w = max(i.width for i, _ in items)
    cell_h = max(i.height for i, _ in items) + LABEL_H
    rows = (len(items) + cols - 1) // cols
    W = cols * cell_w + (cols + 1) * PAD
    H = rows * cell_h + (rows + 1) * PAD
    sheet = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(sheet)
    font = _font()
    for k, (img, label) in enumerate(items):
        r, c = divmod(k, cols)
        x = PAD + c * (cell_w + PAD)
        y = PAD + r * (cell_h + PAD)
        sheet.paste(img, (x, y))
        d.text((x + 2, y + img.height + 8), label[:60], fill=FG, font=font)
    sheet.save(out)
    print(f"{len(items)} миниатюр -> {out}  ({W}x{H})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["pages", "polishing"])
    ap.add_argument("--manifest", default="Data/synth_pilot/build/manifest.jsonl")
    ap.add_argument("--polishing", default="Data/synth_pilot/polishing.jsonl")
    ap.add_argument("--out", default="Data/synth_pilot/build/contact_sheet.png")
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    recs = [json.loads(l) for l in Path(args.manifest).open(encoding="utf-8")]
    ok = {r["id"]: r for r in recs if r["status"] == "ok"}

    if args.mode == "pages":
        items = [(_thumb(r["png"]),
                  f"{r['id']} {r['impl']} {r['tier']}/{r['lang']} "
                  f"{r['dom_nodes']}узл {r['h']}px")
                 for r in ok.values()]
        _sheet(items, args.cols, Path(args.out))
    else:
        items = []
        for line in Path(args.polishing).open(encoding="utf-8"):
            s = json.loads(line)
            if s["page_id"] not in ok:
                continue
            items.append((_thumb(ok[s["page_id"]]["png"]), f"{s['page_id']} ЭТАЛОН"))
            items.append((_thumb(s["current_png"]),
                          f"↑ {s['meta']['what']} (diff {s['meta']['visual_diff']})"))
        _sheet(items, 4, Path(args.out).with_name("contact_sheet_polishing.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
