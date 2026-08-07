"""Проверяет и рендерит сгенерированные страницы: линт -> рендер -> отбраковка.

Ключевая асимметрия для react_cdn (не потерять при правках): скриншот снимается с
МАТЕРИАЛИЗОВАННОГО DOM (после выполнения React+Babel), а в `target_html` едет СЫРОЙ
исходник со скриптами. Учим модель писать исходник, а не разжёванный DOM. Число DOM-узлов
для проверки тира считается тоже по материализованному — в сыром там один `<div id="root">`.

Рендер идёт через Evaluation/render.py и Data/converters/websight/convert_lib.py, теми же
функциями, что и остальные наборы, чтобы скриншоты были сравнимы.

Использование:
    VENDOR_DIR=$(pwd)/Data/vendor .venv/bin/python Data/converters/synth/build.py
"""

import argparse
import json
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "Evaluation"))
sys.path.insert(0, str(REPO / "Data" / "converters" / "websight"))

import numpy as np  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

import render as ev_render  # noqa: E402
from convert_lib import ahash, render_full  # noqa: E402

# Плейсхолдер обязан совпадать посимвольно с тем, что подставляет
# replace_images_with_placeholder в бенче и в конвертерах.
PLACEHOLDER_SIG = 'background-color:#d1d5db;width:100%;height:12rem'

# Разрешённые внешние хосты = ровно те, что перехватывает _VENDOR_MAP. Всё прочее
# route.abort() обрывает, и страница рендерится битой.
ALLOWED_URL_PARTS = (
    "react.development.js", "react.production.min.js",
    "react-dom.development.js", "react-dom.production.min.js",
    "babel.js", "babel.min.js", "cdn.tailwindcss.com", "all.min.css",
)

_URL_RE = re.compile(r'(?:src|href)\s*=\s*["\'](https?://[^"\']+)["\']', re.I)
_FORBIDDEN_TAGS = ("<iframe", "<noscript", "<object", "<embed")

# Полосы приёмки по скриншоту. Пустая/почти пустая страница — главный тихий брак:
# файл валиден, линт проходит, а учиться не на чем.
MIN_STD = 8.0
MIN_COLORS = 24
MIN_HEIGHT, MAX_HEIGHT = 400, 6000


def lint(raw: str, brief: dict) -> list[str]:
    """Статические претензии к исходнику. Пустой список = чисто."""
    bad = []
    head = raw.lstrip()[:200].lower()
    if not head.startswith("<!doctype html"):
        bad.append("нет <!DOCTYPE html> в начале")
    if "```" in raw:
        bad.append("markdown-ограждение в файле")
    if "<think" in raw.lower():
        bad.append("блок <think>")
    if re.search(r"<img\b", raw, re.I):
        bad.append("<img> вместо плейсхолдера")
    for tag in _FORBIDDEN_TAGS:
        if tag in raw.lower():
            bad.append(f"запрещённый тег {tag}>")
    for url in _URL_RE.findall(raw):
        if not any(p in url for p in ALLOWED_URL_PARTS):
            bad.append(f"внешний хост вне белого списка: {url[:80]}")
    if brief["impl"] == "static_inline":
        if re.search(r"<script", raw, re.I):
            bad.append("static_inline со <script>")
        if "tailwind" in raw.lower():
            bad.append("static_inline с tailwind")
    else:
        if "text/babel" not in raw:
            bad.append('react_cdn без <script type="text/babel">')
        if "cdn.tailwindcss.com" not in raw:
            bad.append("react_cdn без Tailwind")
    lo, hi = brief["target_bytes"]
    n = len(raw.encode("utf-8"))
    if not (lo * 0.6 <= n <= hi * 1.7):
        bad.append(f"размер {n} Б вне полосы тира {brief['tier']} ({lo}-{hi})")
    if "fa-" in raw and "font-awesome" not in raw.lower():
        bad.append("классы fa-* (шрифты не вендорятся, отрендерятся квадраты)")
    return bad


def count_nodes(html_text: str) -> int:
    soup = ev_render._make_soup(html_text)
    body = soup.body or soup
    return sum(1 for _ in body.find_all(True))


def screenshot_stats(img: PILImage.Image) -> dict:
    a = np.asarray(img.convert("RGB"))
    return {
        "w": img.width, "h": img.height,
        "std": round(float(a.std()), 2),
        "colors": int(len(np.unique(a.reshape(-1, 3), axis=0))),
    }


def process_one(brief: dict, raw_path: Path, out_dir: Path, work: Path) -> dict:
    rec = {"id": brief["id"], "impl": brief["impl"], "tier": brief["tier"],
           "lang": brief["lang"], "status": "ok", "reasons": []}

    raw = raw_path.read_text(encoding="utf-8", errors="replace")
    rec["bytes"] = len(raw.encode("utf-8"))
    rec["reasons"] += lint(raw, brief)

    # --- рендер ---
    if brief["impl"] == "react_cdn":
        tmp = work / f"{brief['id']}.html"
        tmp.write_text(raw, encoding="utf-8")
        info = ev_render.materialize_dom(str(tmp))
        rec["materialized"] = bool(info.get("materialized"))
        if not rec["materialized"]:
            rec["reasons"].append(f"материализация не удалась: {info.get('error', '?')[:120]}")
            rendered = raw
        else:
            rendered = tmp.read_text(encoding="utf-8", errors="replace")
            # Прирост длины — дешёвый признак того, что React действительно отрисовал:
            # при пустом #root DOM почти не растёт.
            if info["len_after"] < info["len_before"] * 1.2:
                rec["reasons"].append("DOM почти не вырос — похоже, React не отрисовал")
    else:
        rendered = raw
        rec["materialized"] = None

    rec["dom_nodes"] = count_nodes(rendered)
    lo, hi = brief["dom_nodes"]
    if not (lo * 0.7 <= rec["dom_nodes"] <= hi * 1.5):
        rec["reasons"].append(
            f"узлов {rec['dom_nodes']} вне тира {brief['tier']} ({lo}-{hi})")

    rec["placeholders"] = raw.count(PLACEHOLDER_SIG)

    try:
        img = render_full(rendered)
    except Exception as e:
        rec["reasons"].append(f"рендер упал: {type(e).__name__}: {str(e)[:120]}")
        rec["status"] = "reject"
        return rec

    png = out_dir / f"{brief['id']}.png"
    img.save(png)
    rec.update(screenshot_stats(img))
    rec["png"] = str(png.relative_to(REPO))
    rec["ahash"] = ahash(img)

    if rec["std"] < MIN_STD or rec["colors"] < MIN_COLORS:
        rec["reasons"].append(
            f"скриншот пустой (std={rec['std']}, цветов={rec['colors']})")
    if not (MIN_HEIGHT <= rec["h"] <= MAX_HEIGHT):
        rec["reasons"].append(f"высота {rec['h']}px вне полосы {MIN_HEIGHT}-{MAX_HEIGHT}")

    rec["status"] = "reject" if rec["reasons"] else "ok"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--briefs", default="Data/synth_pilot/briefs.jsonl")
    ap.add_argument("--raw", default="Data/synth_pilot/raw")
    ap.add_argument("--out", default="Data/synth_pilot/build")
    ap.add_argument("--only", nargs="*", help="конкретные id")
    args = ap.parse_args()

    briefs = {}
    for line in Path(args.briefs).open(encoding="utf-8"):
        b = json.loads(line)
        briefs[b["id"]] = b

    raw_dir, out_dir = Path(args.raw).resolve(), Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = args.only or sorted(p.stem for p in raw_dir.glob("*.html"))
    ids = [i for i in ids if (raw_dir / f"{i}.html").exists()]
    if not ids:
        print(f"в {raw_dir} нет сгенерированных страниц", file=sys.stderr)
        return 1

    recs = []
    with tempfile.TemporaryDirectory(prefix="synth_build_") as tmp:
        work = Path(tmp)
        for i, pid in enumerate(ids, 1):
            if pid not in briefs:
                print(f"  ? {pid}: нет такого ТЗ в briefs.jsonl — пропуск")
                continue
            rec = process_one(briefs[pid], raw_dir / f"{pid}.html", out_dir, work)
            recs.append(rec)
            mark = "✓" if rec["status"] == "ok" else "✗"
            extra = f" | {'; '.join(rec['reasons'])}" if rec["reasons"] else ""
            print(f"  {mark} [{i}/{len(ids)}] {pid} {rec['impl']:<14} "
                  f"узлов={rec.get('dom_nodes','?'):<4} {rec.get('w','?')}x{rec.get('h','?')} "
                  f"std={rec.get('std','?')}{extra}")

    # near-dup: одинаковые страницы из разных ТЗ — признак схлопывания генерации
    by_hash = Counter(r["ahash"] for r in recs if r.get("ahash"))
    for r in recs:
        if r.get("ahash") and by_hash[r["ahash"]] > 1 and r["status"] == "ok":
            r["status"] = "reject"
            r["reasons"].append("near-dup: совпал average-hash с другой страницей")

    manifest = out_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as w:
        for r in recs:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = [r for r in recs if r["status"] == "ok"]
    print(f"\nпринято {len(ok)}/{len(recs)} ({100*len(ok)/max(len(recs),1):.0f}%)  -> {manifest}")
    if len(recs) - len(ok):
        why = Counter(re.sub(r"\d+", "N", r.split(":")[0]) for x in recs for r in x["reasons"])
        print("причины брака:")
        for reason, n in why.most_common():
            print(f"  {n:>3}  {reason}")
    if ok:
        sizes = sorted(r["bytes"] for r in ok)
        q = lambda p: sizes[min(int(p * len(sizes)), len(sizes) - 1)]  # noqa: E731
        print(f"размер принятых, Б: p50={q(.5)} p95={q(.95)} max={sizes[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
