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
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "Data" / "converters" / "websight"))

from convert_lib import ahash  # noqa: E402
from renderlib import count_nodes, render_page, screenshot_stats  # noqa: E402
from slop import names as slop_names, scan as slop_scan  # noqa: E402

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

# Потолок высоты продиктован пиксель-бюджетом, а не вкусом. При MAX_PIXELS=2_097_152
# (SFT/train/formatting.py:51) и ширине рендера 1280:
#   * h = 1638 px  -> 2.10 Мп, нативный масштаб, ужатия нет вообще;
#   * h = 2048 px  -> 2.62 Мп, линейное ужатие 0.89 — текст ещё читаем;
#   * h = 3300 px  -> 4.22 Мп, ужатие 0.71 — мелкий текст в таблицах теряется.
# Учить модель воспроизводить текст, которого не видно на входе, бессмысленно: она
# получает штраф за то, чего не могла прочитать. Поэтому сложность набираем ПЛОТНОСТЬЮ
# (колонки, таблицы, боковые панели), а не длиной простыни.
MIN_HEIGHT, MAX_HEIGHT = 400, int(os.environ.get("SYNTH_MAX_HEIGHT", 2048))

# Бюджет кода в токенах — из SFT/configs/gen.py: max_length 16384 = 14176 (код)
# + 2048 (визуальные токены при MAX_PIXELS 2.10 Мп) + ~160 (промпт).
CODE_BUDGET_TOKENS = int(os.environ.get("SYNTH_CODE_BUDGET_TOKENS", 14176))
TOKENIZER_ID = os.environ.get("SYNTH_TOKENIZER", "Qwen/Qwen3-VL-8B-Instruct")


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
    if "fa-" in raw and "font-awesome" not in raw.lower():
        bad.append("классы fa-* (шрифты не вендорятся, отрендерятся квадраты)")
    return bad


# Токенайзер тот же, что в конвертерах (convert_lib.TOKENIZER_ID_DEFAULT). Грузим лениво:
# при работе без сети/кеша считать токены не обязательно, а рендерить надо всё равно.
_TOK = {}


def count_tokens(text: str, model_id: str) -> int | None:
    """Длина target_html в токенах. Возвращает None, если токенайзер недоступен.

    Именно токены, а не байты, — настоящий бюджет: в SFT/configs/gen.py на код отводится
    CODE_BUDGET_TOKENS=14176 из max_length 16384 (остальное — 2048 визуальных токенов и
    ~160 на промпт). Байты для этого негодны: у react_cdn исходник компактнее
    отрисованного DOM (данные разворачиваются через .map()), у static_inline — почти
    совпадает, поэтому одна и та же полоса байт означает для двух стилей разное.
    """
    if model_id not in _TOK:
        try:
            from transformers import AutoTokenizer
            _TOK[model_id] = AutoTokenizer.from_pretrained(model_id)
        except Exception as e:
            print(f"  ! токенайзер {model_id} недоступен ({type(e).__name__}), "
                  f"счёт токенов пропущен", file=sys.stderr)
            _TOK[model_id] = None
    tok = _TOK[model_id]
    return None if tok is None else len(tok(text, add_special_tokens=False)["input_ids"])


def process_one(brief: dict, raw_path: Path, out_dir: Path, work: Path) -> dict:
    rec = {"id": brief["id"], "impl": brief["impl"], "tier": brief["tier"],
           "lang": brief["lang"], "status": "ok", "reasons": []}

    raw = raw_path.read_text(encoding="utf-8", errors="replace")
    rec["bytes"] = len(raw.encode("utf-8"))
    rec["reasons"] += lint(raw, brief)

    # --- рендер (асимметрия react_cdn живёт в renderlib.render_page) ---
    try:
        img, rendered, info = render_page(raw, brief["impl"], work, brief["id"])
    except Exception as e:
        rec["reasons"].append(f"рендер упал: {type(e).__name__}: {str(e)[:120]}")
        rec["status"] = "reject"
        return rec

    rec["materialized"] = info["materialized"]
    if info["materialized"] is False:
        rec["reasons"].append(f"материализация не удалась: {str(info.get('error'))[:120]}")
    elif not info["grew"]:
        rec["reasons"].append("DOM почти не вырос — похоже, React не отрисовал")

    rec["dom_nodes"] = count_nodes(rendered)
    lo, hi = brief["dom_nodes"]
    if not (lo * 0.7 <= rec["dom_nodes"] <= hi * 1.5):
        rec["reasons"].append(
            f"узлов {rec['dom_nodes']} вне тира {brief['tier']} ({lo}-{hi})")

    rec["placeholders"] = raw.count(PLACEHOLDER_SIG)
    # Машинные дефолты («AI slop») — ОТЧЁТ, не отбраковка: часть приёмов законна в
    # приборной панели и незаконна на лендинге, решает человек. Метрика важна потому,
    # что slop это корреляция: страницы скатываются к одному шаблону, и разнообразие
    # набора падает. См. Data/converters/synth/slop.py.
    rec["slop"] = slop_scan(raw)

    # Байты — только справочно. Полоса target_bytes в ТЗ это подсказка генератору
    # «насколько крупную страницу писать», а не критерий приёмки: она несогласуема
    # сразу с impl (react компактнее на узел) и с языком (UTF-8 даёт +8-13% на
    # кириллице и CJK — замерено на калибровке). Решает число узлов и бюджет токенов.
    rec["tokens"] = count_tokens(raw, TOKENIZER_ID)
    if rec["tokens"] and rec["tokens"] > CODE_BUDGET_TOKENS:
        rec["reasons"].append(
            f"{rec['tokens']} токенов > бюджета кода {CODE_BUDGET_TOKENS} — не влезет в max_length")

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

    # Манифест СЛИВАЕМ, а не перезаписываем: при прогоне с --only иначе теряются записи
    # обо всех остальных страницах, и следующий же шаг конвейера видит набор из двух
    # строк вместо трёхсот. Порядок — по id, чтобы файл был стабилен между прогонами.
    manifest = out_dir / "manifest.jsonl"
    merged = {}
    if manifest.exists():
        for line in manifest.open(encoding="utf-8"):
            old = json.loads(line)
            merged[old["id"]] = old
    for r in recs:
        merged[r["id"]] = r
    with manifest.open("w", encoding="utf-8") as w:
        for pid in sorted(merged):
            w.write(json.dumps(merged[pid], ensure_ascii=False) + "\n")

    ok = [r for r in recs if r["status"] == "ok"]
    print(f"\nпринято {len(ok)}/{len(recs)} ({100*len(ok)/max(len(recs),1):.0f}%)  -> {manifest}")
    if len(recs) - len(ok):
        why = Counter(re.sub(r"\d+", "N", r.split(":")[0]) for x in recs for r in x["reasons"])
        print("причины брака:")
        for reason, n in why.most_common():
            print(f"  {n:>3}  {reason}")
    if ok:
        def pct(vals, p):
            vals = sorted(vals)
            return vals[min(int(p * len(vals)), len(vals) - 1)]

        sizes = [r["bytes"] for r in ok]
        print(f"размер принятых, Б: p50={pct(sizes,.5)} p95={pct(sizes,.95)} max={max(sizes)}")
        toks = [r["tokens"] for r in ok if r.get("tokens")]
        if toks:
            print(f"токенов: p50={pct(toks,.5)} p95={pct(toks,.95)} max={max(toks)} "
                  f"(бюджет кода {CODE_BUDGET_TOKENS})")
        nodes = [r["dom_nodes"] for r in ok]
        print(f"DOM-узлов: p50={pct(nodes,.5)} p95={pct(nodes,.95)} max={max(nodes)}")

        nm = slop_names()
        pages_with = Counter(t for r in ok for t in r.get("slop", {}))
        if pages_with:
            clean = sum(1 for r in ok if not r.get("slop"))
            print(f"машинные дефолты (отчёт, не отбраковка): чистых {clean}/{len(ok)}")
            for tid, n in pages_with.most_common(6):
                print(f"  {tid} {nm[tid]:<42} {n:>3} стр. ({100 * n // len(ok)}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
