#!/usr/bin/env python3
"""
websight_to_contract_parallel.py — параллельная (многопроцессная) версия конвертера.

Зачем: precompile (Tailwind CLI) + рендер (Playwright) делаются НА КАЖДУЮ страницу и
последовательно это медленно (~1.2с/пример). Работа независима → раскидываем по N
процессам-воркерам (каждый со своим браузером и tailwindcss). GPU не нужен — узкое
место это браузер/subprocess/сеть, а не матрицы; помогает много CPU-ядер на боксе.

Профиль — production (self-contained: плейсхолдеры + precompile + ре-рендер full-page,
фикс. ширина). Исходный последовательный конвертер с приёмкой/handoff в ячейках —
websight_to_contract.ipynb, он НЕ тронут.

Предпосылки:
    pip install datasets pillow beautifulsoup4 transformers playwright pytailwindcss
    playwright install chromium
Запуск (из папки Data/):
    python websight_to_contract_parallel.py --target 5000 --n-workers 16
    python websight_to_contract_parallel.py --target 500 --n-workers $(nproc)
"""
import argparse
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

from bs4 import BeautifulSoup
from datasets import (Dataset, Features, Image, Sequence, Value, load_dataset,
                      load_from_disk)
from PIL import Image as PILImage

# ------------------------------------------------------------------ конфиг
DATASET = "HuggingFaceM4/WebSight"      # v0.2, Tailwind
SPLIT = "train"
RENDER_WIDTH = 1280                     # ширина вьюпорта ре-рендера; высота — по контенту
MIN_PIXELS = 256 * 32 * 32             # оценка визуальных токенов (как в SFT formatting.py)
MAX_PIXELS = 1280 * 32 * 32
TOKENIZER_ID = "Qwen/Qwen3-VL-8B-Instruct"

# конвенция серых плейсхолдеров — РОВНО как в Evaluation/Experiments.ipynb
PLACEHOLDER_CLASSES = ["bg-gray-300", "w-full", "h-48", "rounded"]
PLACEHOLDER_STYLE = ("background-color:#d1d5db;width:100%;height:12rem;"
                     "border-radius:0.5rem;display:block;")

FEATURES = Features({
    "task_type":    Value("string"),
    "images":       Sequence(Image()),
    "current_html": Value("string"),
    "target_html":  Value("string"),
    "instruction":  Value("string"),
})

# ------------------------------------------------------------ гигиена HTML
_BG_RE = re.compile(r'background(-image)?\s*:\s*[^;{}"\']*url\([^)]*\)[^;{}"\']*', re.I)
_TW_CDN_RE = re.compile(r'<script\b[^>]*tailwind[^>]*>\s*</script>|<link\b[^>]*tailwind[^>]*>', re.I)


def strip_background_images(html_text):
    return _BG_RE.sub("background-color:#d1d5db", html_text)


def replace_images_with_placeholder(html_text):
    soup = BeautifulSoup(html_text, "html.parser")
    for img in soup.find_all("img"):
        div = soup.new_tag("div")
        div["class"] = PLACEHOLDER_CLASSES
        div["style"] = PLACEHOLDER_STYLE
        img.replace_with(div)
    return strip_background_images(str(soup))


def ahash(img, size=8):
    g = img.convert("L").resize((size, size))
    px = list(g.getdata()); avg = sum(px) / len(px)
    return sum(1 << i for i, p in enumerate(px) if p > avg)


def _hamming(a, b):
    return bin(a ^ b).count("1")


# ------------------------------------------------- Tailwind precompile (v4)
def precompile_tailwind(html_text):
    """CDN-Tailwind -> инлайновый <style> (только используемые классы). Tailwind v4
    через standalone `pytailwindcss` (без Node). Вызывается в воркере (свой subprocess)."""
    tw = shutil.which("tailwindcss")
    if tw is None:
        raise RuntimeError("нужен tailwindcss CLI: pip install pytailwindcss")
    clean = _TW_CDN_RE.sub("", html_text)
    with tempfile.TemporaryDirectory() as t:
        with open(os.path.join(t, "page.html"), "w", encoding="utf-8") as f:
            f.write(clean)
        with open(os.path.join(t, "in.css"), "w") as f:
            f.write('@import "tailwindcss";\n@source "./page.html";\n')
        subprocess.run([tw, "-i", "in.css", "-o", "out.css", "--minify"],
                       cwd=t, check=True, capture_output=True, text=True)
        with open(os.path.join(t, "out.css"), encoding="utf-8") as f:
            css = f.read()
    style = f"<style>{css}</style>"
    low = clean.lower()
    if "</head>" in low:
        i = low.index("</head>")
        return clean[:i] + style + clean[i:]
    return "<!doctype html><html><head><meta charset='utf-8'>" + style + "</head><body>" + clean + "</body></html>"


# ----------------------------------------- Playwright (свой браузер на процесс)
_PW = {"pw": None, "browser": None}


def _browser():
    # В отдельном ПРОЦЕССЕ нет запущенного asyncio-loop → sync-API Playwright работает
    # напрямую (без потока). Браузер поднимается один раз на воркер и переиспользуется.
    if _PW["browser"] is None:
        from playwright.sync_api import sync_playwright
        _PW["pw"] = sync_playwright().start()
        _PW["browser"] = _PW["pw"].chromium.launch()
    return _PW["browser"]


def render_full(html_text, width):
    page = _browser().new_page(viewport={"width": width, "height": 1024}, device_scale_factor=1)
    try:
        page.set_content(html_text, wait_until="networkidle")
        height = int(page.evaluate(
            "() => Math.max(document.body ? document.body.scrollHeight : "
            "document.documentElement.scrollHeight, 1)"))
        png = page.screenshot(clip={"x": 0, "y": 0, "width": width, "height": max(1, height)})
    finally:
        page.close()
    return png


def process_one(html_text):
    """Воркер: плейсхолдеры + precompile + рендер. Возвращает
    ("ok", target_html, png_bytes) или ("err", msg, traceback) — чтобы видеть причину."""
    try:
        html = replace_images_with_placeholder(html_text)
        html = precompile_tailwind(html)
        png = render_full(html, RENDER_WIDTH)
        return ("ok", html, png)
    except Exception as e:
        import traceback
        return ("err", f"{type(e).__name__}: {e}", traceback.format_exc())


# ------------------------------------------------------------------ фаза 1
def collect_candidates(target, max_scan, near_dup):
    """Стрим + лёгкие фильтры (пустые, дедуп HTML, near-dup картинки). Возвращает список HTML."""
    stream = load_dataset(DATASET, split=SPLIT, streaming=True)
    htmls, seen, hashes = [], set(), []
    scanned = skipped_empty = skipped_dup = skipped_nd = 0
    for r in stream:
        if len(htmls) >= target or scanned >= max_scan:
            break
        scanned += 1
        html = (r.get("text") or r.get("html") or "").strip()
        img = r.get("image")
        if not html or img is None:
            skipped_empty += 1
            continue
        if near_dup is not None:
            hsh = ahash(img)
            if any(_hamming(hsh, o) <= near_dup for o in hashes):
                skipped_nd += 1
                continue
            hashes.append(hsh)
        h = hashlib.sha1(html.encode("utf-8")).hexdigest()
        if h in seen:
            skipped_dup += 1
            continue
        seen.add(h)
        htmls.append(html)
    print(f"[фаза 1] кандидатов: {len(htmls)} (просмотрено {scanned}; "
          f"пустых={skipped_empty}, дублей={skipped_dup}, near-dup={skipped_nd})")
    return htmls


# --------------------------------------------------------------- токен-отчёт
def qwen_image_tokens(w, h, patch=28):
    px = min(max(w * h, MIN_PIXELS), MAX_PIXELS)
    return round(px / (patch * patch))


def token_report(rows):
    from transformers import AutoTokenizer
    from token_len import count_tokens  # из ./analysis
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    pairs = [(count_tokens(r["target_html"], tok), qwen_image_tokens(*r["images"][0].size)) for r in rows]
    code = sorted(c for c, _ in pairs)
    img = sorted(m for _, m in pairs)
    total = sorted(c + m for c, m in pairs)
    def q(a, x): return a[min(len(a) - 1, int(len(a) * x))]
    p99_code, p99_img = q(code, .99), q(img, .99)
    ml = ((p99_code + 63) // 64) * 64 + p99_img
    print(f"[токены] код:      median={code[len(code)//2]}, p99={p99_code}, max={code[-1]}")
    print(f"[токены] картинка: median={img[len(img)//2]}, p99={p99_img}, max={img[-1]}")
    print(f"[токены] всего:    median={total[len(total)//2]}, p99={q(total,.99)}, max={total[-1]}")
    print(f"[токены] рекомендуемый max_length (код p99 + картинка p99): {ml}")
    return ml


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=500, help="сколько принятых сэмплов собрать")
    ap.add_argument("--max-scan", type=int, default=50000, help="предел просмотренных сырых")
    ap.add_argument("--n-workers", type=int, default=os.cpu_count(), help="число процессов")
    ap.add_argument("--out", default="./websight_drafting_pilot", help="папка датасета")
    ap.add_argument("--near-dup", type=int, default=None, help="порог Хэмминга near-dup (None=выкл)")
    args = ap.parse_args()

    sys.path.append(os.path.abspath("analysis"))   # token_len.py
    out_dir = os.path.abspath(args.out)
    print(f"источник: {DATASET} | target: {args.target} | воркеров: {args.n_workers} | out: {out_dir}")

    htmls = collect_candidates(args.target, args.max_scan, args.near_dup)

    # preflight: тест precompile+render на одной странице (в главном процессе) — сразу видно причину
    print("[preflight] тест precompile + render...")
    _t = process_one("<html><head><script src='https://cdn.tailwindcss.com'></script></head>"
                     "<body><div class='bg-blue-500 text-white p-4 flex'>test</div></body></html>")
    if _t[0] != "ok":
        print("[preflight] ПРОВАЛ:\n" + _t[2])
        raise SystemExit("precompile или render не работают — почини окружение. Частое: нет "
                         "tailwindcss (pip install pytailwindcss) или chromium (playwright install chromium).")
    print("[preflight] OK")

    # фаза 2: параллельно precompile + render
    rows, errs = [], []
    from collections import Counter
    from tqdm import tqdm
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        for res in tqdm(ex.map(process_one, htmls, chunksize=4), total=len(htmls), desc="[фаза 2] render"):
            if res[0] == "ok":
                _, html, png = res
                rows.append({
                    "task_type": "drafting",
                    "images": [{"bytes": png, "path": None}],
                    "current_html": "",
                    "target_html": html,
                    "instruction": "",
                })
            else:
                errs.append((res[1], res[2]))
    print(f"[фаза 2] готово сэмплов: {len(rows)} | ошибок: {len(errs)}")
    for msg, cnt in Counter(m for m, _ in errs).most_common(3):
        print(f"  ✗ {cnt}×  {msg}")
    if not rows:
        raise SystemExit("Все воркеры упали. Первый traceback:\n" + errs[0][1])

    # фаза 3: сохранить + приёмка + отчёт
    ds = Dataset.from_list(rows, features=FEATURES)
    ds.save_to_disk(out_dir)
    ds2 = load_from_disk(out_dir)
    widths = {s["images"][0].size[0] for s in ds2}
    assert widths == {RENDER_WIDTH}, f"ширины разные: {widths}"
    assert all(s["target_html"] and "<img" not in s["target_html"].lower() for s in ds2)
    print(f"[приёмка] OK: {len(ds2)} сэмплов, ширина {RENDER_WIDTH}, нет <img>, load_from_disk ✓")
    token_report(list(ds2))
    print(f"\n=== ПЕРЕДАЧА SFT ===\nпуть: {out_dir}\nзагрузка: load_from_disk(<путь>)\n"
          f"монтирование (§7): -v {out_dir}:/data")


if __name__ == "__main__":
    main()
