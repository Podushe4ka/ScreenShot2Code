#!/usr/bin/env python3
"""
convert_parallel.py — параллельный (многопроцессный) конвертер WebSight → формат контракта.

Логика вынесена в convert_lib.py (единый источник правды). Здесь — только оркестрация:
  фаза 1: стрим + лёгкие фильтры -> список HTML;
  фаза 2: ProcessPoolExecutor(spawn) -> воркеры convert_lib.process_one (placeholder+precompile+render);
  фаза 3: save_to_disk + приёмка + токен-отчёт.

Профиль — production (self-contained). Интерактивная версия — convert.ipynb.
Запуск обычно через Docker (см. Dockerfile): контейнер = браузеры + либы + зависимости.
    python convert_parallel.py --target 5000 --n-workers 32
"""
import argparse
import hashlib
import multiprocessing
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from datasets import Dataset, load_dataset, load_from_disk

from convert_lib import (FEATURES, MAX_PIXELS, MIN_PIXELS, RENDER_WIDTH,
                         TOKENIZER_ID_DEFAULT, ahash, count_tokens, hamming,
                         process_one, qwen_image_tokens)

DATASET = "HuggingFaceM4/WebSight"     # v0.2, Tailwind
SPLIT = "train"


# ------------------------------------------------------------------ фаза 1
def collect_candidates(target, max_scan, near_dup):
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
            if any(hamming(hsh, o) <= near_dup for o in hashes):
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
def token_report(rows):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID_DEFAULT)
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


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--max-scan", type=int, default=50000)
    ap.add_argument("--n-workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="./websight_drafting_pilot")
    ap.add_argument("--near-dup", type=int, default=None)
    ap.add_argument("--token-report", action="store_true",
                    help="пересчитать токены (тянет transformers/torch). Без флага — печатаем "
                         "зафиксированные числа WebSight v0.2.")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    print(f"источник: {DATASET} | target: {args.target} | воркеров: {args.n_workers} | out: {out_dir}")

    htmls = collect_candidates(args.target, args.max_scan, args.near_dup)

    # preflight: тест precompile+render в главном процессе (сразу видно причину проблем)
    print("[preflight] тест precompile + render...")
    _t = process_one("<html><head><script src='https://cdn.tailwindcss.com'></script></head>"
                     "<body><div class='bg-blue-500 text-white p-4 flex'>test</div></body></html>")
    if _t[0] != "ok":
        print("[preflight] ПРОВАЛ:\n" + _t[2])
        raise SystemExit("precompile или render не работают — почини окружение (tailwindcss / chromium).")
    print("[preflight] OK")

    # фаза 2: параллельно. spawn (не fork!) — Playwright ломается через fork после старта браузера.
    rows, errs = [], []
    ctx = multiprocessing.get_context("spawn")
    from tqdm import tqdm
    with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=ctx) as ex:
        for res in tqdm(ex.map(process_one, htmls, chunksize=4), total=len(htmls), desc="[фаза 2] render"):
            if res[0] == "ok":
                _, html, png = res
                rows.append({"task_type": "drafting", "images": [{"bytes": png, "path": None}],
                             "current_html": "", "target_html": html, "instruction": ""})
            else:
                errs.append((res[1], res[2]))
    print(f"[фаза 2] готово сэмплов: {len(rows)} | ошибок: {len(errs)}")
    for msg, cnt in Counter(m for m, _ in errs).most_common(3):
        print(f"  ✗ {cnt}×  {msg}")
    if not rows:
        raise SystemExit("Все воркеры упали. Первый traceback:\n" + errs[0][1])

    # фаза 3
    ds = Dataset.from_list(rows, features=FEATURES)
    ds.save_to_disk(out_dir)
    ds2 = load_from_disk(out_dir)
    widths = {s["images"][0].size[0] for s in ds2}
    assert widths == {RENDER_WIDTH}, f"ширины разные: {widths}"
    assert all(s["target_html"] and "<img" not in s["target_html"].lower() for s in ds2)
    print(f"[приёмка] OK: {len(ds2)} сэмплов, ширина {RENDER_WIDTH}, нет <img>, load_from_disk ✓")
    if args.token_report:
        token_report(list(ds2))
    else:
        # Зафиксировано на WebSight v0.2 production (пересчёт: --token-report, нужен torch).
        print("[токены] отчёт пропущен (--token-report для пересчёта). Известные WebSight v0.2:")
        print("[токены]   код p99≈3860, картинка p99≈1672, всего p99≈5532 -> max_length≈6144")
    print(f"\n=== ПЕРЕДАЧА SFT ===\nпуть: {out_dir}\nзагрузка: load_from_disk(<путь>)\n"
          f"max_length: 6144 (WebSight v0.2; пересчёт --token-report)\n"
          f"монтирование (§7): -v {out_dir}:/data")


if __name__ == "__main__":
    main()
