#!/usr/bin/env python3
"""convert_parallel.py (WebCode2M) — параллельный (многопроцессный) конвертер
WebCode2M -> формат drafting-контракта.

Зеркалит `Data/converters/websight/convert_parallel.py`, но:
  • источник и поля берутся из `convert_lib` этой папки (WebCode2M, поле `text`);
  • `process_one` НЕ делает Tailwind-precompile (реальный CSS), зато санитайзит внешние ресурсы.

Логика — в `convert_lib.py`. Здесь только оркестрация:
  фаза 1: стрим + лёгкие фильтры -> список HTML;
  фаза 2: ProcessPoolExecutor(spawn) -> convert_lib.process_one;
  фаза 3: save_to_disk + приёмка + токен-отчёт.

    python convert_parallel.py --target 5000 --n-workers 32
"""
import argparse
import gzip
import hashlib
import json
import multiprocessing
import os
import shutil
import struct
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk

# Сколько сэмплов складывать в один кусок при записи. Все картинки куска лежат
# в ОДНОМ бинарном массиве Arrow, а у него 32-битные оффсеты — предел 2 ГБ на
# массив. 15k сэмплов это ~7.5 ГБ, и `Dataset.from_list` на всём наборе сразу
# падает с «offset overflow while concatenating arrays» (на 3k проходило
# впритык, ~1.5 ГБ). Кусок в 500 страниц — это сотни МБ, с запасом.
CHUNK_ROWS = 500

from convert_lib import (DATASET_ID, HTML_FIELD, IMAGE_FIELD, FEATURES, RENDER_WIDTH,
                         SPLIT, TOKENIZER_ID_DEFAULT, ahash, count_tokens, hamming,
                         process_one, qwen_image_tokens)

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def png_size(data):
    """(width, height) из заголовка PNG — без декодирования пикселей (IHDR за 8-байтной сигнатурой)."""
    assert data[:8] == _PNG_SIG, "не PNG — png_size рассчитан на скриншоты Playwright"
    return struct.unpack(">II", data[16:24])


# фаза 1
def load_html_cache(path, target):
    """Кандидаты из кэша прошлого прогона, если их там хватает.

    Фаза 1 стримит шарды источника целиком (для 15k это ~4 ГБ и ~40 минут), а
    нужен из них один HTML — картинку мы всё равно рендерим сами. Падение любой
    следующей фазы без кэша означало бы повторную скачку с нуля.
    """
    if not path or not os.path.exists(path):
        return None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        htmls = [json.loads(line)["html"] for line in f]
    if len(htmls) < target:
        print(f"[фаза 1] в кэше {len(htmls)} < target {target} — стримлю заново")
        return None
    print(f"[фаза 1] кандидаты из кэша {path}: беру {target} из {len(htmls)}")
    return htmls[:target]


def save_html_cache(path, htmls):
    if not path:
        return
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for h in htmls:
            f.write(json.dumps({"html": h}, ensure_ascii=False) + "\n")
    os.replace(tmp, path)   # атомарно: недописанный кэш не должен выглядеть готовым
    print(f"[фаза 1] кандидаты сохранены в кэш: {path}")


def collect_candidates(target, max_scan, near_dup):
    stream = load_dataset(DATASET_ID, split=SPLIT, streaming=True)
    htmls, seen, hashes = [], set(), []
    scanned = skipped_empty = skipped_dup = skipped_nd = 0
    for r in stream:
        if len(htmls) >= target or scanned >= max_scan:
            break
        scanned += 1
        html = (r.get(HTML_FIELD) or "").strip()
        img = r.get(IMAGE_FIELD)
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


# токен-отчёт
def token_report(rows, sizes):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID_DEFAULT)
    pairs = [(count_tokens(r["target_html"], tok), qwen_image_tokens(w, h))
             for r, (w, h) in zip(rows, sizes)]
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
    if p99_code > 8192:
        print("[токены] ⚠ WebCode2M p99 кода ВЫШЕ SFT-окна 8192 — нужен фильтр по бюджету "
              "(отсечь длинный хвост) перед SFT. Для pretrain ограничение мягче.")


# main
def main():
    ap = argparse.ArgumentParser(description="WebCode2M -> формат контракта (параллельно).")
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--max-scan", type=int, default=50000)
    ap.add_argument("--n-workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="./webcode2m_pilot")
    ap.add_argument("--near-dup", type=int, default=None)
    ap.add_argument("--html-cache", default=None,
                    help="кэш кандидатов фазы 1 (jsonl.gz); по умолчанию <out>_htmls.jsonl.gz")
    ap.add_argument("--token-report", action="store_true",
                    help="пересчитать токены (тянет transformers/torch).")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    cache_path = os.path.abspath(args.html_cache or out_dir + "_htmls.jsonl.gz")
    print(f"источник: {DATASET_ID} | target: {args.target} | воркеров: {args.n_workers} | out: {out_dir}")

    htmls = load_html_cache(cache_path, args.target)
    if htmls is None:
        htmls = collect_candidates(args.target, args.max_scan, args.near_dup)
        save_html_cache(cache_path, htmls)

    # preflight: тест рендера в главном процессе (реальная страница WebCode2M — без Tailwind)
    print("[preflight] тест sanitize + render...")
    _t = process_one("<html><head><style>.b{background:#2563eb;color:#fff;padding:20px}</style>"
                     "</head><body><div class='b'>test</div><img src='x.png'></body></html>")
    if _t[0] != "ok":
        print("[preflight] ПРОВАЛ:\n" + _t[2])
        raise SystemExit("render не работает — почини окружение (chromium/playwright).")
    print("[preflight] OK")

    # фаза 2: параллельно. spawn (не fork!) — Playwright ломается через fork после старта браузера.
    # Готовые сэмплы копятся не до конца прогона, а до CHUNK_ROWS, после чего кусок
    # уходит на диск: и в предел 2 ГБ на массив Arrow не упираемся, и падение на
    # последнем сэмпле не уносит с собой весь отрендеренный набор.
    parts_dir = out_dir + "_parts"
    shutil.rmtree(parts_dir, ignore_errors=True)
    os.makedirs(parts_dir, exist_ok=True)
    buf, parts, errs, sizes = [], [], [], []

    def flush():
        if not buf:
            return
        path = os.path.join(parts_dir, f"part-{len(parts):05d}")
        Dataset.from_list(buf, features=FEATURES).save_to_disk(path)
        parts.append(path)
        buf.clear()

    ctx = multiprocessing.get_context("spawn")
    from tqdm import tqdm
    with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=ctx) as ex:
        for res in tqdm(ex.map(process_one, htmls, chunksize=4), total=len(htmls), desc="[фаза 2] render"):
            if res[0] == "ok":
                _, html, png = res
                sizes.append(png_size(png))
                buf.append({"task_type": "drafting", "images": [{"bytes": png, "path": None}],
                            "current_html": "", "target_html": html, "instruction": ""})
                if len(buf) >= CHUNK_ROWS:
                    flush()
            else:
                errs.append((res[1], res[2]))
    flush()
    n_rows = sum(len(load_from_disk(p)) for p in parts) if parts else 0
    print(f"[фаза 2] готово сэмплов: {n_rows} | ошибок: {len(errs)} | кусков: {len(parts)}")
    for msg, cnt in Counter(m for m, _ in errs).most_common(3):
        print(f"  ✗ {cnt}×  {msg}")
    if not parts:
        raise SystemExit("Все воркеры упали. Первый traceback:\n" + errs[0][1])

    # фаза 3: куски memory-mapped, поэтому склейка не поднимает набор в память целиком.
    ds = concatenate_datasets([load_from_disk(p) for p in parts])
    ds.save_to_disk(out_dir, max_shard_size="500MB")
    del ds
    shutil.rmtree(parts_dir, ignore_errors=True)
    ds2 = load_from_disk(out_dir)
    assert len(ds2) == len(sizes), f"рассинхрон: ds2={len(ds2)} sizes={len(sizes)}"
    # только нужная колонка: иначе приёмка тянет с сетевого диска все ~8 ГБ картинок
    assert all(s["target_html"] and "<img" not in s["target_html"].lower()
               for s in ds2.select_columns(["target_html"]))
    widths = {w for w, _ in sizes}
    assert widths == {RENDER_WIDTH}, f"ширины разные: {widths}"
    heights = sorted(h for _, h in sizes)
    p95 = heights[min(len(heights) - 1, int(len(heights) * .95))]
    print(f"[приёмка] OK: {len(ds2)} сэмплов, ширина {RENDER_WIDTH}, нет <img>, load_from_disk ✓")
    print(f"[приёмка] высота: min={heights[0]}, median={heights[len(heights)//2]}, "
          f"p95={p95}, max={heights[-1]}")
    if args.token_report:
        token_report(list(ds2), sizes)
    else:
        print("[токены] отчёт пропущен (--token-report для пересчёта). WebCode2M (EDA):")
        print("[токены]   код p99≈9920 (ВЫШЕ SFT-окна 8192 — нужен фильтр по длине для SFT).")
    print(f"\n=== ПЕРЕДАЧА ===\nпуть: {out_dir}\nзагрузка: load_from_disk(<путь>)\n"
          f"⚠ WebCode2M — реальные страницы: проверь долю ошибок рендера выше и "
          f"фильтр по токенам (p99 кода ~9920 > 8192).")


if __name__ == "__main__":
    main()
