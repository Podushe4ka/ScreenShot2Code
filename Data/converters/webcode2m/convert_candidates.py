#!/usr/bin/env python3
"""convert_candidates.py (WebCode2M) — конвертация ОТОБРАННЫХ кандидатов в staging-формат.

Чем отличается от соседнего `convert_parallel.py`: тот сам стримит датасет и берёт страницы
подряд, а сюда приходит готовый список сложных кандидатов от `scan_complex.py`. Для солянки
нужен именно этот путь — сначала просеять корпус по сложности, потом рендерить только
выживших (рендер дороже разбора на три порядка).

Сама конвертация не дублируется: используется `convert_lib.process_one` этой же папки
(санитайз внешних ресурсов -> де-блоб -> плейсхолдеры -> рендер). Tree-shaking здесь НЕ
нужен — в WebCode2M CSS уже лежит внутри страницы, а не отдельным стайлшитом сайта.

Формат выхода — тот же staging, что у WebUI (`../webui/convert_parallel.py`):
`manifest.jsonl` + `pages/<id>.html` + `pages/<id>.png`, чтобы солянка собиралась из
одинаковых каталогов.

    python convert_candidates.py --candidates wc2m_candidates.jsonl.gz --out /path/stage_wc2m -j 4
"""
import argparse
import gzip
import hashlib
import io
import json
import multiprocessing
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

# Каталог скрипта — для воркеров пула (им нужен convert_lib соседом).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# Пролог доступа к общему ядру — ОДИН И ТОТ ЖЕ во всех точках входа (см. common/__init__.py).
_CONV = os.path.dirname(_HERE)
if _CONV not in sys.path:
    sys.path.insert(0, _CONV)

from common.budget import TOKENIZER_ID_DEFAULT  # noqa: E402  — не литерал: один на трек

_W = {}


def _init(out_dir):
    import convert_lib
    _W["cl"] = convert_lib
    _W["out"] = out_dir
    os.makedirs(os.path.join(out_dir, "pages"), exist_ok=True)


def _one(item):
    cl = _W["cl"]
    t0 = time.time()
    sid = item["id"]
    status, a, b = cl.process_one(item["html"])
    rec = {"sample_id": sid, "source_name": "webcode2m", "lang": item.get("lang"),
           "nodes_static": item.get("nodes"), "html_len_src": len(item["html"]),
           "secs": round(time.time() - t0, 2)}
    if status != "ok":
        rec.update(status="error", error=str(a)[:300])
        return rec
    target_html, png = a, b
    pages = os.path.join(_W["out"], "pages")
    html_path = os.path.join(pages, f"{sid}.html")
    png_path = os.path.join(pages, f"{sid}.png")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(target_html)
    with open(png_path, "wb") as f:
        f.write(png)
    from PIL import Image
    img = Image.open(io.BytesIO(png))
    rec.update(status="ok",
               html=os.path.relpath(html_path, _W["out"]),
               png=os.path.relpath(png_path, _W["out"]),
               w=img.width, h=img.height,
               sha1=hashlib.sha1(target_html.encode("utf-8")).hexdigest(),
               ahash=cl.ahash(img), target_len=len(target_html))
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--candidates", required=True, help="jsonl.gz от scan_complex.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("-j", "--n-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--near-dup", type=int, default=4, help="-1 = выключить")
    ap.add_argument("--tokenizer", default=TOKENIZER_ID_DEFAULT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    items = []
    with gzip.open(args.candidates, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            items.append({"id": rec["id"], "html": rec["html"],
                          "lang": rec.get("lang"), "nodes": rec.get("nodes")})
            if args.limit and len(items) >= args.limit:
                break
    print(f"[фаза 1] кандидатов: {len(items)}")

    t0, records = time.time(), []
    manifest = os.path.join(args.out, "manifest.jsonl")
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=ctx,
                             initializer=_init, initargs=(args.out,)) as ex, \
            open(manifest, "w", encoding="utf-8") as mf:
        for i, rec in enumerate(ex.map(_one, items, chunksize=1), 1):
            records.append(rec)
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            mf.flush()
            if i % 200 == 0 or i == len(items):
                ok = sum(1 for r in records if r["status"] == "ok")
                rate = i / max(1e-9, time.time() - t0)
                print(f"[фаза 2] {i}/{len(items)}  ok={ok} ({100*ok/i:.1f}%)  "
                      f"{rate:.2f} стр/с  ETA {(len(items)-i)/max(1e-9,rate)/60:.0f} мин", flush=True)

    # Дедуп и подсчёт токенов заимствуем у WebUI-конвертера: логика там одна и та же,
    # а второй копии этих трёх функций трек уже наелся. Путь — через тот же _CONV.
    sys.path.insert(0, os.path.join(_CONV, "webui"))
    from convert_parallel import add_tokens, mark_exact_dups, mark_near_dups
    for rec in records:
        rec["_root"] = args.out
    n_dup = mark_exact_dups(records)
    n_nd = mark_near_dups(records, None if args.near_dup < 0 else args.near_dup)
    add_tokens(records, args.tokenizer)
    for rec in records:
        rec.pop("_root", None)
    with open(manifest, "w", encoding="utf-8") as mf:
        for rec in records:
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    st = Counter(r["status"] for r in records)
    ok = [r for r in records if r["status"] == "ok"]
    print(f"\n=== КОНВЕРТАЦИЯ WebCode2M-complex ({len(records)} стр., {(time.time()-t0)/60:.0f} мин) ===")
    for k, v in st.most_common():
        print(f"  {k:10s} {v:6d} ({100*v/len(records):.1f}%)")
    print(f"  точных дублей {n_dup}, near-dup {n_nd}")
    if ok:
        def q(vals, p):
            vals = sorted(vals)
            return vals[min(len(vals) - 1, int(len(vals) * p))]
        tok = [r.get("tokens_code", 0) for r in ok]
        print(f"  токены кода p50/p90/p99 = {q(tok,.5)}/{q(tok,.9)}/{q(tok,.99)}")


if __name__ == "__main__":
    main()
