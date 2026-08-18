#!/usr/bin/env python3
"""convert_parallel.py (WebUI) — батч-конвертация WebUI -> staging-каталог.

Оркестрация, вся логика — в `convert_lib.py`:
  фаза 1: читаем кэш кандидатов от `fetch_columns.py` (уже дедуплицирован по `sample_id`);
  фаза 2: ProcessPoolExecutor(spawn) -> `convert_lib.convert_one` (шаги 2–6 плана);
  фаза 3: токены + near-dup + сводка приёмки.

ПОЧЕМУ STAGING-КАТАЛОГ, А НЕ СРАЗУ `Dataset`. Дальше идут скоринг сложности
(`../complexity/score.py`) и отбор по перцентилям, и обоим нужны отрендеренная страница и
её HTML, а не упакованный арро. Плюс солянка (`../mix/build_mix.py`) мешает три источника —
удобнее, когда каждый лежит в одинаковом staging-формате:

    <out>/manifest.jsonl        одна строка на страницу: статус, статистика, метрики приёмки
    <out>/pages/<id>.html       принятый target_html
    <out>/pages/<id>.png        наш ре-рендер (шаг 5)

Тот же формат, что у синтетики (`Data/synth_pilot/build/manifest.jsonl`), — солянка потом
собирается из трёх одинаковых staging'ов.

    python convert_parallel.py --cache webui_desktop.jsonl.gz --out /path/staging --n-workers 4
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


# ── воркер ────────────────────────────────────────────────────────────────────
_W = {}


def _init_worker(out_dir, accept_threshold, keep_dead_fontface, do_accept,
                 min_class_coverage, pixel_threshold, metric_every):
    import convert_lib
    _W["cl"] = convert_lib
    _W["out"] = out_dir
    _W["work"] = os.path.join(out_dir, "_work", str(os.getpid()))
    os.makedirs(_W["work"], exist_ok=True)
    os.makedirs(os.path.join(out_dir, "pages"), exist_ok=True)
    _W["cfg"] = dict(accept_threshold=accept_threshold,
                     keep_dead_fontface=keep_dead_fontface,
                     do_accept=do_accept, min_class_coverage=min_class_coverage,
                     pixel_threshold=pixel_threshold)
    _W["metric_every"] = metric_every
    _W["seen"] = 0


def _work_one(row):
    """Одна страница. Исключение НЕ поднимается наверх: страница выпадает, пул живёт."""
    cl = _W["cl"]
    t0 = time.time()
    _W["seen"] += 1
    # Каждая metric_every-я страница считается полной метрикой даже при совпавших
    # пикселях — иначе в отчёте не останется распределения final_score.
    force = _W["metric_every"] > 0 and _W["seen"] % _W["metric_every"] == 0
    try:
        res = cl.convert_one(row, _W["work"], force_metric=force, **_W["cfg"])
    except Exception as e:
        import traceback
        res = {"sample_id": row.get("sample_id"), "status": "error",
               "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}

    rec = {
        "sample_id": row.get("sample_id"),
        "source_name": row.get("source_name"),
        "component_type": row.get("component_type"),
        "framework": row.get("framework"),
        "css_framework": row.get("css_framework"),
        "element_count": row.get("element_count"),
        "css_len_src": len(row.get("css") or ""),
        "html_len_src": len(row.get("html") or ""),
        "status": res.get("status"),
        "reason": res.get("reason"),
        "error": (res.get("error") or "")[:300],
        "stat": res.get("stat") or {},
        "metrics": {k: v for k, v in (res.get("metrics") or {}).items()
                    if k in ("final_score", "block_match", "text", "position", "color",
                             "clip", "pixel_sim")},
        "accept_via": res.get("accept_via"),
        "secs": round(time.time() - t0, 2),
    }
    if res.get("status") == "ok":
        sid = res["sample_id"]
        pages = os.path.join(_W["out"], "pages")
        html_path = os.path.join(pages, f"{sid}.html")
        png_path = os.path.join(pages, f"{sid}.png")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(res["target_html"])
        with open(png_path, "wb") as f:
            f.write(res["png"])
        from PIL import Image
        img = Image.open(io.BytesIO(res["png"]))
        rec.update(html=os.path.relpath(html_path, _W["out"]),
                   png=os.path.relpath(png_path, _W["out"]),
                   w=img.width, h=img.height,
                   sha1=hashlib.sha1(res["target_html"].encode("utf-8")).hexdigest(),
                   ahash=cl.ahash(img),
                   target_len=len(res["target_html"]))
    return rec


# ── фаза 3: токены и near-dup ────────────────────────────────────────────────

def add_tokens(records, tokenizer_id):
    """Длина таргета в токенах + визуальный бюджет картинки. Один раз в главном процессе:
    грузить токенайзер в каждом воркере — это транформерсы×N на 8 ГБ памяти."""
    from transformers import AutoTokenizer
    import convert_lib as cl
    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    for rec in records:
        if rec.get("status") != "ok":
            continue
        path = os.path.join(rec["_root"], rec["html"])
        with open(path, encoding="utf-8") as f:
            text = f.read()
        rec["tokens_code"] = cl.count_tokens(text, tok)
        rec["tokens_img"] = cl.qwen_image_tokens(rec["w"], rec["h"])
        rec["tokens_total"] = rec["tokens_code"] + rec["tokens_img"]


def mark_near_dups(records, near_dup):
    """Пометить почти-дубли по average-hash НАШЕГО рендера.

    Своя картинка, а не источника: скриншот WebUI снят по первому экрану, и по нему
    десяток разных страниц одного дизайн-система выглядят одинаково. У WebUI это особенно
    важно — 3 962 сэмпла из 9 803 это `component_type == "button"`, то есть витрины
    компонентов, отличающиеся парой строк.
    """
    if near_dup is None:
        return 0
    kept, n = [], 0
    for rec in records:
        if rec.get("status") != "ok":
            continue
        h = rec.get("ahash")
        if h is None:
            continue
        import convert_lib as cl
        if any(cl.hamming(h, o) <= near_dup for o in kept):
            rec["status"] = "near_dup"
            rec["reason"] = f"near-dup (ahash <= {near_dup})"
            n += 1
        else:
            kept.append(h)
    return n


def mark_exact_dups(records):
    seen, n = set(), 0
    for rec in records:
        if rec.get("status") != "ok":
            continue
        h = rec.get("sha1")
        if h in seen:
            rec["status"] = "dup"
            rec["reason"] = "точный дубль target_html"
            n += 1
        else:
            seen.add(h)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", default=None, help="jsonl.gz от fetch_columns.py")
    ap.add_argument("--out", required=True, help="staging-каталог")
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="сколько страниц взять (0 = все)")
    ap.add_argument("--accept-threshold", type=float, default=0.95,
                    help="порог приёмки конвертации (шаг 6): рендер до/после tree-shaking")
    ap.add_argument("--no-accept", action="store_true",
                    help="пропустить шаг 6 (только для отладки — это самопроверка конвертера)")
    ap.add_argument("--keep-dead-fontface", action="store_true",
                    help="оставлять @font-face без живого src (по умолчанию режем — мёртвый код)")
    ap.add_argument("--min-class-coverage", type=float, default=0.10,
                    help="минимальная доля классов разметки, у которых есть правило в CSS; "
                         "ниже -> стайлшита компонентов нет в источнике (см. styling_coverage)")
    ap.add_argument("--metric-every", type=int, default=10,
                    help="считать полную метрику Design2Code на каждой N-й странице даже при "
                         "совпавших пикселях (0 = только там, где она решает)")
    ap.add_argument("--pixel-threshold", type=float, default=0.995,
                    help="запасной порог приёмки: попиксельное совпадение рендеров до/после")
    ap.add_argument("--near-dup", type=int, default=4,
                    help="порог Хэмминга по average-hash нашего рендера; -1 = выключить")
    ap.add_argument("--tokenizer", default=TOKENIZER_ID_DEFAULT)
    ap.add_argument("--finalize", action="store_true",
                    help="не конвертировать, а доиграть фазу 3 (токены + дедуп) на готовом "
                         "манифесте — для прогона, остановленного на середине")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Фаза 2 длинная (часы), а токены и near-dup считаются только после неё. Без этого
    # режима остановленный на середине прогон терял бы весь пост-обсчёт: манифест пишется
    # построчно и переживает остановку, а вот `tokens_*` и пометки дублей — нет.
    if args.finalize:
        manifest_path = os.path.join(args.out, "manifest.jsonl")
        with open(manifest_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        print(f"[финализация] записей в манифесте: {len(records)}")
        # Метаданные (источник, тип компонента) в файлах страницы не лежат — подтягиваем
        # их из того же кэша кандидатов, если он передан.
        meta = {}
        if args.cache:
            with gzip.open(args.cache, "rt", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    meta[row["sample_id"]] = {
                        "source_name": row.get("source_name"),
                        "component_type": row.get("component_type"),
                        "framework": row.get("framework"),
                        "css_framework": row.get("css_framework"),
                        "css_len_src": len(row.get("css") or ""),
                        "html_len_src": len(row.get("html") or ""),
                    }
        recovered = recover_orphan_pages(records, args.out, meta)
        if recovered:
            print(f"[финализация] подобрано страниц с диска, не попавших в манифест: {recovered}")
        finalize(records, args, manifest_path)
        report(records, args, 0.0, *finalize.counts)
        return

    if not args.cache:
        raise SystemExit("нужен --cache (или --finalize для доигрывания фазы 3)")
    rows = []
    with gzip.open(args.cache, "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break
    print(f"[фаза 1] кандидатов из кэша: {len(rows)}")

    t0 = time.time()
    records = []
    ctx = multiprocessing.get_context("spawn")
    manifest_path = os.path.join(args.out, "manifest.jsonl")
    with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=ctx,
                             initializer=_init_worker,
                             initargs=(args.out, args.accept_threshold,
                                       args.keep_dead_fontface, not args.no_accept,
                                       args.min_class_coverage, args.pixel_threshold,
                                       args.metric_every)) as ex, \
            open(manifest_path, "w", encoding="utf-8") as mf:
        for i, rec in enumerate(ex.map(_work_one, rows, chunksize=1), 1):
            records.append(rec)
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            mf.flush()          # манифест переживает падение прогона — фаза 2 самая долгая
            if i % 50 == 0 or i == len(rows):
                ok = sum(1 for r in records if r["status"] == "ok")
                rate = i / max(1e-9, time.time() - t0)
                eta = (len(rows) - i) / max(1e-9, rate)
                print(f"[фаза 2] {i}/{len(rows)}  ok={ok} ({100*ok/i:.1f}%)  "
                      f"{rate:.2f} стр/с  ETA {eta/60:.0f} мин", flush=True)

    finalize(records, args, manifest_path)
    report(records, args, time.time() - t0, *finalize.counts)


def recover_orphan_pages(records, out_dir, meta=None):
    """Подобрать страницы, которые лежат в `pages/`, но до манифеста не доехали.

    ЗАЧЕМ. Файлы страницы пишет САМ ВОРКЕР, а строку манифеста — главный процесс, и только
    когда `ex.map` отдаст результат ПО ПОРЯДКУ. Одна тяжёлая страница в голове очереди
    задерживает выдачу, пока остальные воркеры уходят вперёд: на замере в манифесте было
    3 418 строк при 4 185 уже сконвертированных страницах на диске. Прерванный прогон без
    этого подбора выбросил бы ~750 честно посчитанных страниц.

    Страница попадает в `pages/` ТОЛЬКО со статусом `ok` (см. `_work_one`), то есть приёмку
    она уже прошла — восстановить нельзя лишь её метрики приёмки, а не сам факт.
    """
    import convert_lib as cl
    from PIL import Image

    known = {r.get("sample_id") for r in records}
    meta = meta or {}
    pages = os.path.join(out_dir, "pages")
    if not os.path.isdir(pages):
        return 0
    added = 0
    for name in sorted(os.listdir(pages)):
        if not name.endswith(".html"):
            continue
        sid = name[:-5]
        if sid in known:
            continue
        html_path = os.path.join(pages, name)
        png_path = os.path.join(pages, f"{sid}.png")
        if not os.path.exists(png_path):
            continue
        with open(html_path, encoding="utf-8") as f:
            target = f.read()
        img = Image.open(png_path)
        records.append({
            "sample_id": sid, "status": "ok", "recovered": True,
            "stat": {}, "metrics": {}, "accept_via": None,
            "html": os.path.relpath(html_path, out_dir),
            "png": os.path.relpath(png_path, out_dir),
            "w": img.width, "h": img.height,
            "sha1": hashlib.sha1(target.encode("utf-8")).hexdigest(),
            "ahash": cl.ahash(img), "target_len": len(target),
            "css_len_src": 0, "html_len_src": 0,
            **{k: v for k, v in (meta.get(sid) or {}).items()},
        })
        added += 1
    return added


def finalize(records, args, manifest_path):
    """Фаза 3: дедуп + токены + перезапись манифеста. Вынесена отдельно, чтобы её можно
    было доиграть на прерванном прогоне (`--finalize`)."""
    for rec in records:
        rec["_root"] = args.out
    n_dup = mark_exact_dups(records)
    n_nd = mark_near_dups(records, None if args.near_dup < 0 else args.near_dup)
    add_tokens(records, args.tokenizer)
    for rec in records:
        rec.pop("_root", None)
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for rec in records:
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finalize.counts = (n_dup, n_nd)
    return n_dup, n_nd


def report(records, args, secs, n_dup, n_nd):
    n = len(records)
    st = Counter(r["status"] for r in records)
    ok = [r for r in records if r["status"] == "ok"]
    print(f"\n=== ПРИЁМКА КОНВЕРТАЦИИ WebUI ({n} страниц, {secs/60:.0f} мин) ===")
    for k, v in st.most_common():
        print(f"  {k:10s} {v:6d}  ({100*v/n:.1f}%)")
    print(f"  из них точных дублей {n_dup}, near-dup {n_nd}")

    rej = [r for r in records if r["status"] == "rejected"]
    if rej:
        why = Counter((r.get("reason") or "").split(":")[0] for r in rej)
        print("  причины отбраковки:", dict(why))
    err = [r for r in records if r["status"] == "error"]
    if err:
        print("  топ ошибок:", Counter(r["error"].split(":")[0] for r in err).most_common(5))

    if not ok:
        return

    def q(vals, p):
        vals = sorted(vals)
        return vals[min(len(vals) - 1, int(len(vals) * p))]

    css_src = [r["css_len_src"] for r in ok]
    css_min = [r["stat"].get("css_len_min", 0) for r in ok]
    print("\n  CSS, символов:  источник p50/p99 = "
          f"{q(css_src,.5)}/{q(css_src,.99)}  ->  после шейкинга {q(css_min,.5)}/{q(css_min,.99)}")
    tok = [r.get("tokens_code", 0) for r in ok]
    tot = [r.get("tokens_total", 0) for r in ok]
    print(f"  токены кода p50/p90/p99 = {q(tok,.5)}/{q(tok,.9)}/{q(tok,.99)}; "
          f"код+картинка p99 = {q(tot,.99)}")
    sc = [r["metrics"].get("final_score") for r in ok if r["metrics"].get("final_score")]
    if sc:
        print(f"  приёмка (рендер до/после шейкинга) p1/p50 = {q(sc,.01):.3f}/{q(sc,.5):.3f} "
              f"при пороге {args.accept_threshold}")
    px = [r["metrics"].get("pixel_sim") for r in ok if r["metrics"].get("pixel_sim") is not None]
    if px:
        print(f"  попиксельное совпадение до/после p1/p50 = {q(px,.01):.4f}/{q(px,.5):.4f}")
    print(f"  принято по сигналам: {Counter(r.get('accept_via') for r in ok)}")
    print(f"  источники: {Counter(r.get('source_name') or '?' for r in ok).most_common(8)}")


if __name__ == "__main__":
    main()
