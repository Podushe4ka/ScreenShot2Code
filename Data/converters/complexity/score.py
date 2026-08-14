#!/usr/bin/env python3
"""score.py — посчитать признаки сложности для набора страниц и сложить `features.jsonl`.

Вход — одно из двух:
  * `--staging DIR`  — каталог конвертера (`manifest.jsonl` + `pages/<id>.html`), т.е. WebUI
    или синтетика после приёмки;
  * `--jsonl FILE`   — строки `{"id":…, "html":…}` или `{"id":…, "html_path":…}`, т.е. кандидаты
    WebCode2M от `../webcode2m/scan_complex.py`.

Режимы:
  * `--mode rendered` (по умолчанию) — считать по ОТРЕНДЕРЕННОЙ странице, как требует план.
    ~0.3–1 с/страница;
  * `--mode static`   — без браузера, ~1 мс/страница. Нужен только чтобы просеять
    ~200 тысяч кандидатов WebCode2M до шортлиста, который уже можно отрендерить.

Сводная сложность (`complexity`) считается ПО НАБОРУ: это взвешенное среднее перцентильных
рангов признаков внутри данного прогона (`features.composite_scores`). Поэтому число
сравнимо только внутри одного файла — что и требуется отбору по перцентилям.

    python score.py --staging /path/stage_webui --out features.jsonl --mode rendered -j 4
"""
import argparse
import gzip
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_W = {}


def _init(mode, width):
    import features
    _W["f"] = features
    _W["mode"] = mode
    _W["width"] = width


def _one(item):
    f = _W["f"]
    html = item.get("html_text")
    if html is None:
        try:
            with open(item["html_path"], encoding="utf-8") as fh:
                html = fh.read()
        except Exception as e:
            return {"id": item["id"], "error": f"read: {e}"}
    try:
        if _W["mode"] == "rendered":
            feats = f.rendered_features(html, tokens_code=item.get("tokens_code"),
                                        width=_W["width"])
        else:
            feats = f.static_features(html, tokens_code=item.get("tokens_code"))
    except Exception as e:
        return {"id": item["id"], "error": f"{type(e).__name__}: {e}"}
    # Чернила считаются по УЖЕ снятому скриншоту — второй рендер ради этого не нужен.
    if item.get("png"):
        feats.update(f.ink_features(item["png"]))
    feats["id"] = item["id"]
    for k in ("source", "source_name", "component_type", "tokens_code", "tokens_total",
              "png", "html_path", "w", "h"):
        if k in item and k not in feats:
            feats[k] = item[k]
    return feats


def load_staging(path):
    """Принятые страницы staging-каталога (WebUI / синтетика)."""
    items = []
    with open(os.path.join(path, "manifest.jsonl"), encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("status") != "ok" or not rec.get("html"):
                continue
            items.append({"id": rec["sample_id"],
                          "html_path": os.path.join(path, rec["html"]),
                          "png": os.path.join(path, rec["png"]) if rec.get("png") else None,
                          "tokens_code": rec.get("tokens_code"),
                          "tokens_total": rec.get("tokens_total"),
                          "source_name": rec.get("source_name"),
                          "component_type": rec.get("component_type"),
                          "w": rec.get("w"), "h": rec.get("h")})
    return items


def load_jsonl(path):
    op = gzip.open if path.endswith(".gz") else open
    items = []
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            item = {"id": rec.get("id") or rec.get("sample_id")}
            if "html" in rec:
                item["html_text"] = rec["html"]
            elif "html_path" in rec:
                item["html_path"] = rec["html_path"]
            for k in ("tokens_code", "tokens_total", "source", "source_name", "component_type"):
                if k in rec:
                    item[k] = rec[k]
            items.append(item)
    return items


def fill_tokens(items, tokenizer_id):
    """Досчитать `tokens_code` там, где его нет. Один раз в главном процессе — грузить
    токенайзер в каждый воркер значит держать N копий transformers в памяти."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    for it in items:
        if it.get("tokens_code"):
            continue
        text = it.get("html_text")
        if text is None:
            try:
                with open(it["html_path"], encoding="utf-8") as f:
                    text = f.read()
            except OSError:
                continue
        it["tokens_code"] = len(tok(text, add_special_tokens=False)["input_ids"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--staging", help="каталог конвертера (manifest.jsonl + pages/)")
    src.add_argument("--jsonl", help="кандидаты: {id, html|html_path, tokens_code}")
    ap.add_argument("--out", required=True, help="куда писать features.jsonl")
    ap.add_argument("--mode", choices=["rendered", "static"], default="rendered")
    ap.add_argument("--width", type=int, default=1280, help="ширина вьюпорта при рендере")
    ap.add_argument("-j", "--n-workers", type=int, default=4)
    ap.add_argument("--source", default=None,
                    help="метка источника, которая уедет в колонку `source` солянки")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-VL-8B-Instruct",
                    help="чем считать tokens_code, если его нет во входе")
    ap.add_argument("--no-tokens", action="store_true",
                    help="не досчитывать токены (тогда density не будет посчитана)")
    args = ap.parse_args()

    items = load_staging(args.staging) if args.staging else load_jsonl(args.jsonl)
    if args.limit:
        items = items[:args.limit]
    if args.source:
        for it in items:
            it["source"] = args.source
    print(f"[скоринг] страниц: {len(items)}  режим={args.mode}  воркеров={args.n_workers}")

    # ПЛОТНОСТЬ — ключевой признак отбора, а она считается «блоков на токен кода». Если во
    # входе токенов нет (манифест ещё не прошёл свою фазу подсчёта), density молча вышла бы
    # нулевой у ВСЕХ страниц, и барьер --density-min либо не отсёк бы ничего, либо выкосил
    # весь набор. Поэтому считаем сами, а не подставляем ноль.
    n_missing = sum(1 for it in items if not it.get("tokens_code"))
    if n_missing and not args.no_tokens:
        print(f"[скоринг] у {n_missing} страниц нет tokens_code — считаю токенайзером "
              f"{args.tokenizer}")
        fill_tokens(items, args.tokenizer)

    t0 = time.time()
    out = []
    with ProcessPoolExecutor(max_workers=args.n_workers, initializer=_init,
                             initargs=(args.mode, args.width)) as ex:
        for i, feats in enumerate(ex.map(_one, items, chunksize=8), 1):
            out.append(feats)
            if i % 500 == 0 or i == len(items):
                rate = i / max(1e-9, time.time() - t0)
                print(f"[скоринг] {i}/{len(items)}  {rate:.1f} стр/с  "
                      f"ETA {(len(items)-i)/max(1e-9,rate)/60:.0f} мин", flush=True)

    good = [f for f in out if not f.get("error")]
    import features as F
    scores = F.composite_scores(good)
    for f, s in zip(good, scores):
        f["complexity"] = round(s, 6)

    with open(args.out, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_err = len(out) - len(good)
    n_render = sum(1 for f in good if f.get("render_ok"))
    print(f"[скоринг] готово: {len(good)} с признаками ({n_render} по рендеру), "
          f"{n_err} с ошибкой -> {args.out}  ({(time.time()-t0)/60:.1f} мин)")
    if good:
        report_distribution(good)


def report_distribution(feats):
    def q(key, p):
        vals = sorted(float(f.get(key) or 0) for f in feats)
        return vals[min(len(vals) - 1, int(len(vals) * p))]

    print("\n  распределение признаков (p50 / p90 / p99):")
    for key in ("nodes", "depth", "distinct_tags", "css_decls", "unique_classes",
                "columns", "blocks_visible", "tokens_code"):
        print(f"    {key:16s} {q(key,.5):9.0f} {q(key,.9):9.0f} {q(key,.99):9.0f}")
    dens = [f.get("density") for f in feats if f.get("density")]
    if dens:
        dens.sort()
        def dq(p): return dens[min(len(dens) - 1, int(len(dens) * p))]
        print(f"    {'density':16s} {dq(.5):9.4f} {dq(.9):9.4f} {dq(.99):9.4f}  "
              f"(видимых блоков на токен кода)")
    n400 = sum(1 for f in feats if (f.get("nodes") or 0) >= 400)
    print(f"    страниц >=400 узлов: {n400} ({100*n400/len(feats):.1f}%)")


if __name__ == "__main__":
    main()
