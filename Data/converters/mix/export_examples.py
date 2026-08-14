#!/usr/bin/env python3
"""export_examples.py — выгрузить N примеров из собранного набора, чтобы посмотреть глазами.

Датасет лежит в Arrow с картинками-байтами внутри: заглянуть в него без кода нельзя, а
смотреть на данные глазами — единственный способ поймать то, что не ловится числами
(поехавшая вёрстка, пустой рендер, страница-заглушка). Тот же смысл, что у контактного
листа синтетики (`../synth/contact_sheet.py`), но по готовой солянке и с исходниками рядом.

На выходе:
    <out>/<source>__<id>.png     скриншот — то, что модель видит на входе
    <out>/<source>__<id>.html    target_html — то, что она должна написать
    <out>/index.html             витрина: превью, источник, длина таргета, ссылки

Выборка стратифицирована по колонке `source`: доли сохраняются, но каждому источнику
гарантируется минимум `--min-per-source`, иначе мелкие части (синтетика — 124 сэмпла на
фоне тысяч) в сотню примеров просто не попадут.

    python export_examples.py /path/mix_split --out Data/mix_examples -n 100
"""
import argparse
import html as html_mod
import os
import random
from collections import Counter, defaultdict


def pick_indices(sources, n, min_per_source, seed):
    """Индексы для выгрузки: пропорционально долям, но не меньше min_per_source на источник."""
    by_src = defaultdict(list)
    for i, s in enumerate(sources):
        by_src[s].append(i)
    rng = random.Random(seed)
    for idxs in by_src.values():
        rng.shuffle(idxs)

    total = len(sources)
    quota = {}
    for src, idxs in by_src.items():
        want = round(n * len(idxs) / total)
        quota[src] = min(len(idxs), max(min_per_source, want))

    # Пропорции могли раздуться из-за пола min_per_source — ужимаем самые крупные части.
    while sum(quota.values()) > n:
        big = max(quota, key=lambda s: (quota[s] - min_per_source, quota[s]))
        if quota[big] <= min_per_source:
            break
        quota[big] -= 1
    while sum(quota.values()) < n:
        cand = [s for s in quota if quota[s] < len(by_src[s])]
        if not cand:
            break
        big = max(cand, key=lambda s: len(by_src[s]))
        quota[big] += 1

    out = []
    for src, k in quota.items():
        out.extend(by_src[src][:k])
    return sorted(out)


INDEX_CSS = """
body{font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     margin:0;padding:24px;background:#f6f7f9;color:#1c1e21}
h1{font-size:20px;margin:0 0 4px}
.sub{color:#606770;margin:0 0 20px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}
.card{background:#fff;border:1px solid #dcdfe3;border-radius:8px;overflow:hidden;
      display:flex;flex-direction:column}
.shot{height:200px;overflow:hidden;background:#fff;border-bottom:1px solid #eceef0}
.shot img{width:100%;display:block}
.meta{padding:10px 12px;font-size:12px;color:#606770}
.src{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;
     color:#fff;margin-bottom:6px}
.links a{margin-right:10px}
"""

SRC_COLORS = {"webui_inline": "#2d6cdf", "webcode2m_complex": "#c2410c", "synth": "#15803d"}


def write_index(out_dir, rows, dataset_path):
    counts = Counter(r["source"] for r in rows)
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Примеры из солянки</title>",
        f"<style>{INDEX_CSS}</style>",
        f"<h1>Примеры из солянки — {len(rows)} шт.</h1>",
        f"<p class='sub'>Источник набора: <code>{html_mod.escape(dataset_path)}</code><br>"
        + " · ".join(f"{html_mod.escape(s)}: {c}" for s, c in counts.most_common())
        + "<br>Слева направо: скриншот — это ВХОД модели, рядом target_html — то, "
          "что она должна написать.</p>",
        "<div class='grid'>",
    ]
    for r in rows:
        color = SRC_COLORS.get(r["source"], "#606770")
        parts.append(
            f"<div class='card'>"
            f"<div class='shot'><a href='{r['png']}'><img src='{r['png']}' loading='lazy'></a></div>"
            f"<div class='meta'>"
            f"<span class='src' style='background:{color}'>{html_mod.escape(r['source'])}</span><br>"
            f"<code>{html_mod.escape(r['id'])}</code><br>"
            f"{r['w']}×{r['h']} px · таргет {r['chars']:,} симв.<br>"
            f"<span class='links'><a href='{r['png']}'>скриншот</a>"
            f"<a href='{r['html']}'>target_html</a></span>"
            f"</div></div>".replace(",", " ")
        )
    parts.append("</div>")
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset", help="путь к Dataset или DatasetDict (load_from_disk)")
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", "--count", type=int, default=100)
    ap.add_argument("--split", default="train", help="какой сплит брать у DatasetDict")
    ap.add_argument("--min-per-source", type=int, default=8,
                    help="минимум примеров на источник, чтобы мелкие части не пропали")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import DatasetDict, load_from_disk
    ds = load_from_disk(args.dataset)
    if isinstance(ds, DatasetDict):
        ds = ds[args.split]

    os.makedirs(args.out, exist_ok=True)
    sources = ds["source"]
    idxs = pick_indices(sources, args.count, args.min_per_source, args.seed)
    print(f"[примеры] беру {len(idxs)} из {len(ds)} ({args.split})")

    rows = []
    for i in idxs:
        s = ds[i]
        sid = (s.get("page_id") or f"n{i}").replace(":", "__").replace("/", "_")
        stem = sid if sid.startswith(s["source"]) else f"{s['source']}__{sid}"
        png_name, html_name = f"{stem}.png", f"{stem}.html"
        img = s["images"][0]
        img.save(os.path.join(args.out, png_name))
        with open(os.path.join(args.out, html_name), "w", encoding="utf-8") as f:
            f.write(s["target_html"])
        rows.append({"id": sid, "source": s["source"], "png": png_name, "html": html_name,
                     "w": img.width, "h": img.height, "chars": len(s["target_html"])})

    rows.sort(key=lambda r: (r["source"], r["id"]))
    write_index(args.out, rows, args.dataset)
    print(f"[примеры] по источникам: {dict(Counter(r['source'] for r in rows))}")
    print(f"[примеры] готово -> {args.out}/index.html")


if __name__ == "__main__":
    main()
