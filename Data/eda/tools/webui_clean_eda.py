#!/usr/bin/env python3
"""Распределения ОЧИЩЕННОГО WebUI — те же метрики, что в Data/eda/notebooks/webui.ipynb,
но по выходу конвертера (Data/converters/webui), а не по сырому HF-датасету.

    python webui_clean_eda.py --staging stage_webui --cache webui_desktop.jsonl.gz --out-dir .
"""
import argparse, gzip, json, os, re
from urllib.parse import urlparse

import pandas as pd
from lxml import html as lxml_html

# ── хелперы ноутбука (1:1) ────────────────────────────────────────────────────
_DECL_RE = re.compile(r"[A-Za-z-]+\s*:[^;{}]+")
_RULE_BLOCK_RE = re.compile(r"[^{}]+\{([^{}]*)\}")
_DATA_URI_RE = re.compile(r"data:[A-Za-z0-9+/=;,._%-]{40,}")


def strip_data_uris(text):
    if not text:
        return text or "", 0
    n = 0
    def _repl(m):
        nonlocal n
        n += 1
        return "data:,"
    return _DATA_URI_RE.sub(_repl, text), n


def count_dom_nodes(tree):
    return len(tree.xpath(".//*")) + len(tree.xpath("//text()[normalize-space()]"))


def count_css_declarations(css_text):
    if not css_text:
        return 0
    return sum(sum(1 for _ in _DECL_RE.finditer(b)) for b in re.findall(r"\{([^{}]*)\}", css_text))


def count_css_rule_blocks(css_text):
    if not css_text:
        return 0
    return len(_RULE_BLOCK_RE.findall(css_text))


def count_unique_domains(html_text):
    if not html_text:
        return 0
    try:
        tree = lxml_html.fromstring(html_text)
    except Exception:
        return 0
    domains = set()
    for url in tree.xpath("//@src | //@href | //@data-src"):
        url = (url or "").strip()
        if not url or url.startswith(("#", "mailto:", "tel:", "javascript:", "data:")):
            continue
        target = "http:" + url if url.startswith("//") else url
        netloc = urlparse(target).netloc.lower()
        if netloc:
            domains.add(netloc)
    return len(domains)


def inline_css_text(html_text):
    """Очищенная страница self-contained: CSS лежит в <style>. Вынимаем его целиком."""
    try:
        tree = lxml_html.fromstring(html_text)
    except Exception:
        return ""
    return "\n".join(s.text_content() or "" for s in tree.xpath("//style"))


# ── сбор ──────────────────────────────────────────────────────────────────────
def rows_clean(staging):
    mf = os.path.join(staging, "manifest.jsonl")
    for line in open(mf, encoding="utf-8"):
        rec = json.loads(line)
        if rec.get("status") != "ok":
            yield rec, None
            continue
        p = os.path.join(staging, rec.get("html") or os.path.join("pages", rec["sample_id"] + ".html"))
        if not os.path.exists(p):
            p = os.path.join(staging, "pages", rec["sample_id"] + ".html")
        try:
            html_text = open(p, encoding="utf-8").read()
        except OSError:
            html_text = None
        rec["_staging"] = staging
        yield rec, html_text


def png_is_blank(path):
    """Белая картинка 1280x960 — известный грабль рендера (take_screenshot глотает
    исключение и пишет пустой кадр). Ловим по нулевому разбросу пикселей."""
    try:
        import numpy as np
        from PIL import Image
        a = np.asarray(Image.open(path).convert("L").resize((160, 120)))
        return float(a.std()) < 1.0
    except Exception:
        return None


def features_clean(rec, html_text):
    css_text = inline_css_text(html_text)
    out = {
        "sample_id": rec["sample_id"],
        "source_name": rec.get("source_name"),
        "framework": rec.get("framework"),
        "css_framework": rec.get("css_framework"),
        "component_type": rec.get("component_type"),
        "element_count_reported": rec.get("element_count"),
        "code_tokens": rec.get("tokens_code"),
        "tokens_img": rec.get("tokens_img"),
        "tokens_total": rec.get("tokens_total"),
        "img_w": rec.get("w"), "img_h": rec.get("h"),
        "html_chars": len(html_text),
        "css_chars": len(css_text),
        "css_chars_src": rec.get("css_len_src"),
        "html_chars_src": rec.get("html_len_src"),
        "css_decls": count_css_declarations(css_text),
        "css_rules": count_css_rule_blocks(css_text),
        "n_domains": count_unique_domains(html_text),
        "accept_via": rec.get("accept_via"),
        "final_score": (rec.get("metrics") or {}).get("final_score"),
        "pixel_sim": (rec.get("metrics") or {}).get("pixel_sim"),
        "dom_nodes": None, "parse_ok": False,
        "png_blank": None,
    }
    png = os.path.join(rec["_staging"], "pages", rec["sample_id"] + ".png")
    if os.path.exists(png):
        out["png_blank"] = png_is_blank(png)
    try:
        out["dom_nodes"] = count_dom_nodes(lxml_html.fromstring(html_text))
        out["parse_ok"] = True
    except Exception:
        pass
    return out


def features_raw(row, tok=None):
    html_text = row.get("html") or ""
    css_text = row.get("css") or ""
    html_clean, nb1 = strip_data_uris(html_text)
    css_clean, nb2 = strip_data_uris(css_text)
    out = {
        "sample_id": row.get("sample_id"),
        "source_name": row.get("source_name"),
        "framework": row.get("framework"),
        "css_framework": row.get("css_framework"),
        "component_type": row.get("component_type"),
        "element_count_reported": row.get("element_count"),
        "html_chars": len(html_text),
        "css_chars": len(css_text),
        "css_decls": count_css_declarations(css_text),
        "css_rules": count_css_rule_blocks(css_text),
        "n_domains": count_unique_domains(html_text),
        "n_blobs": nb1 + nb2,
        "code_tokens": None,
        "dom_nodes": None, "parse_ok": False,
    }
    if tok is not None:
        from token_len import count_tokens
        out["code_tokens"] = count_tokens(html_clean + "\n" + css_clean, tok)
    try:
        out["dom_nodes"] = count_dom_nodes(lxml_html.fromstring(html_text))
        out["parse_ok"] = True
    except Exception:
        pass
    return out


def skew_table(df, cols):
    idx, rows = [], []
    for label, c in cols.items():
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        idx.append(label)
        rows.append({"n": len(s), "median": s.median(), "mean": s.mean(),
                     "mean/median": (s.mean() / s.median()) if s.median() else float("nan"),
                     "p90": s.quantile(.90), "p99": s.quantile(.99), "max": s.max()})
    return pd.DataFrame(rows, index=idx).round(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", required=True)
    ap.add_argument("--cache", default=None, help="webui_desktop.jsonl.gz — для колонки 'сырой'")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-VL-8B-Instruct")
    args = ap.parse_args()

    # token_len.py импортируется ниже (`from token_len import ...`) — он лежит рядом,
    # в eda/tools, и Python сам кладёт каталог запускаемого скрипта в sys.path[0].
    # Раньше здесь стояли ещё две строки, добавлявшие тот же путь вручную — одна
    # дублировала это же поведение, вторая была абсолютным путём под конкретную
    # машину и ломалась бы при запуске где угодно ещё (тот же класс проблемы,
    # что чинили в eda/tools/design2code_study.py).

    # 1. очищенные
    clean, statuses = [], []
    for rec, html_text in rows_clean(args.staging):
        statuses.append(rec.get("status"))
        if html_text is None:
            continue
        clean.append(features_clean(rec, html_text))
    dfc = pd.DataFrame(clean)
    ok = dfc[dfc["parse_ok"]]

    print("=" * 78)
    print(f"ОЧИЩЕННЫЙ WebUI: {len(dfc)} страниц со status=ok, распарсено {len(ok)}")
    print("Статусы конвейера:", pd.Series(statuses).value_counts().to_dict())
    nb = dfc["png_blank"].sum() if "png_blank" in dfc else 0
    print(f"Белых (пустых) PNG среди принятых: {int(nb)} из {len(dfc)}")
    print("=" * 78)

    cols = {
        "Код HTML+CSS (токены)": "code_tokens",
        "DOM-узлы (наш подсчёт)": "dom_nodes",
        "DOM-узлы (element_count источника)": "element_count_reported",
        "CSS декларации": "css_decls",
        "CSS правила (блоки)": "css_rules",
        "Уникальные домены в HTML": "n_domains",
        "Ширина рендера, px": "img_w",
        "Высота рендера, px": "img_h",
        "Токены картинки": "tokens_img",
        "Токены всего (код+картинка)": "tokens_total",
    }
    print("\n[распределения — очищенный]")
    print(skew_table(ok, cols).to_string())

    # 2. сырой, те же sample_id
    if args.cache and os.path.exists(args.cache):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        want = set(dfc["sample_id"])
        raw = []
        with gzip.open(args.cache, "rt", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("sample_id") in want:
                    raw.append(features_raw(row, tok))
        dfr = pd.DataFrame(raw)
        okr = dfr[dfr["parse_ok"]]
        print(f"\n[распределения — СЫРОЙ источник, те же {len(okr)} страниц]")
        rcols = {k: v for k, v in cols.items() if v in dfr.columns}
        print(skew_table(okr, rcols).to_string())
        dfr.to_csv(os.path.join(args.out_dir, "webui_raw_same_pages.csv"), index=False)

        m = dfc.merge(dfr, on="sample_id", suffixes=("_clean", "_raw"))
        m["css_ratio"] = m["css_chars_raw"] / m["css_chars_clean"].replace(0, pd.NA)
        m["tok_ratio"] = m["code_tokens_raw"] / m["code_tokens_clean"].replace(0, pd.NA)
        print("\n[во сколько раз ужалось: сырой / очищенный]")
        print(m[["css_ratio", "tok_ratio"]].describe(percentiles=[.1, .5, .9]).round(2).to_string())

    # 3. хвост и max_length
    from token_len import recommend_max_length
    q = pd.to_numeric(ok["code_tokens"], errors="coerce").dropna()
    if len(q):
        pct = {f"p{p}": int(q.quantile(p / 100)) for p in [50, 90, 95, 99, 99.9]}
        print("\n[хвост длины кода]", pct, "| max:", int(q.max()))
        for cap in [8192, 16384, 32768]:
            keep = q[q <= cap]
            print(f"  cap={cap:>6}: остаётся {len(keep) / len(q) * 100:5.1f}% | "
                  f"p99={int(keep.quantile(.99)):>6} | max_length={recommend_max_length(keep.tolist())}")
        print("  рекомендуемый max_length (p99, без картинки):", recommend_max_length(q.tolist()))

    # 4. категории
    print("\n[категориальное разнообразие — очищенный]")
    for col in ["source_name", "framework", "css_framework", "component_type"]:
        vc = dfc[col].value_counts()
        print(f"\n{col}: {dfc[col].nunique(dropna=True)} уникальных")
        print(vc.head(8).to_string())

    dfc.to_csv(os.path.join(args.out_dir, "webui_clean_features.csv"), index=False)

    # 5. гистограммы, сетка как в §14 ноутбука
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()
    plots = [("code_tokens", "Код HTML+CSS (токены)"), ("dom_nodes", "DOM-узлы"),
             ("css_decls", "CSS-декларации"), ("css_rules", "CSS-правила (блоки)"),
             ("n_domains", "Доменов на страницу"), ("img_h", "Высота рендера, px"),
             ("tokens_total", "Токены код+картинка"), ("element_count_reported", "element_count источника")]
    for ax, (c, title) in zip(axes, plots):
        s = pd.to_numeric(ok[c], errors="coerce").dropna()
        if len(s):
            ax.hist(s, bins=40)
        ax.set_title(title)
    fig.suptitle(f"WebUI очищенный (конвертер) — n={len(ok)}", fontsize=14)
    plt.tight_layout()
    png = os.path.join(args.out_dir, "hist_webui_clean.png")
    plt.savefig(png, dpi=110)
    print("\nгистограммы:", png)


if __name__ == "__main__":
    main()
