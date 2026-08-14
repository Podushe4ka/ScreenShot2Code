#!/usr/bin/env python3
"""compare_raw.py — сличить ИСХОДНУЮ страницу WebUI с тем, что отдал конвертер.

Приёмка внутри конвертера (`convert_lib.accept_page`) сравнивает рендер ДО и ПОСЛЕ
tree-shaking. Это ловит поломки шейкера, но НЕ отвечает на вопрос «насколько итог вообще
похож на исходную страницу»: всё, что происходит до шейкинга (снос скриптов, плейсхолдеры
вместо картинок, отрезание внешних ресурсов), в то сравнение не попадает по построению.

Поэтому здесь три колонки на страницу, и границы между ними разделяют разные по природе
потери:

    A  исходник   html + <style>css</style> КАК ЕСТЬ, без единой нашей правки
    B  гигиена    + снос скриптов/внешних ссылок, де-блоб, <img> -> серый плейсхолдер
    C  итог       + tree-shaking CSS = target_html, который уезжает в датасет

  A → B  потери НАМЕРЕННЫЕ: картинок нет по конвенции бенча, скриптов нет ради
         детерминизма, внешних ресурсов нет ради оффлайна. Расхождение тут ожидаемо
         и само по себе не дефект.
  B → C  потерь быть НЕ ДОЛЖНО: шейкинг обязан оставить картинку прежней. Расхождение
         здесь — настоящая поломка.

Все три рендерятся ОДИНАКОВО и БЕЗ СЕТИ (внешние запросы режутся на уровне маршрута):
иначе колонка A получила бы фору в виде подгруженных шрифтов и картинок, которых у
остальных двух нет, и сравнение мерило бы доступность сети, а не работу конвертера.

    python compare_raw.py --cache webui_desktop.jsonl.gz --staging /path/stage_webui \
        --out Data/webui_compare -n 30
"""
import argparse
import gzip
import html as html_mod
import io
import json
import os
import random

import numpy as np
from PIL import Image

import convert_lib as cl

RENDER_WIDTH = 1280


def render_offline(browser, html_text, width=RENDER_WIDTH, timeout_ms=30000):
    """Полностраничный скриншот с ЗАРЕЗАННОЙ сетью. Возвращает PIL.Image или None."""
    ctx = browser.new_context(viewport={"width": width, "height": 1024}, device_scale_factor=1)
    try:
        page = ctx.new_page()
        page.set_default_timeout(timeout_ms)
        # Режем всё внешнее: колонка A иначе подтянула бы шрифты и картинки, а B и C — нет.
        page.route("**/*", lambda route, req: route.abort()
                   if req.url.startswith(("http://", "https://")) else route.continue_())
        page.set_content(html_text, wait_until="load")
        png = page.screenshot(full_page=True, animations="disabled")
        img = Image.open(io.BytesIO(png)).convert("RGB")
        return img.crop((0, 0, width, img.height)) if img.width > width else img
    except Exception:
        return None
    finally:
        ctx.close()


def similarity(a, b):
    """Попиксельное совпадение двух рендеров [0..1] с штрафом за разные габариты."""
    if a is None or b is None:
        return None
    w, h = min(a.width, b.width), min(a.height, b.height)
    if w == 0 or h == 0:
        return 0.0
    scale = (w * h) / max(a.width * a.height, b.width * b.height)
    x = np.asarray(a.crop((0, 0, w, h)), dtype=np.int16)
    y = np.asarray(b.crop((0, 0, w, h)), dtype=np.int16)
    return float(scale * (1.0 - np.abs(x - y).mean() / 255.0))


def build_variants(row):
    """(A, B) как строки HTML. C берётся готовым из staging'а."""
    raw_css = row.get("css") or ""
    raw_html = row.get("html") or ""
    # A: ровно то, что лежит в источнике — html плюс его стайлшит одним <style>.
    soup_raw = cl.make_soup(raw_html)
    head = soup_raw.find("head")
    if head is None:
        head = soup_raw.new_tag("head")
        (soup_raw.find("html") or soup_raw).insert(0, head)
    if raw_css.strip():
        st = soup_raw.new_tag("style")
        st.string = raw_css
        head.append(st)
    a_html = str(soup_raw)

    # B: гигиена и плейсхолдеры, но стайлшит ещё ПОЛНЫЙ (шейкинга нет).
    soup, css_all, _ = cl.assemble_page(raw_html, raw_css)
    cl.replace_images_with_placeholder(soup)
    b_html = cl.build_html(soup, css_all)
    return a_html, b_html


CSS = """
body{font:13px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     margin:0;padding:24px;background:#f6f7f9;color:#1c1e21}
h1{font-size:20px;margin:0 0 4px}.sub{color:#606770;margin:0 0 20px;max-width:900px}
.row{background:#fff;border:1px solid #dcdfe3;border-radius:8px;margin-bottom:20px;padding:12px}
.hd{font-size:12px;color:#606770;margin-bottom:8px}
.hd b{color:#1c1e21;font-size:13px}
.cols{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.col{border:1px solid #eceef0;border-radius:6px;overflow:hidden}
.cap{padding:5px 8px;font-size:11px;background:#f0f2f5;border-bottom:1px solid #eceef0}
.shot{height:340px;overflow:hidden;background:#fff}.shot img{width:100%;display:block}
.ok{color:#15803d;font-weight:600}.warn{color:#b45309;font-weight:600}.bad{color:#b91c1c;font-weight:600}
table{border-collapse:collapse;margin:12px 0 20px}td,th{border:1px solid #dcdfe3;padding:5px 10px;text-align:right}
th:first-child,td:first-child{text-align:left}
"""


def verdict(sim, good, warn):
    if sim is None:
        return "bad", "рендер не удался"
    cls = "ok" if sim >= good else ("warn" if sim >= warn else "bad")
    return cls, f"{sim:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True, help="jsonl.gz от fetch_columns.py (исходные колонки)")
    ap.add_argument("--staging", required=True, help="каталог конвертера с pages/")
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", "--count", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    converted = {}
    with open(os.path.join(args.staging, "manifest.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") == "ok" and r.get("html"):
                converted[r["sample_id"]] = r
    print(f"[сравнение] в staging'е принятых страниц: {len(converted)}")

    rows = []
    with gzip.open(args.cache, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["sample_id"] in converted:
                rows.append(row)
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.count]
    print(f"[сравнение] беру {len(rows)} страниц")

    from playwright.sync_api import sync_playwright
    out_rows = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--disable-gpu", "--disable-dev-shm-usage"])
        for i, row in enumerate(rows, 1):
            sid = row["sample_id"]
            rec = converted[sid]
            a_html, b_html = build_variants(row)
            with open(os.path.join(args.staging, rec["html"]), encoding="utf-8") as f:
                c_html = f.read()

            imgs = {}
            for tag, text in (("a", a_html), ("b", b_html), ("c", c_html)):
                img = render_offline(browser, text)
                imgs[tag] = img
                if img is not None:
                    img.save(os.path.join(args.out, f"{sid}_{tag}.png"))
            with open(os.path.join(args.out, f"{sid}_c.html"), "w", encoding="utf-8") as f:
                f.write(c_html)

            out_rows.append({
                "id": sid, "source": row.get("source_name"),
                "component": row.get("component_type"),
                "css_raw": len(row.get("css") or ""),
                "css_min": rec.get("stat", {}).get("css_len_min"),
                "sim_ab": similarity(imgs["a"], imgs["b"]),
                "sim_bc": similarity(imgs["b"], imgs["c"]),
                "sim_ac": similarity(imgs["a"], imgs["c"]),
                "size_a": f"{imgs['a'].width}×{imgs['a'].height}" if imgs["a"] else "—",
                "size_c": f"{imgs['c'].width}×{imgs['c'].height}" if imgs["c"] else "—",
            })
            print(f"  [{i}/{len(rows)}] {sid} {row.get('source_name')}: "
                  f"A→B {out_rows[-1]['sim_ab']} · B→C {out_rows[-1]['sim_bc']}", flush=True)
        browser.close()

    write_index(args.out, out_rows)
    summarize(out_rows)


def summarize(rows):
    def stats(key):
        v = sorted(r[key] for r in rows if r[key] is not None)
        if not v:
            return "—"
        return f"p10={v[len(v)//10]:.3f} p50={v[len(v)//2]:.3f}"

    print(f"\n=== СЛИЧЕНИЕ ИСХОДНИКА И ИТОГА ({len(rows)} страниц) ===")
    print(f"  A→B (гигиена + плейсхолдеры, потери намеренные): {stats('sim_ab')}")
    print(f"  B→C (tree-shaking, потерь быть не должно):       {stats('sim_bc')}")
    print(f"  A→C (исходник против итога):                     {stats('sim_ac')}")
    broken = [r for r in rows if r["sim_bc"] is not None and r["sim_bc"] < 0.99]
    print(f"  страниц, где ШЕЙКИНГ изменил картинку (B→C < 0.99): {len(broken)}")
    for r in broken[:10]:
        print(f"    {r['id']} {r['source']}: B→C={r['sim_bc']:.3f}")


def write_index(out_dir, rows):
    parts = [
        "<!doctype html><meta charset='utf-8'><title>WebUI: исходник против итога</title>",
        f"<style>{CSS}</style>",
        "<h1>WebUI: исходник против того, что отдал конвертер</h1>",
        "<p class='sub'><b>A</b> — страница как есть в источнике. <b>B</b> — после гигиены "
        "(снос скриптов и внешних ресурсов, <code>&lt;img&gt;</code> → серый плейсхолдер), "
        "стайлшит ещё полный. <b>C</b> — после tree-shaking, это и есть "
        "<code>target_html</code> в датасете.<br>"
        "Все три сняты одинаково и <b>без сети</b>. Расхождение <b>A→B ожидаемо</b> "
        "(картинок нет по конвенции бенча), расхождение <b>B→C — дефект</b>: шейкинг обязан "
        "оставить картинку прежней.</p>",
    ]
    parts.append("<table><tr><th>страница</th><th>A→B</th><th>B→C</th><th>A→C</th>"
                 "<th>CSS исходный</th><th>CSS итог</th></tr>")
    for r in rows:
        _, ab = verdict(r["sim_ab"], 0.9, 0.7)
        cb, bc = verdict(r["sim_bc"], 0.99, 0.95)
        _, ac = verdict(r["sim_ac"], 0.9, 0.7)
        parts.append(f"<tr><td>{html_mod.escape(r['id'])}</td><td>{ab}</td>"
                     f"<td class='{cb}'>{bc}</td><td>{ac}</td>"
                     f"<td>{r['css_raw']}</td><td>{r['css_min'] if r['css_min'] is not None else '—'}</td></tr>")
    parts.append("</table>")

    for r in rows:
        cb, bc = verdict(r["sim_bc"], 0.99, 0.95)
        parts.append(
            f"<div class='row'><div class='hd'><b>{html_mod.escape(r['id'])}</b> · "
            f"{html_mod.escape(str(r['source']))} · {html_mod.escape(str(r['component']))} · "
            f"CSS {r['css_raw']} → {r['css_min'] if r['css_min'] is not None else '—'} симв. · "
            f"A→B {r['sim_ab'] if r['sim_ab'] is None else round(r['sim_ab'],3)} · "
            f"B→C <span class='{cb}'>{bc}</span></div><div class='cols'>"
            f"<div class='col'><div class='cap'>A · исходник ({r['size_a']})</div>"
            f"<div class='shot'><img src='{r['id']}_a.png' loading='lazy'></div></div>"
            f"<div class='col'><div class='cap'>B · гигиена + плейсхолдеры</div>"
            f"<div class='shot'><img src='{r['id']}_b.png' loading='lazy'></div></div>"
            f"<div class='col'><div class='cap'>C · итог, target_html ({r['size_c']})</div>"
            f"<div class='shot'><img src='{r['id']}_c.png' loading='lazy'></div></div>"
            f"</div></div>")
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"[сравнение] витрина -> {out_dir}/index.html")


if __name__ == "__main__":
    main()
