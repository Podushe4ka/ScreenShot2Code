#!/usr/bin/env python3
"""web2code_stream_json.py — реальная выборка Web2Code без скачивания всех 1.6 ГБ.

`Web2Code.json` — единый JSON-массив (не JSONL), поэтому `datasets` стримингом
берёт лишь дефолтный конфиг-сэмпл (100 шт). Здесь тянем HTTP-поток `Web2Code.json`
и инкрементально вычленяем объекты верхнего уровня сканером глубины скобок,
останавливаясь на SAMPLE_SIZE — по сети уходит лишь префикс (~десятки МБ).

Метрики (DOM / CSS-декларации / домены / токены) — 1-в-1 из webcode2m.ipynb.
    .venv/bin/python Data/eda/tools/web2code_stream_json.py
"""
import io
import json
import os
import re
import sys
from urllib.parse import urlparse
from urllib.request import Request, urlopen

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from lxml import html as lxml_html
from token_len import count_tokens, recommend_max_length
from transformers import AutoTokenizer

# Пролог доступа к общему ядру — тот же, что в конвертерах (см. converters/common/__init__.py).
_CONV = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "converters")
if _CONV not in sys.path:
    sys.path.insert(0, _CONV)
from common.budget import TOKENIZER_ID_DEFAULT  # noqa: E402

URL = "https://huggingface.co/datasets/MBZUAI/Web2Code/resolve/main/Web2Code.json"
SAMPLE_SIZE = 5000
# Токенайзер общий: числа этого прогона стоят в мастер-таблице datasets_overview.md
# рядом с числами остальных корпусов, и считаться они обязаны одним и тем же.
TOKENIZER_ID = TOKENIZER_ID_DEFAULT
PAPER_COUNT = 1_179_626


def count_dom_nodes(tree):
    return len(tree.xpath(".//*")) + len(tree.xpath("//text()[normalize-space()]"))


_DECL_RE = re.compile(r"[A-Za-z-]+\s*:[^;{}]+")


def count_css_declarations(tree):
    total = 0
    for style_attr in tree.xpath("//*/@style"):
        total += sum(1 for _ in _DECL_RE.finditer(style_attr))
    for style_text in tree.xpath("//style//text()"):
        for block in re.findall(r"\{([^{}]*)\}", style_text):
            total += sum(1 for _ in _DECL_RE.finditer(block))
    return total


def count_unique_domains(tree):
    domains = set()
    for url in tree.xpath("//@src | //@href | //@data-src"):
        url = (url or "").strip()
        if not url or url.startswith(("#", "mailto:", "tel:", "javascript:", "data:")):
            continue
        t = "http:" + url if url.startswith("//") else url
        netloc = urlparse(t).netloc.lower()
        if netloc:
            domains.add(netloc)
    return len(domains)


def stream_objects(url, limit):
    """Итерируем объекты верхнего уровня JSON-массива по HTTP-потоку, останавливаясь на limit."""
    req = Request(url, headers={"User-Agent": "eda/1.0"})
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    resp = urlopen(req, timeout=60)
    dec = io.TextIOWrapper(resp, encoding="utf-8")
    buf, depth, in_str, esc, started = [], 0, False, False, False
    n = 0
    while True:
        chunk = dec.read(65536)
        if not chunk:
            break
        for ch in chunk:
            if not started:
                if ch == "[":
                    started = True
                continue
            if depth == 0 and ch in " \t\r\n,":
                continue
            if depth == 0 and ch == "]":
                return
            buf.append(ch)
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        yield json.loads("".join(buf))
                        buf = []
                        n += 1
                        if n >= limit:
                            return


def extract_html(conversations):
    for turn in conversations or []:
        if turn.get("from") == "gpt":
            return turn.get("value") or ""
    return ""


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    print(f"tokenizer: {TOKENIZER_ID}\nстримим {URL}\nцель: {SAMPLE_SIZE}")
    records = []
    for obj in stream_objects(URL, SAMPLE_SIZE):
        text = extract_html(obj.get("conversations"))
        rec = {"html_chars": len(text), "html_tokens": count_tokens(text, tokenizer),
               "dom_nodes": None, "css_decls": None, "n_domains": None, "parse_ok": False}
        try:
            tree = lxml_html.fromstring(text)
            rec["dom_nodes"] = count_dom_nodes(tree)
            rec["css_decls"] = count_css_declarations(tree)
            rec["n_domains"] = count_unique_domains(tree)
            rec["parse_ok"] = True
        except Exception:
            # Битый HTML в корпусе — ожидаемое явление, а не сбой скрипта: доля
            # непарсящихся страниц и есть одна из измеряемых величин. Исход
            # записан в parse_ok, по нему считается сводка, поэтому глушим молча.
            pass
        records.append(rec)
        if len(records) % 500 == 0:
            print(f"  {len(records)}...", flush=True)

    df = pd.DataFrame(records)
    ok = df[df.parse_ok]
    print(f"\nсобрано: {len(df)}, распарсено: {int(df.parse_ok.sum())}")
    cols = ["html_tokens", "html_chars", "dom_nodes", "css_decls", "n_domains"]
    summ = ok[cols].agg(["mean", "median", "std", "min", "max"]).T
    summ = summ.join(ok[cols].quantile([0.9, 0.99]).T.rename(columns={0.9: "p90", 0.99: "p99"}))
    pd.set_option("display.width", 140)
    print(summ.round(1))
    tok = sorted(ok.html_tokens.tolist())
    print(f"\nТокены: median={tok[len(tok)//2]}, p99={tok[int(len(tok)*0.99)]}, max={tok[-1]}")
    print(f"max_length (p99→64): {recommend_max_length(tok)}")
    print("\n=== 7 метрик ===")
    print(f"1. Примеров (статья): {PAPER_COUNT:,}")
    print(f"3. Код токены mean/median/p99: {ok.html_tokens.mean():.0f}/{ok.html_tokens.median():.0f}/{tok[int(len(tok)*0.99)]}")
    print(f"4. DOM mean/median: {ok.dom_nodes.mean():.1f}/{ok.dom_nodes.median():.0f}")
    print(f"6. CSS декл mean/median: {ok.css_decls.mean():.1f}/{ok.css_decls.median():.0f}")
    print(f"7. Домены/стр mean: {ok.n_domains.mean():.2f}")
    # Рядом с обзорами (Data/eda), а не рядом со скриптом: на csv ссылается
    # datasets_overview.md, и он лежит под git именно там.
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web2code_stream_result.csv")
    ok.to_csv(out, index=False)
    print(f"\nсырые строки: {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
