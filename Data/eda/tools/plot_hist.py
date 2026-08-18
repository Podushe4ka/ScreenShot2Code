#!/usr/bin/env python3
"""matplotlib-гистограммы длины кода (токены).

Что рисует:
  * `hist_datasets.png` — WebSight (боевой drafting-таргет) и WebCode2M, по РЕАЛЬНЫМ данным;
  * `hist_webui.png`    — WebUI ПОСЛЕ конвертера, только если дать `--webui-manifest`.

⚠ ПОЧЕМУ У WebUI ОТДЕЛЬНОЕ УСЛОВИЕ. Прежняя версия этого скрипта рисовала WebUI
из воздуха: брала семь перцентилей сырого источника (p50 3052 … p99 171 851 … max 363 819),
лог-интерполяцией разворачивала их в 20 000 «наблюдений» и строила по ним гистограмму —
без единой пометки на картинке, что это реконструкция, а не замер. Хуже того, сами
перцентили к тому моменту были ОТОЗВАНЫ: хвост давала не вёрстка, а отдельная колонка
`css` со стайлшитом всего сайта, и он снимается tree-shaking'ом (`converters/webui/`,
14 августа). То есть картинка иллюстрировала опровергнутое утверждение.

Теперь так нельзя: либо есть манифест конвертера с реальными `tokens_code`, либо картинки
нет и скрипт говорит, какого файла ему не хватает.

Токенайзер — тот же, что у мастер-таблицы `datasets_overview.md`
(`converters/websight/convert_lib.py: TOKENIZER_ID_DEFAULT`), иначе подписи под картинкой
разойдутся с числами в тексте.

    .venv/bin/python Data/eda/tools/plot_hist.py
    .venv/bin/python Data/eda/tools/plot_hist.py --webui-manifest /path/stage_webui/manifest.jsonl
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2]          # …/Data
OUT = str(DATA / "eda" / "examples")
ACC = "#2f8f9a"; GRID = "#9aa5ab"; TXT = "#7a848a"; BUD = "#c0503f"

# Токенайзер берём из конвертера, а не отдельным литералом: числа под картинкой обязаны
# сходиться с мастер-таблицей, а она снята этим же токенайзером.
sys.path.insert(0, str(DATA / "converters" / "websight"))
from convert_lib import TOKENIZER_ID_DEFAULT, qwen_image_tokens  # noqa: E402

WEBCODE2M_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--xcodemind--webcode2m_purified/snapshots/*/data/*.parquet")
SAMPLE = 2000

# Бюджет кода в рабочем окне: окно минус потолок картинки минус накладные шаблона.
WINDOW = 16384
CODE_BUDGET = WINDOW - qwen_image_tokens(1280, 10 ** 6) - 160


def make_tlen(tok):
    """Длины списка текстов в токенах. Батчами по 64 — токенайзер на длинных
    страницах иначе съедает память."""
    def tlen(texts):
        out = []
        for i in range(0, len(texts), 64):
            out += [len(x) for x in tok(texts[i:i + 64], add_special_tokens=False)["input_ids"]]
        return np.array(out)
    return tlen


def load_lengths(tokenizer_id, revision=None):
    """(websight, webcode2m) — длины таргетов в токенах по выборке SAMPLE."""
    from transformers import AutoTokenizer
    from datasets import load_from_disk
    import pyarrow.parquet as pq

    tlen = make_tlen(AutoTokenizer.from_pretrained(tokenizer_id, revision=revision))

    pilot = DATA / "websight_drafting_pilot"
    if not pilot.exists():
        sys.exit(f"нет собранного набора {pilot} — соберите конвертером websight")
    ds = load_from_disk(str(pilot))
    ws = tlen([ds[i]["target_html"] for i in range(min(SAMPLE, len(ds)))])

    files = sorted(glob.glob(WEBCODE2M_GLOB))
    if not files:
        sys.exit(f"нет кэша WebCode2M по маске {WEBCODE2M_GLOB}")
    texts = []
    for f in files:
        for v in pq.read_table(f, columns=["text"]).column("text").to_pylist():
            if v:
                texts.append(v)
            if len(texts) >= SAMPLE:
                break
        if len(texts) >= SAMPLE:
            break
    return ws, tlen(texts)


def webui_lengths(manifest_path):
    """Длины кода принятых страниц WebUI — из манифеста конвертера, БЕЗ реконструкций.

    Манифест пишет `converters/webui/convert_parallel.py`: строка на страницу, поля
    `status` и `tokens_code` (плюс `tokens_total` = код + картинка). Берём только
    `status == "ok"` — отбракованные в набор не попадают и в распределение не входят.
    """
    code, total = [], []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            if rec.get("tokens_code"):
                code.append(rec["tokens_code"])
            if rec.get("tokens_total"):
                total.append(rec["tokens_total"])
    if not code:
        sys.exit(f"в {manifest_path} нет страниц со status=ok и tokens_code — "
                 f"прогони конвертер с фазой токенов (--finalize)")
    return np.array(code), np.array(total)


def style(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)


def hist(ax, data, xmax, title, budget=True):
    d = data[data <= xmax]
    ax.hist(d, bins=40, range=(0, xmax), color=ACC, edgecolor="white", linewidth=0.4)
    if budget and xmax >= CODE_BUDGET:
        ax.axvline(CODE_BUDGET, color=BUD, ls="--", lw=1.3)
        ax.text(CODE_BUDGET, ax.get_ylim()[1] * 0.92, f" бюджет кода {CODE_BUDGET}",
                color=BUD, fontsize=9, va="top")
    ax.set_title(title, fontsize=12, color="#4a555b", pad=8, loc="left")
    ax.set_xlabel("длина кода, токены", fontsize=9)
    style(ax)


def q(a, p):
    return int(np.percentile(a, p))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--webui-manifest", default=None,
                    help="manifest.jsonl конвертера WebUI (converters/webui). Без него "
                         "hist_webui.png НЕ рисуется — выдумывать распределение нельзя")
    ap.add_argument("--tokenizer", default=TOKENIZER_ID_DEFAULT,
                    help="должен совпадать с токенайзером мастер-таблицы")
    ap.add_argument("--revision", default=None,
                    help="ревизия токенайзера на хабе (пин для воспроизводимости чисел)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "text.color": TXT, "axes.labelcolor": TXT, "xtick.color": TXT, "ytick.color": TXT,
        "axes.edgecolor": GRID, "font.size": 11, "figure.dpi": 150,
    })

    ws, wc = load_lengths(args.tokenizer, args.revision)
    os.makedirs(OUT, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    hist(axes[0], ws, 6000, f"WebSight (drafting)   med {q(ws,50)} · p99 {q(ws,99)}")
    hist(axes[1], wc, 12000, f"WebCode2M   med {q(wc,50)} · p99 {q(wc,99)}")
    fig.tight_layout()
    fig.savefig(f"{OUT}/hist_datasets.png", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"saved hist_datasets.png (токенайзер {args.tokenizer})")
    print(f"WebSight  n={len(ws)} med={q(ws,50)} p99={q(ws,99)} max={int(ws.max())}")
    print(f"WebCode2M n={len(wc)} med={q(wc,50)} p99={q(wc,99)} max={int(wc.max())}")

    if not args.webui_manifest:
        print("\nhist_webui.png НЕ нарисована: нужен --webui-manifest "
              "<stage_webui>/manifest.jsonl.\nРаспределение WebUI после конвертации "
              "восстанавливать по перцентилям сырого источника нельзя — они относятся\n"
              "к колонке `css`, которой в конвертированном наборе уже нет.")
        return

    code, total = webui_lengths(args.webui_manifest)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    hist(axes[0], code, max(6000, q(code, 99)),
         f"WebUI после конвертера · код   med {q(code,50)} · p99 {q(code,99)}")
    if len(total):
        hist(axes[1], total, max(6000, q(total, 99)),
             f"WebUI · код + картинка   med {q(total,50)} · p99 {q(total,99)}")
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "нет tokens_total в манифесте", ha="center", va="center")
    fig.suptitle(f"WebUI после tree-shaking, n={len(code)} принятых страниц",
                 color="#4a555b", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{OUT}/hist_webui.png", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved hist_webui.png")
    print(f"WebUI n={len(code)} код med={q(code,50)} p95={q(code,95)} "
          f"p99={q(code,99)} max={int(code.max())}")
    if len(total):
        print(f"WebUI код+картинка p99={q(total,99)} max={int(total.max())} "
              f"(окно {WINDOW})")


if __name__ == "__main__":
    main()
