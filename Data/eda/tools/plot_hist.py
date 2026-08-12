#!/usr/bin/env python3
"""matplotlib-гистограммы длины кода (токены). 3 реальных датасета + WebUI (3 зума).

Кладёт две картинки в Data/eda/examples/ — на них ссылается datasets_overview.md.
Требует локально собранный `Data/websight_drafting_pilot` и кэш WebCode2M в
~/.cache/huggingface; без них падает с внятным сообщением.

    .venv/bin/python Data/eda/tools/plot_hist.py
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2]          # …/Data
OUT = str(DATA / "eda" / "examples")
ACC = "#2f8f9a"; GRID = "#9aa5ab"; TXT = "#7a848a"; BUD = "#c0503f"

# Токенайзер прибит к ревизии: длины в токенах — это числа в datasets_overview.md,
# и они обязаны воспроизводиться, а не плыть вместе с обновлением модели на хабе.
TOKENIZER_ID = "Qwen/Qwen3.5-9B"
TOKENIZER_REV = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"

WEBCODE2M_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--xcodemind--webcode2m_purified/snapshots/*/data/*.parquet")
SAMPLE = 2000


def make_tlen(tok):
    """Длины списка текстов в токенах. Батчами по 64 — токенайзер на длинных
    страницах иначе съедает память."""
    def tlen(texts):
        out = []
        for i in range(0, len(texts), 64):
            out += [len(x) for x in tok(texts[i:i + 64], add_special_tokens=False)["input_ids"]]
        return np.array(out)
    return tlen


def load_lengths():
    """(websight, webcode2m) — длины таргетов в токенах по выборке SAMPLE."""
    from transformers import AutoTokenizer
    from datasets import load_from_disk
    import pyarrow.parquet as pq

    tlen = make_tlen(AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REV))

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


def webui_lengths():
    """WebUI восстанавливаем из перцентилей ноутбука — сырых данных локально нет.

    Лог-интерполяция обратной CDF: между перцентилями плотность спадает
    естественно (право-скошенное распределение), а линейная дала бы плоское плато.
    """
    q = [0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0]
    v = [80, 3052, 39366, 63736, 171851, 250454, 363819]
    u = np.random.RandomState(0).rand(20000)
    return np.exp(np.interp(u, q, np.log(v)))


def style(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)


def hist(ax, data, xmax, title, budget=True):
    d = data[data <= xmax]
    ax.hist(d, bins=40, range=(0, xmax), color=ACC, edgecolor="white", linewidth=0.4)
    if budget and xmax >= 6000:
        ax.axvline(6000, color=BUD, ls="--", lw=1.3)
        ax.text(6000, ax.get_ylim()[1] * 0.92, " бюджет ~6k", color=BUD, fontsize=9, va="top")
    ax.set_title(title, fontsize=12, color="#4a555b", pad=8, loc="left")
    ax.set_xlabel("длина кода, токены", fontsize=9)
    style(ax)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "text.color": TXT, "axes.labelcolor": TXT, "xtick.color": TXT, "ytick.color": TXT,
        "axes.edgecolor": GRID, "font.size": 11, "figure.dpi": 150,
    })

    ws, wc = load_lengths()
    webui = webui_lengths()
    os.makedirs(OUT, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    hist(axes[0], ws, 6000, f"WebSight (drafting)   med {int(np.median(ws))} · p99 {int(np.percentile(ws,99))}")
    hist(axes[1], wc, 12000, f"WebCode2M   med {int(np.median(wc))} · p99 {int(np.percentile(wc,99))}")
    fig.tight_layout()
    fig.savefig(f"{OUT}/hist_datasets.png", transparent=True, bbox_inches="tight")
    plt.close(fig)

    p90, p99, mx = 39366, 171851, 363819
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4))
    hist(axes[0], webui, mx, "WebUI · полный (0–max 364k)", budget=False)
    hist(axes[1], webui, p99, "WebUI · 0–p99 (172k)", budget=False)
    hist(axes[2], webui, p90, "WebUI · 0–p90 (39k) — тело без хвоста")
    for ax in axes:
        ax.axvline(6000, color=BUD, ls="--", lw=1.0, alpha=0.7)
    fig.suptitle("WebUI: med 3052, но p90 39k, p99 172k — хвост уходит за шкалу",
                 color="#4a555b", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{OUT}/hist_webui.png", transparent=True, bbox_inches="tight")
    plt.close(fig)

    print("saved hist_datasets.png, hist_webui.png")
    print(f"WebSight n={len(ws)} med={int(np.median(ws))}")
    print(f"WebCode2M n={len(wc)} med={int(np.median(wc))} p99={int(np.percentile(wc,99))}")


if __name__ == "__main__":
    main()
