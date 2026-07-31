#!/usr/bin/env python3
"""matplotlib-гистограммы длины кода (токены). 3 реальных датасета + WebUI (3 зума)."""
import csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from datasets import load_from_disk

OUT = "/Users/vyacheslav/Screenshot2Code/ScreenShot2Code/Data/analysis/examples"
ACC = "#2f8f9a"; TAIL = "#c07f27"; GRID = "#9aa5ab"; TXT = "#7a848a"; BUD = "#c0503f"
plt.rcParams.update({
    "text.color": TXT, "axes.labelcolor": TXT, "xtick.color": TXT, "ytick.color": TXT,
    "axes.edgecolor": GRID, "font.size": 11, "figure.dpi": 150,
})

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a")
def tlen(texts):
    out = []
    for i in range(0, len(texts), 64):
        out += [len(x) for x in tok(texts[i:i+64], add_special_tokens=False)["input_ids"]]
    return np.array(out)

# --- реальные данные ---
w2 = []
with open("/Users/vyacheslav/Screenshot2Code/ScreenShot2Code/Data/analysis/web2code_stream_result.csv") as f:
    for r in csv.DictReader(f):
        try: w2.append(int(r["html_tokens"]))
        except: pass
w2 = np.array(w2)

ds = load_from_disk("/Users/vyacheslav/Screenshot2Code/ScreenShot2Code/Data/websight_drafting_pilot")
ws = tlen([ds[i]["target_html"] for i in range(min(2000, len(ds)))])

texts = []
for f in sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--xcodemind--webcode2m_purified/snapshots/*/data/*.parquet"))):
    for v in pq.read_table(f, columns=["text"]).column("text").to_pylist():
        if v: texts.append(v)
        if len(texts) >= 2000: break
    if len(texts) >= 2000: break
wc = tlen(texts)

# --- WebUI: восстановление из перцентилей ноутбука (сырые данные не локально) ---
# лог-интерполяция обратной CDF: между перцентилями плотность спадает естественно
# (право-скошенное распределение), а не даёт плоское плато.
q  = [0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0]
v  = [80, 3052, 39366, 63736, 171851, 250454, 363819]
u = np.random.RandomState(0).rand(20000)
webui = np.exp(np.interp(u, q, np.log(v)))

def style(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)

def hist(ax, data, xmax, title, budget=True):
    d = data[data <= xmax]
    ax.hist(d, bins=40, range=(0, xmax), color=ACC, edgecolor="white", linewidth=0.4)
    if budget and xmax >= 6000:
        ax.axvline(6000, color=BUD, ls="--", lw=1.3)
        ax.text(6000, ax.get_ylim()[1]*0.92, " бюджет ~6k", color=BUD, fontsize=9, va="top")
    ax.set_title(title, fontsize=12, color="#4a555b", pad=8, loc="left")
    ax.set_xlabel("длина кода, токены", fontsize=9)
    style(ax)

# === Фигура 1: три реальных датасета ===
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4))
hist(axes[0], w2, 4000, f"Web2Code   med {int(np.median(w2))} · p99 {int(np.percentile(w2,99))}")
hist(axes[1], ws, 6000, f"WebSight (drafting)   med {int(np.median(ws))} · p99 {int(np.percentile(ws,99))}")
hist(axes[2], wc, 12000, f"WebCode2M   med {int(np.median(wc))} · p99 {int(np.percentile(wc,99))}")
fig.tight_layout()
fig.savefig(f"{OUT}/hist_datasets.png", transparent=True, bbox_inches="tight")
plt.close(fig)

# === Фигура 2: WebUI, три зума ===
p90, p99, mx = 39366, 171851, 363819
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4))
hist(axes[0], webui, mx,  "WebUI · полный (0–max 364k)", budget=False)
hist(axes[1], webui, p99, "WebUI · 0–p99 (172k)", budget=False)
hist(axes[2], webui, p90, "WebUI · 0–p90 (39k) — тело без хвоста")
for ax in axes: ax.axvline(6000, color=BUD, ls="--", lw=1.0, alpha=0.7)
fig.suptitle("WebUI: med 3052, но p90 39k, p99 172k — хвост уходит за шкалу", color="#4a555b", fontsize=12, x=0.01, ha="left")
fig.tight_layout(rect=[0,0,1,0.93])
fig.savefig(f"{OUT}/hist_webui.png", transparent=True, bbox_inches="tight")
plt.close(fig)

print("saved hist_datasets.png, hist_webui.png")
print(f"Web2Code n={len(w2)} med={int(np.median(w2))}")
print(f"WebSight n={len(ws)} med={int(np.median(ws))}")
print(f"WebCode2M n={len(wc)} med={int(np.median(wc))} p99={int(np.percentile(wc,99))}")
