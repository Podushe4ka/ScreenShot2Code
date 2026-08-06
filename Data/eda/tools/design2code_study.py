#!/usr/bin/env python3
"""
Исследование Design2Code: почему дообучение на WebCode2M ухудшает score.

Что считает:
  1. Распределение длины таргетов в ТОКЕНАХ (Design2Code vs WebCode2M).
  2. Долю таргетов, превышающих лимиты генерации (8k/16k/24k/32k).
  3. Гистограмму длин -> PNG.

Зачем: Design2Code median ~17.5k токенов, WebCode2M ~4.1k. Это объясняет
и провал sanity-overfit (Design2Code не влезал в max_length 16384), и почему
контекст важен для бенча, но не для обучения на WebCode2M.

Запуск (в контейнере sft, где есть transformers/datasets; matplotlib ставится на лету):
  docker run --rm -v /mnt/storage-1:/storage -e HF_HOME=/storage/Screenshot2Code/hf_cache \
    --entrypoint bash sft -c \
    "/opt/venv/bin/pip install -q matplotlib 2>/dev/null; \
     /opt/venv/bin/python /storage/.../design2code_study.py \
       --webcode2m /storage/Screenshot2Code/data/webcode2m_3000_split \
       --out /storage/Screenshot2Code/checkpoints_exps/design2code_lengths.png"
"""
import argparse
import numpy as np


def token_lengths(tokenizer, texts):
    return sorted(len(tokenizer(t, add_special_tokens=False).input_ids) for t in texts)


def summarize(name, lengths, limits=(8192, 16384, 24384, 32768)):
    a = np.array(lengths)
    print(f"\n{name} (n={len(a)})")
    print(f"  median {int(np.median(a))} | p90 {int(np.percentile(a,90))} | "
          f"p99 {int(np.percentile(a,99))} | max {int(a.max())}")
    for lim in limits:
        print(f"  > {lim:5d} ток: {100*(a>lim).mean():.0f}%")
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--design2code", default="SALT-NLP/Design2Code-hf")
    ap.add_argument("--webcode2m", default=None, help="локальный путь (load_from_disk) или пропустить")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="design2code_lengths.png")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from datasets import load_dataset, load_from_disk
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    ds = load_dataset(args.design2code, name="default", split="train", streaming=True)
    d2c = [r["text"] for _, r in zip(range(args.n), ds)]
    a_d2c = summarize("Design2Code (target HTML)", token_lengths(tok, d2c))

    a_wc = None
    if args.webcode2m:
        wc = load_from_disk(args.webcode2m)["train"]
        wc_texts = [wc[i]["target_html"] for i in range(min(args.n, len(wc)))]
        a_wc = summarize("WebCode2M (target HTML)", token_lengths(tok, wc_texts))

    # --- график ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        bins = np.linspace(0, 40000, 60)
        ax.hist(a_d2c, bins=bins, alpha=0.6, label=f"Design2Code (med {int(np.median(a_d2c))})")
        if a_wc is not None:
            ax.hist(a_wc, bins=bins, alpha=0.6, label=f"WebCode2M (med {int(np.median(a_wc))})")
        for lim, c in [(16384, "red"), (32768, "orange")]:
            ax.axvline(lim, color=c, ls="--", lw=1, label=f"лимит {lim}")
        ax.set_xlabel("длина таргета, токены (Qwen)")
        ax.set_ylabel("число страниц")
        ax.set_title("Длина таргетов: Design2Code длиннее WebCode2M в ~4 раза")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out, dpi=120)
        print(f"\nграфик -> {args.out}")
    except ImportError:
        print("\n(matplotlib нет — график пропущен, числа выше)")


if __name__ == "__main__":
    main()
