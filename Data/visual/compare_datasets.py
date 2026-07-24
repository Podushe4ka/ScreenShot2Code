#!/usr/bin/env python3
"""
compare_datasets.py — сетка 3x3 скриншотов из трёх датасетов, чтобы глазами
увидеть разницу. По 3 сэмпла из каждого (WebSight / WebCode2M / WebUI),
берём поле `image`, кропим в квадрат (центр) для удобного просмотра.

Стриминг — без скачивания. На выходе PNG + окно (если есть дисплей).
    pip install datasets pillow matplotlib
    python compare_datasets.py                 # 3x3, сохранит datasets_compare.png
    python compare_datasets.py -n 4 -o out.png # по 4 (сетка 3xN)
"""
import argparse

import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import Image as PILImage

# (метка, путь, split, поле картинки)
DATASETS = [
    ("WebSight",  "HuggingFaceM4/WebSight",       "train", "image"),
    ("WebCode2M", "xcodemind/webcode2m_purified", "train", "image"),
    ("WebUI",     "ronantakizawa/webui",          "train", "image"),
]
THUMB = 512   # сторона квадратного превью, px


def crop_square(img, size=THUMB):
    """Центральный квадратный кроп + ресайз к size×size."""
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((size, size))


def sample_images(path, split, n, field):
    """Первые n примеров с непустой картинкой (стриминг, без скачивания)."""
    out = []
    for r in load_dataset(path, split=split, streaming=True):
        im = r.get(field)
        if im is None:
            continue
        out.append(crop_square(im))
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=3, help="сколько сэмплов на датасет")
    ap.add_argument("-o", "--out", default="datasets_compare.png")
    args = ap.parse_args()

    rows = []
    for label, path, split, field in DATASETS:
        print(f"беру {args.n} из {label} ({path})...")
        rows.append((label, sample_images(path, split, args.n, field)))

    fig, axes = plt.subplots(len(DATASETS), args.n, figsize=(args.n * 3, len(DATASETS) * 3.2))
    if args.n == 1:
        axes = [[ax] for ax in axes]
    for i, (label, imgs) in enumerate(rows):
        for j in range(args.n):
            ax = axes[i][j]
            if j < len(imgs):
                ax.imshow(imgs[j])
            else:
                ax.text(0.5, 0.5, "нет", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(label, fontsize=13, fontweight="bold", rotation=90, labelpad=10)

    fig.suptitle("Сравнение датасетов — по строке на датасет, квадратный кроп", fontsize=13)
    plt.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"сохранено: {args.out}")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
