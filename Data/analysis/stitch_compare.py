#!/usr/bin/env python3
"""
Склеить сравнение эталон | генерация модели для каждого сэмпла бенча.

Бенч сохраняет examples/batch_XXXX/sample_XXXXX/{ref.png, pred.png}. Скрипт
кладёт их рядом (эталон слева, генерация справа, подпись сверху) в
compare_side_by_side/ и делает одну обзорную «контактную» сетку миниатюр.

Запуск (в любом образе с PIL, напр. sft):
  docker run --rm -v /mnt/storage-1:/storage --entrypoint /opt/venv/bin/python sft \
    /storage/.../stitch_compare.py \
      --examples /storage/Screenshot2Code/checkpoints_exps/d2c-ctx32k-base/examples \
      --out /storage/Screenshot2Code/checkpoints_exps/d2c-ctx32k-base/compare
"""
import argparse
import os
from pathlib import Path
from PIL import Image, ImageDraw

LABEL_H = 28


def side_by_side(ref_path, pred_path, out_path, max_h=1600):
    ref = Image.open(ref_path).convert("RGB")
    pred = Image.open(pred_path).convert("RGB")
    # ужать высокие страницы до max_h для читаемости
    for im_name in ("ref", "pred"):
        pass
    def fit(im):
        if im.height > max_h:
            w = int(im.width * max_h / im.height)
            im = im.resize((w, max_h))
        return im
    ref, pred = fit(ref), fit(pred)
    h = max(ref.height, pred.height)
    w = ref.width + pred.width + 20
    canvas = Image.new("RGB", (w, h + LABEL_H), "white")
    d = ImageDraw.Draw(canvas)
    d.text((5, 6), "ЭТАЛОН (Design2Code)", fill="black")
    d.text((ref.width + 25, 6), "БАЗА Qwen3.5-4B", fill="black")
    canvas.paste(ref, (0, LABEL_H))
    canvas.paste(pred, (ref.width + 20, LABEL_H))
    canvas.save(out_path)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid-cols", type=int, default=5)
    ap.add_argument("--thumb-w", type=int, default=360)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pairs = []
    for root, _, files in os.walk(args.examples):
        if "ref.png" in files and "pred.png" in files:
            pairs.append((Path(root) / "ref.png", Path(root) / "pred.png",
                          Path(root).name))
    pairs.sort(key=lambda x: x[2])
    print(f"нашёл {len(pairs)} пар")

    thumbs = []
    for ref, pred, name in pairs:
        try:
            canvas = side_by_side(ref, pred, out / f"{name}.png")
            t = canvas.copy()
            t.thumbnail((args.thumb_w, args.thumb_w * 3))
            thumbs.append(t)
        except Exception as e:  # noqa: BLE001
            print(f"  пропуск {name}: {e}")

    # контактная сетка
    if thumbs:
        cols = args.grid_cols
        rows = (len(thumbs) + cols - 1) // cols
        cw = max(t.width for t in thumbs) + 8
        ch = max(t.height for t in thumbs) + 8
        grid = Image.new("RGB", (cols * cw, rows * ch), "white")
        for i, t in enumerate(thumbs):
            grid.paste(t, ((i % cols) * cw + 4, (i // cols) * ch + 4))
        grid.save(out / "_overview_grid.png")
        print(f"обзорная сетка -> {out / '_overview_grid.png'}")
    print(f"отдельные сравнения -> {out}/*.png")


if __name__ == "__main__":
    main()
