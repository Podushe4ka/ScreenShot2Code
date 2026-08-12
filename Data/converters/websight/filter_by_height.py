#!/usr/bin/env python3
"""filter_by_height.py — отфильтровать drafting-датасет по высоте скриншота.

⚠ ИСТОРИЧЕСКИЙ. Писался, когда SFT-процессор не ужимал картинки по `max_pixels`:
высокие full-page скрины превышали визуальный бюджет и роняли vision-башню
(mismatch патчей и позиционных эмбеддингов), и резать приходилось по высоте.
Клэмп с тех пор появился — `SFT/train/train_sft.py` передаёт `min_pixels`/`max_pixels`
в `AutoProcessor`, — так что для новых наборов скрипт не нужен: слишком высокая
страница теперь ужимается, а не ломает обучение. Остаётся для воспроизведения
старых наборов и для случая, когда ужатие нежелательно (мелкий текст становится
нечитаемым, см. `Data/eda/tools/pixel_budget.py`).

Порог выбирай по распределению (`height_dist.py`), а не вслепую: при ширине 1280
высота H даёт ~1.25*H визуальных токенов.

Быстро (без ре-рендера): читает готовый датасет, фильтрует по высоте, save_to_disk.

    python filter_by_height.py IN OUT [--max-height 1024]
"""
import argparse

import datasets
from datasets import load_from_disk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", help="путь к датасету (load_from_disk)")
    ap.add_argument("out", help="куда сохранить отфильтрованный")
    ap.add_argument("--max-height", type=int, default=1024,
                    help="макс. высота скриншота в px (по умолчанию 1024)")
    args = ap.parse_args()

    # keep_in_memory + disable_caching: не писать служебный кэш в исходную папку — она
    # часто root-owned (датасет собран Docker'ом от root), запись туда падает PermissionError.
    datasets.disable_caching()
    d = load_from_disk(args.inp)
    n0 = len(d)
    d2 = d.filter(lambda ex: ex["images"][0].size[1] <= args.max_height, keep_in_memory=True)
    d2.save_to_disk(args.out)
    print(f"оставлено {len(d2)}/{n0} (height<={args.max_height}px), "
          f"отброшено {n0 - len(d2)} -> {args.out}")


if __name__ == "__main__":
    main()
