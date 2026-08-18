#!/usr/bin/env python3
"""make_examples.py — по нескольку примеров-скриншотов из каждого датасета для overview.

Стриминг (без полного скачивания). Верхний кроп (для веб-страниц информативнее
центрального), ширина 360px. Складывает в examples/<label>_<i>.jpg.
Web2Code пропускаем: картинки только в Web2Code_image.zip (~30 ГБ), в стриме — путь.
"""
import os
from datasets import load_dataset

# Каталог примеров живёт рядом с обзорами (Data/eda/examples), а не рядом со
# скриптом: на них ссылается datasets_overview.md.
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples")
os.makedirs(OUT, exist_ok=True)
W = 360
N = 3

DATASETS = [
    ("websight",  "HuggingFaceM4/WebSight",       "train", "image"),
    ("webcode2m", "xcodemind/webcode2m_purified", "train", "image"),
    ("webui",     "ronantakizawa/webui",          "train", "image"),
]


def top_thumb(img, width=W, max_h=520):
    img = img.convert("RGB")
    w, h = img.size
    nh = int(h * width / w)
    img = img.resize((width, nh))
    if nh > max_h:  # верхний кроп — видно шапку страницы
        img = img.crop((0, 0, width, max_h))
    return img


def main():
    for label, path, split, field in DATASETS:
        print(f"[{label}] стримим {path}...", flush=True)
        got = 0
        try:
            for r in load_dataset(path, split=split, streaming=True):
                im = r.get(field)
                if im is None:
                    continue
                try:
                    thumb = top_thumb(im)
                except Exception:
                    continue
                fp = os.path.join(OUT, f"{label}_{got}.jpg")
                thumb.save(fp, quality=82)
                print(f"   -> {fp}  ({im.size[0]}x{im.size[1]})", flush=True)
                got += 1
                if got >= N:
                    break
        except Exception as e:
            print(f"   ! {label}: {e}", flush=True)
    print("готово:", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
