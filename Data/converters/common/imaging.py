"""Хэши и подгонка картинок — общее для near-dup всех конвертеров."""
from PIL import Image as PILImage


def ahash(img, size=8):
    """Average-hash картинки (near-dup) — чистый PIL, без зависимостей."""
    g = img.convert("L").resize((size, size))
    px = list(g.getdata()); avg = sum(px) / len(px)
    return sum(1 << i for i, p in enumerate(px) if p > avg)


def hamming(a, b):
    return bin(a ^ b).count("1")


def fit_to_size(img, size):
    """Привести скриншот к (W,H): паддинг белым + обрезка. Только для смоука SIZE_MODE='pad'."""
    tw, th = size
    img = img.convert("RGB")
    canvas = PILImage.new("RGB", size, (255, 255, 255))
    canvas.paste(img.crop((0, 0, min(img.width, tw), min(img.height, th))), (0, 0))
    return canvas
