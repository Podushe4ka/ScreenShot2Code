#!/usr/bin/env python3
"""height_dist.py — распределение высот скриншотов в drafting-датасете.

Высоты читаются БЫСТРО из PNG-заголовка (IHDR), без декодирования пикселей.
Печатает перцентили, сколько сэмплов переживёт разные пороги height<=T и
грубую гистограмму — чтобы выбрать отсечку осознанно, а не брать 1024 вслепую.

    python height_dist.py <путь-к-датасету (load_from_disk) ИЛИ .arrow>
"""
import os
import struct
import sys
from collections import Counter

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def png_hw(data):
    """(width, height) из PNG-заголовка (сигнатура 8 байт + IHDR: len,'IHDR',w,h)."""
    assert data[:8] == _PNG_SIG, "не PNG"
    return struct.unpack(">II", data[16:24])


def heights_from_disk(path):
    from datasets import Image, Sequence, load_from_disk
    d = load_from_disk(path)
    d = d.cast_column("images", Sequence(Image(decode=False)))   # не декодировать пиксели
    for row in d:
        yield png_hw(row["images"][0]["bytes"])[1]


def heights_from_arrow(path):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    with open(path, "rb") as f:
        data = f.read()
    try:
        table = ipc.open_file(pa.BufferReader(data)).read_all()
    except pa.lib.ArrowInvalid:
        table = ipc.open_stream(pa.BufferReader(data)).read_all()
    for v in table.column("images").to_pylist():
        item = v[0] if isinstance(v, list) else v
        yield png_hw(item["bytes"])[1]


def main():
    if len(sys.argv) != 2:
        sys.exit("использование: python height_dist.py <датасет-dir | .arrow>")
    path = sys.argv[1]
    src = heights_from_disk(path) if os.path.isdir(path) else heights_from_arrow(path)
    hs = sorted(src)
    n = len(hs)
    if not n:
        sys.exit("нет сэмплов")

    def pct(p): return hs[min(n - 1, int(n * p))]
    print(f"сэмплов: {n}")
    print(f"min={hs[0]}  p50={pct(.5)}  p90={pct(.9)}  p95={pct(.95)}  p99={pct(.99)}  max={hs[-1]}")

    print("\nсколько переживёт height<=T:")
    for t in (768, 1024, 1280, 1536, 2048, 3072, 4096, 6144):
        k = sum(1 for h in hs if h <= t)
        print(f"  <= {t:5d}px : {k:6d}  ({100 * k / n:5.1f}%)")

    print("\nгистограмма (бакет 512px):")
    buck = Counter((h // 512) * 512 for h in hs)
    mx = max(buck.values())
    for b in sorted(buck):
        print(f"  {b:5d}-{b + 511:5d} | {'#' * int(40 * buck[b] / mx)} {buck[b]}")


if __name__ == "__main__":
    main()
