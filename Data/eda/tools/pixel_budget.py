#!/usr/bin/env python3
"""pixel_budget.py — оценка бюджета MIN/MAX_PIXELS (Qwen-VL) vs читаемость скриншота.

Контекст (что такое min/max pixels):
  Процессор Qwen-VL (`AutoProcessor(min_pixels=..., max_pixels=...)`) прогоняет
  каждый скриншот через `smart_resize`: держит ПЛОЩАДЬ картинки в [MIN,MAX] пикселей,
  сохраняя пропорции, стороны кратны factor = patch_size*merge_size (=32 у Qwen3-VL).
  Значит MAX_PIXELS — это потолок разрешения, в котором модель ВИДИТ страницу.
  Мы рендерим HTML при ширине RENDER_WIDTH=1280, высота — по контенту. Высокая
  страница 1280xH с площадью > MAX_PIXELS ужимается => падает эфф.ширина => мельчает
  текст. Ниже ~8-11px кегля растровый текст перестаёт читаться моделью.

⚠ Константы НЕ свои: берутся из `Data/converters/common/budget.py`, то есть из тех же
переменных окружения, что читает `SFT/train/formatting.py`. Раньше здесь стоял свой литерал
`MAX_PIXELS = 1280*32*32` (1.31 Мп), и скрипт, который специально существует ради ответа
«какой бюджет нам нужен», печатал таблицу для бюджета, отменённого ещё в Tier A.

Использование:
    python pixel_budget.py                 # таблица читаемости + нужные MAX_PIXELS
    python pixel_budget.py <parquet-glob>  # + реальное распределение высот корпуса
    python pixel_budget.py --max-pixels 3932160   # прикинуть другой потолок
"""
import argparse
import glob
import math
import os
import struct
import sys

# Пролог доступа к общему ядру — тот же, что в конвертерах (см. converters/common/__init__.py).
_CONV = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "converters")
if _CONV not in sys.path:
    sys.path.insert(0, _CONV)

from common.budget import (MAX_PIXELS, MIN_PIXELS, PATCH as FACTOR,  # noqa: E402
                           RENDER_WIDTH)
READABLE_COMFORT = 11.0          # эвристика: комфортный кегль после ужатия, px
READABLE_EDGE = 8.0              # эвристика: край читаемости, px


def smart_resize(w, h, factor=FACTOR, minp=MIN_PIXELS, maxp=MAX_PIXELS):
    """Упрощённый Qwen smart_resize: площадь в [min,max], стороны кратны factor."""
    h_ = max(factor, round(h / factor) * factor)
    w_ = max(factor, round(w / factor) * factor)
    if w_ * h_ > maxp:
        beta = math.sqrt((w * h) / maxp)
        h_ = max(factor, math.floor(h / beta / factor) * factor)
        w_ = max(factor, math.floor(w / beta / factor) * factor)
    elif w_ * h_ < minp:
        beta = math.sqrt(minp / (w * h))
        h_ = math.ceil(h * beta / factor) * factor
        w_ = math.ceil(w * beta / factor) * factor
    return w_, h_


def font_after(H, base_px=16, maxp=MAX_PIXELS):
    """Во что превращается текст base_px на странице RENDER_WIDTH x H после ужатия."""
    w2, _ = smart_resize(RENDER_WIDTH, H, maxp=maxp)
    return base_px * w2 / RENDER_WIDTH


def pct(vals, p):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(round(p / 100 * (len(vals) - 1))))]


def readability_table(maxp=MAX_PIXELS):
    print("Страница %dxH, MAX_PIXELS=%d (%.2f Мпикс):" % (RENDER_WIDTH, maxp, maxp / 1e6))
    print("  H(рендер)  ужатие  эфф.ширина  текст16px  визтокенов  вердикт")
    for H in [1024, 1280, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384]:
        w2, h2 = smart_resize(RENDER_WIDTH, H, maxp=maxp)
        f = 16 * w2 / RENDER_WIDTH
        verdict = "OK" if f >= READABLE_COMFORT else ("край" if f >= READABLE_EDGE else "НЕЧИТАЕМО")
        print("  %6d    x%.3f  %6dpx   %5.1fpx   %6d      %s"
              % (H, w2 / RENDER_WIDTH, w2, f, (w2 // FACTOR) * (h2 // FACTOR), verdict))


def needed_maxpixels(target_font=12.8, base_px=16):
    """Какой MAX_PIXELS нужен, чтобы текст base_px держал >= target_font при 1280xH."""
    scale = target_font / base_px
    print("\nЧтобы текст16px оставался >= %.1fpx (эфф.ширина>=%d):" % (target_font, int(RENDER_WIDTH * scale)))
    for H in [2048, 3072, 4096, 6144, 8192]:
        P = scale ** 2 * RENDER_WIDTH * H
        print("  H=%5d -> MAX_PIXELS>=%d (%.2f Мпикс, x%.1f от текущего), ~%d визтокенов"
              % (H, P, P / 1e6, P / MAX_PIXELS, P / (FACTOR * FACTOR)))


def _img_size(b):
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", b[16:24])
    if b[:2] == b"\xff\xd8":
        i, n = 2, len(b)
        while i < n:
            if b[i] != 0xFF:
                i += 1
                continue
            m = b[i + 1]
            if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
                h, w = struct.unpack(">HH", b[i + 5:i + 9])
                return w, h
            i += 2 + struct.unpack(">H", b[i + 2:i + 4])[0]
    return None


def corpus_report(parquet_glob, image_col="image", nfiles=4):
    import pyarrow.parquet as pq
    ws, hs = [], []
    for f in sorted(glob.glob(parquet_glob))[:nfiles]:
        t = pq.read_table(f)
        if image_col not in t.column_names:
            print("нет колонки %r; есть: %s" % (image_col, t.column_names))
            return
        for v in t.column(image_col).to_pylist():
            b = v.get("bytes") if isinstance(v, dict) else None
            if not b:
                continue
            s = _img_size(b)
            if s:
                ws.append(s[0])
                hs.append(s[1])
    if not ws:
        print("картинок не найдено")
        return
    rr = [RENDER_WIDTH * h / w for w, h in zip(ws, hs)]  # высота ре-рендера @1280
    print("\nКорпус: n=%d скриншотов" % len(ws))
    print("  исходная  W p50=%d p95=%d | H p50=%d p95=%d p99=%d max=%d"
          % (pct(ws, 50), pct(ws, 95), pct(hs, 50), pct(hs, 95), pct(hs, 99), max(hs)))
    print("  ре-рендер@1280 H: p50=%d p90=%d p95=%d p99=%d" % (pct(rr, 50), pct(rr, 90), pct(rr, 95), pct(rr, 99)))
    print("  эфф.текст16px:    p50=%.1f p90=%.1f p95=%.1f px" % (font_after(pct(rr, 50)), font_after(pct(rr, 90)), font_after(pct(rr, 95))))
    for thr, lab in [(READABLE_COMFORT, "комфорт"), (READABLE_EDGE, "край")]:
        share = 100 * sum(1 for H in rr if font_after(H) >= thr) / len(rr)
        print("  доля читаемых (>=%.0fpx, %s): %.1f%%" % (thr, lab, share))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("glob", nargs="?", default=None,
                    help="маска parquet-шардов корпуса: добавит реальное распределение высот")
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS,
                    help="прикинуть ДРУГОЙ потолок (по умолчанию — рабочий из common/budget.py)")
    args = ap.parse_args()
    readability_table(args.max_pixels)
    needed_maxpixels()
    if args.glob:
        corpus_report(args.glob)


if __name__ == "__main__":
    main()
