#!/usr/bin/env python3
"""hold_gpus.py — занять память карт, пока прогон до них ещё не дошёл.

Зачем: между стартом оркестратора и стартом обучения проходит время (сборка
датасета — десятки минут), карты в этот момент пустые, и их успевает забрать
сосед. Держатель занимает память сразу и отпускает её ровно тогда, когда
оркестратор доходит до фазы обучения.

Отпускает по маркеру: строка MARKER_TEXT появилась в файле MARKER_FILE.
Оркестратор печатает её ДО своей проверки занятости карт, так что порядок
такой: маркер -> держатель освобождает -> wait_for_gpus видит карты пустыми.

    GPU_HOLD_GB=75 MARKER_FILE=... MARKER_TEXT=... python hold_gpus.py

MAX_HOLD_MIN — предохранитель: если оркестратор упал и маркера не будет
никогда, держатель не должен держать карты вечно.
"""
import os
import time

import torch

HOLD_GB = float(os.environ.get("GPU_HOLD_GB", 70))
MARKER_FILE = os.environ.get("MARKER_FILE", "")
MARKER_TEXT = os.environ.get("MARKER_TEXT", "")
MAX_HOLD_MIN = float(os.environ.get("MAX_HOLD_MIN", 360))
POLL_S = 5


def main():
    n = torch.cuda.device_count()
    bufs = []
    for i in range(n):
        free, total = torch.cuda.mem_get_info(i)
        # Не пытаемся взять больше, чем реально свободно: чужой процесс на карте
        # важнее нашего резерва, падать с OOM здесь незачем.
        take = min(HOLD_GB * 1024**3, free - 2 * 1024**3)
        if take <= 0:
            print(f"[hold] cuda:{i} занята кем-то ({free/1024**3:.1f} ГБ свободно) — пропускаю", flush=True)
            continue
        bufs.append(torch.empty(int(take), dtype=torch.uint8, device=f"cuda:{i}"))
        print(f"[hold] cuda:{i}: занято {take/1024**3:.1f} ГБ", flush=True)

    deadline = time.time() + MAX_HOLD_MIN * 60
    print(f"[hold] держу до маркера '{MARKER_TEXT}' в {MARKER_FILE} "
          f"(предохранитель {MAX_HOLD_MIN:.0f} мин)", flush=True)
    while time.time() < deadline:
        try:
            with open(MARKER_FILE, encoding="utf-8", errors="replace") as f:
                if MARKER_TEXT in f.read():
                    print("[hold] маркер найден — освобождаю карты", flush=True)
                    break
        except OSError:
            pass
        time.sleep(POLL_S)
    else:
        print("[hold] предохранитель сработал — освобождаю карты", flush=True)

    del bufs
    torch.cuda.empty_cache()
    print("[hold] карты свободны", flush=True)


if __name__ == "__main__":
    main()
