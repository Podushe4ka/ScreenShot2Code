#!/usr/bin/env python3
"""make_split.py — разрезать готовый drafting-датасет на train/validation.

Зачем: конвертер (`drafting/convert_parallel.py`, `webcode2m/convert_parallel.py`) отдаёт
единый `Dataset`, а SFT-трек считает eval_loss по отдельному сплиту. Здесь финальный шаг:
shuffle -> train_test_split -> `DatasetDict{train, validation}` -> save_to_disk.

ПОЧЕМУ ПОСЛЕ ДЕДУПА, А НЕ ДО. Точные и near-dup дубли режутся ещё в фазе 1 конвертера
(`collect_candidates`: SHA1 по HTML + average-hash по картинке). Если бы дубли дожили до
разреза, почти-копии расползлись бы по ОБОИМ сплитам — модель «видела» бы валидацию на train,
и eval_loss оказался бы занижен. Поэтому вход сюда — уже дедуплицированный арро; мы только режем.
Приёмка ниже проверяет пересечение `target_html` между сплитами как дешёвую страховку.

ЭТО НЕ ДЕКОТАМИНАЦИЯ. Held-out бенч (Design2Code «никогда в train», PLAN §4a) держится вне
этого датасета целиком. Здешний validation — случайные N% для eval_loss во время обучения,
он бенч не заменяет.

    python make_split.py IN OUT [--val-frac 0.05] [--seed 42]
"""
import argparse

from datasets import DatasetDict, load_from_disk


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inp", help="путь к дедуплицированному датасету (load_from_disk)")
    ap.add_argument("out", help="куда сохранить DatasetDict{train, validation}")
    ap.add_argument("--val-frac", type=float, default=0.05, help="доля в validation (0.05 = 5%)")
    ap.add_argument("--seed", type=int, default=42, help="общий сид shuffle+split (воспроизводимость)")
    args = ap.parse_args()

    ds = load_from_disk(args.inp)
    if isinstance(ds, DatasetDict):
        raise SystemExit(f"вход уже DatasetDict ({list(ds)}) — повторный разрез не нужен")

    n = len(ds)
    if n < 2:
        raise SystemExit(f"слишком мало сэмплов для разреза: {n}")

    # shuffle ДО разреза: у конвертера порядок = порядок стрима источника. Без перемешивания
    # validation зачерпнёт хвост одного распределения (последние страницы стрима), а не срез.
    split = ds.shuffle(seed=args.seed).train_test_split(test_size=args.val_frac, seed=args.seed)
    out = DatasetDict({"train": split["train"], "validation": split["test"]})  # ключ test -> validation
    out.save_to_disk(args.out)

    # приёмка: оба сплита непусты, грузятся, не пересекаются по target_html.
    reloaded = load_from_disk(args.out)
    assert set(reloaded) == {"train", "validation"}, f"неожиданные сплиты: {list(reloaded)}"
    assert len(reloaded["train"]) and len(reloaded["validation"]), "пустой сплит после разреза"
    tr = {s["target_html"] for s in reloaded["train"]}
    va = {s["target_html"] for s in reloaded["validation"]}
    overlap = len(tr & va)
    print(f"[split] train={len(reloaded['train'])}  validation={len(reloaded['validation'])}  "
          f"(val-frac={args.val_frac}, seed={args.seed}) -> {args.out}")
    if overlap:
        print(f"[split] ⚠ пересечение target_html между сплитами: {overlap} точных совпадений. "
              f"Дедуп во входе неполный — eval_loss будет занижен. Прогони конвертер с near-dup "
              f"или дедуп перед разрезом.")
    else:
        print("[split] пересечений target_html между сплитами нет ✓ (near-dup проверка — на стороне конвертера)")


if __name__ == "__main__":
    main()
