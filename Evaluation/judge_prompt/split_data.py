"""
split_data.py — делит labels.csv (результат labeler.py) на train (300) и
test (100), стратифицированно по winner (baseline/checkpoint), чтобы доля
baseline-побед была примерно одинаковой в train и test — иначе, если,
скажем, baseline случайно преобладает в train, подобранный на train промпт
может быть настроен под "перекошенное" распределение и не обобщаться на
test.

train используется для подбора промпта/модели судьи (сколько угодно
итераций, подглядывать в него можно свободно). test — только для финального
одного замера в конце, когда промпт уже зафиксирован.
"""

import argparse
import csv
import random
from pathlib import Path

TRAIN_SIZE = 276
TEST_SIZE = 100


def read_labels(path: Path) -> list[dict]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stratified_split(rows: list[dict], train_size: int, test_size: int, seed: int):
    by_class: dict[str, list[dict]] = {}
    for r in rows:
        by_class.setdefault(r["winner"], []).append(r)

    rng = random.Random(seed)
    for cls_rows in by_class.values():
        rng.shuffle(cls_rows)

    total = train_size + test_size
    train, test = [], []
    for cls, cls_rows in by_class.items():
        # Пропорция класса сохраняется отдельно и в train, и в test — доля
        # округляется по каждому классу независимо, остаток (из-за
        # округления) добирается в train, чтоббы test не терял сэмплы из-за
        # округления в меньшую сторону при малом числе классов (тут их
        # всего два, но код не завязан на это число).
        frac = len(cls_rows) / len(rows)
        n_train = round(frac * train_size)
        n_test = round(frac * test_size)
        train.extend(cls_rows[:n_train])
        test.extend(cls_rows[n_train:n_train + n_test])

    rng.shuffle(train)
    rng.shuffle(test)

    if len(train) + len(test) != total:
        print(f"[split_data] Предупреждение: из-за округления получилось "
              f"train={len(train)}, test={len(test)} вместо {train_size}/{test_size}.")

    return train, test


def main():
    parser = argparse.ArgumentParser(description="Стратифицированный train/test split labels.csv")
    parser.add_argument("--labels", type=Path, default=Path("labels.csv"))
    parser.add_argument("--train-out", type=Path, default=Path("labels_train.csv"))
    parser.add_argument("--test-out", type=Path, default=Path("labels_test.csv"))
    parser.add_argument("--train-size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--test-size", type=int, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_labels(args.labels)
    if len(rows) < args.train_size + args.test_size:
        raise SystemExit(
            f"В {args.labels} только {len(rows)} размеченных сэмплов, "
            f"а нужно минимум {args.train_size + args.test_size}. "
            f"Доразметьте остальное через labeler.py."
        )
    if len(rows) > args.train_size + args.test_size:
        print(f"[split_data] Внимание: размечено {len(rows)} сэмплов, "
              f"используются только первые {args.train_size + args.test_size} "
              f"после перемешивания (seed={args.seed}); остальные не попадут "
              f"ни в train, ни в test.")

    train, test = stratified_split(rows, args.train_size, args.test_size, args.seed)

    fieldnames = list(rows[0].keys())
    write_rows(args.train_out, train, fieldnames)
    write_rows(args.test_out, test, fieldnames)

    def class_counts(subset):
        counts: dict[str, int] = {}
        for r in subset:
            counts[r["winner"]] = counts.get(r["winner"], 0) + 1
        return counts

    print(f"[split_data] train: {len(train)} -> {args.train_out}  {class_counts(train)}")
    print(f"[split_data] test:  {len(test)} -> {args.test_out}  {class_counts(test)}")


if __name__ == "__main__":
    main()
