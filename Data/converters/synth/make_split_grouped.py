"""Групповой разрез train/validation по странице-источнику.

Почему НЕ Data/converters/make_split.py. Тот режет случайно
(`shuffle(42).train_test_split(0.05)`), и для WebSight/WebCode2M это верно: там сэмпл
и страница — одно и то же. У нас из ОДНОЙ страницы выходит до семи сэмплов сразу трёх
задач: drafting, шесть editing-правок, две деградации polishing. Все они содержат один
и тот же `target_html`. При случайном разрезе почти-дубли расползаются по обоим сплитам,
модель на валидации отвечает по уже виденному таргету, и eval_loss оказывается занижен —
причём тем сильнее, чем лучше пайплайн размножает сэмплы.

Поэтому режем ГРУППАМИ по `page_id`: страница целиком уходит либо в train, либо в
validation. Внутри этого — стратификация по `task_type`, чтобы редкая задача не пропала
из валидации целиком (требование §1a контракта для случая нескольких задач).
"""

import random
from collections import defaultdict

from datasets import Dataset, DatasetDict

SEED = 42
VAL_FRACTION = 0.05
VAL_CAP = 1000          # §1a контракта: 5%, но не больше ~1000 сэмплов


def grouped_split(ds: Dataset, group_key: str = "page_id",
                  strat_key: str = "task_type", seed: int = SEED,
                  frac: float = VAL_FRACTION, cap: int = VAL_CAP) -> DatasetDict:
    groups = defaultdict(list)
    for i, g in enumerate(ds[group_key]):
        groups[g].append(i)

    # Группы упорядочиваем по составу задач, чтобы стратификация работала на уровне
    # групп: сначала раскладываем по «профилю» (какие task_type есть в группе), затем
    # из каждого профиля берём одинаковую долю в валидацию.
    task_of = ds[strat_key]
    profiles = defaultdict(list)
    for g, idxs in groups.items():
        profile = tuple(sorted({task_of[i] for i in idxs}))
        profiles[profile].append(g)

    target = min(cap, max(1, round(len(ds) * frac)))
    rng = random.Random(seed)
    val_groups = []
    n_val = 0
    for profile in sorted(profiles):
        gs = sorted(profiles[profile])
        rng.shuffle(gs)
        # Доля групп этого профиля пропорциональна его доле в датасете.
        share = sum(len(groups[g]) for g in gs) / len(ds)
        want = max(1, round(target * share))
        for g in gs:
            if n_val >= want:
                break
            val_groups.append(g)
            n_val += len(groups[g])

    val_set = set(val_groups)
    val_idx = [i for g in val_groups for i in groups[g]]
    train_idx = [i for g, idxs in groups.items() if g not in val_set for i in idxs]

    if not train_idx or not val_idx:
        raise ValueError(
            f"разрез вырожден: train={len(train_idx)}, validation={len(val_idx)}. "
            f"Групп всего {len(groups)} — для доли {frac} их слишком мало.")

    return DatasetDict({
        "train": ds.select(sorted(train_idx)),
        "validation": ds.select(sorted(val_idx)),
    })


def assert_no_leak(dd: DatasetDict, group_key: str = "page_id") -> None:
    """Приёмка: ни одна страница не должна встречаться в обоих сплитах."""
    a = set(dd["train"][group_key])
    b = set(dd["validation"][group_key])
    overlap = a & b
    if overlap:
        raise AssertionError(
            f"утечка: {len(overlap)} страниц в обоих сплитах, например "
            f"{sorted(overlap)[:5]}")
