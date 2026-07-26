"""Чтение датасета с диска.

Только чтение и отбор по типу задачи. Отбраковка по бюджету токенов живёт в
`data/filtering.py`: loader отвечает за «что лежит на диске», фильтрация — за
«что мы считаем пригодным для обучения», и эти решения меняются независимо.
"""

from datasets import Dataset, load_from_disk

ALLOWED = {"drafting"}  # , "polishing", "editing"


def load_sft_dataset(path: str) -> Dataset:
    dataset = load_from_disk(path)
    dataset_filtered = dataset.filter(lambda ex: ex["task_type"] in ALLOWED)
    if len(dataset_filtered) == 0:
        raise ValueError(
            f"В {path} нет сэмплов с task_type из {sorted(ALLOWED)} "
            f"(всего сэмплов: {len(dataset)})"
        )
    return dataset_filtered
