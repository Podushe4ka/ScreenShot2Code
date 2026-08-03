"""
Чтение датасета с диска.
"""

import logging

from datasets import Dataset, DatasetDict, load_from_disk

logger = logging.getLogger(__name__)

ALLOWED = {"drafting"}  # , "polishing", "editing"


def load_sft_dataset(
    path: str, split: str = "train", required: bool = True
) -> Dataset | None:
    """Прочитать сплит датасета и оставить только поддерживаемые задачи.

    `required=False` — вернуть None вместо ошибки, если сплита нет
    """
    data = load_from_disk(path)

    if isinstance(data, DatasetDict):
        if split not in data:
            available = sorted(data.keys())
            if required:
                raise ValueError(
                    f"В {path} нет сплита '{split}'. Доступны: {available}"
                )
            logger.warning(
                "В %s нет сплита '%s' (есть %s) — продолжаем без него.",
                path,
                split,
                available,
            )
            return None
        dataset = data[split]
    else:
        if split != "train":
            if required:
                raise ValueError(
                    f"{path} сохранён без сплитов, сплит '{split}' недоступен."
                )
            logger.warning(
                "%s сохранён без сплитов — сплита '%s' нет, продолжаем без него.",
                path,
                split,
            )
            return None
        dataset = data

    filtered = dataset.filter(lambda ex: ex["task_type"] in ALLOWED)
    if len(filtered) == 0:
        raise ValueError(
            f"В сплите '{split}' ({path}) нет сэмплов с task_type из "
            f"{sorted(ALLOWED)} (всего сэмплов: {len(dataset)})"
        )
    return filtered
