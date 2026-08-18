"""Схема сэмпла — контракт §2 (`SFT/DATA_FORMAT_CONTRACT.md`).

Одна на все конвертеры: разъедься схема между источниками, и солянка (`converters/mix/`)
не собралась бы `concatenate_datasets`.
"""
from datasets import Features, Image, Sequence, Value

FEATURES = Features({
    "task_type":    Value("string"),
    "images":       Sequence(Image()),
    "current_html": Value("string"),
    "target_html":  Value("string"),
    "instruction":  Value("string"),
})
