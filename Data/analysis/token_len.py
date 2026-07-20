"""Токенная длина кода для анализа датасетов. `compute_max_length` — из SFT-трека."""

import math

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

TEXT_FIELDS = ("current_html", "target_html", "instruction")


def count_tokens(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    """Длина одного текста в токенах (без спец-токенов — чистая длина кода)."""
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def recommend_max_length(
    token_lengths,
    quantile: float = 0.99,
    round_to: int = 64,
    image_token_budget: int = 0,
) -> int:
    """Ориентир `max_length` по готовому списку длин (в EDA считаем их в цикле)."""
    s = sorted(token_lengths)
    if not s:
        return image_token_budget
    idx = min(math.ceil(len(s) * quantile) - 1, len(s) - 1)
    return math.ceil(s[idx] / round_to) * round_to + image_token_budget


def compute_max_length(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    quantile: float = 0.99,
    round_to: int = 64,
    text_fields: tuple[str, ...] = TEXT_FIELDS,
    num_proc: int = 4,
    image_token_budget: int = 0,
) -> int:
    """Токенный `max_length` по датасету формата контракта (версия SFT-трека).

    NB: считает НЕ идентично `recommend_max_length`, которым пользуется EDA —
    здесь `add_special_tokens=True` и поля склеены без разделителя, там
    `add_special_tokens=False` по одному полю. Расхождение в пару токенов;
    держать в голове при сверке чисел EDA ↔ SFT.
    """
    lengths = dataset.map(
        lambda example: {
            "_len": len(
                tokenizer(
                    "".join(example[f] for f in text_fields), add_special_tokens=True
                )["input_ids"]
            )
        },
        num_proc=num_proc,
    )["_len"]
    sorted_lengths = sorted(lengths)
    idx = min(
        int(math.ceil(len(sorted_lengths) * quantile)) - 1, len(sorted_lengths) - 1
    )
    return math.ceil(sorted_lengths[idx] / round_to) * round_to + image_token_budget
