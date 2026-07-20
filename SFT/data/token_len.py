import math
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

TEXT_FIELDS = ("current_html", "target_html", "instruction")

# добавить разумное число токенов для картинок
def compute_max_length(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    quantile: float = 0.99,
    round_to: int = 64,
    text_fields: tuple[str, ...] = TEXT_FIELDS,
    num_proc: int = 4,
    image_token_budget: int = 0,
) -> int:
    """
    Token length at `quantile` over the dataset, rounded up to `round_to`
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
