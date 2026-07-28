"""
Отбраковка сэмплов по бюджету токенов.
"""

import logging
from dataclasses import dataclass

from datasets import Dataset

from train.formatting import visual_token_budget

logger = logging.getLogger(__name__)


def estimate_example_length(example: dict, processor, image_budget: int) -> int:
    """
    Длина сэмпла в токенах — то, что увидит коллатор, оценка сверху.
    """
    messages = example.get("messages")
    if not messages:
        raise KeyError(
            "В сэмпле нет 'messages' — примените to_message() до замера длины."
        )

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    n_text_tokens = len(processor.tokenizer(text, add_special_tokens=False)["input_ids"])

    n_images = sum(
        1
        for message in messages
        for part in message["content"]
        if part.get("type") == "image"
    )
    return n_text_tokens - n_images + n_images * image_budget


@dataclass
class LengthReport:
    """
    Что показал замер бюджета — печатается до загрузки модели.
    """

    total: int
    kept: int
    max_length: int
    image_budget: int
    longest: int

    @property
    def dropped(self) -> int:
        return self.total - self.kept

    @property
    def dropped_share(self) -> float:
        return self.dropped / self.total if self.total else 0.0

    def format(self) -> str:
        return (
            f"Бюджет токенов: max_length={self.max_length}, "
            f"картинка={self.image_budget} ток., "
            f"самый длинный сэмпл={self.longest} ток.\n"
            f"Отбраковано по длине: {self.dropped}/{self.total} "
            f"({self.dropped_share:.1%}), остаётся {self.kept}."
        )


def filter_by_length(
    dataset: Dataset,
    processor,
    max_length: int,
    num_proc: int | None = None,
) -> tuple[Dataset, LengthReport]:
    """Убрать сэмплы, не влезающие в `max_length`.
    Возвращает отфильтрованный датасет и отчёт
    """
    image_budget = visual_token_budget(processor)
    lengths = dataset.map(
        lambda ex: {"_len": estimate_example_length(ex, processor, image_budget)},
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Измерение длины сэмплов",
    )["_len"]

    keep = [i for i, n in enumerate(lengths) if n <= max_length]
    report = LengthReport(
        total=len(lengths),
        kept=len(keep),
        max_length=max_length,
        image_budget=image_budget,
        longest=max(lengths) if lengths else 0,
    )

    if not keep:
        raise ValueError(
            f"Ни один сэмпл не влезает в max_length={max_length} "
            f"(самый короткий — {min(lengths)} ток., картинка одна занимает "
            f"{image_budget}). Проверьте max_length и MAX_PIXELS."
        )
    if report.dropped_share > 0.2:
        logger.warning(
            "По длине отброшено %.1f%% датасета — возможно, max_length занижен "
            "или скриншоты крупнее ожидаемого.",
            report.dropped_share * 100,
        )

    return dataset.select(keep), report
