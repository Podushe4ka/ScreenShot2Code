"""Батчинг по бюджету токенов вместо фиксированного числа примеров.

Зачем. При фиксированном `per_device_train_batch_size` потолок памяти считается
по самому длинному примеру, и на коротких батч недозаполняется. Разброс длин у
webcode2m — 26x, поэтому число запусков forward/backward выходит почти вдвое
больше необходимого, а профиль показал, что мы упираемся не в вычисления, а в
диспетчеризацию: CPU тратит вдвое больше времени, чем GPU (см. THROUGHPUT.md).

Стоимость батча считается по паддингу — `n * L_max`, а не суммой длин: в память
ложится прямоугольник, и набор по сумме даёт OOM на батче из разнодлинных
примеров.

`length_bucket` округляет `L_max` вверх. Это не косметика: `fla` компилирует
ядро под каждую новую форму тензора, и в профиле видно 272 загрузки ядер уже
на третьем шаге. Округление сводит число различных форм к десятку. Работает
только вместе с `pad_to_multiple_of` у коллатора — иначе батч всё равно
паддится до своего максимума.
"""

import logging
import math
import random
from dataclasses import dataclass, field
from functools import partial
from statistics import median

from torch.utils.data import DataLoader, Sampler
from transformers.trainer_utils import seed_worker
from trl import SFTTrainer

logger = logging.getLogger(__name__)


@dataclass
class BatchingArguments:
    """Ручки батчинга по токенам. Всё выключено по умолчанию."""

    max_tokens_per_batch: int | None = field(
        default=None,
        metadata={
            "help": "Бюджет токенов на микробатч (n * L_max). Если задан, "
            "per_device_train_batch_size не используется."
        },
    )
    max_batch_size: int | None = field(
        default=None,
        metadata={"help": "Потолок числа примеров в батче поверх бюджета токенов."},
    )
    length_bucket: int = field(
        default=1024,
        metadata={
            "help": "Округлять длину батча вверх до кратного. Снижает число "
            "различных форм тензоров и загрузок CUDA-ядер. 0 — не округлять."
        },
    )


def bucketed(length: int, bucket: int) -> int:
    if bucket <= 1:
        return length
    return math.ceil(length / bucket) * bucket


def plan_token_batches(
    lengths: list[int],
    max_tokens: int,
    max_batch_size: int | None = None,
    bucket: int = 1024,
) -> list[list[int]]:
    """Разложить индексы по батчам так, чтобы `n * L_max <= max_tokens`.

    Сортировка по убыванию длины: самые тяжёлые батчи идут первыми, поэтому
    OOM (если он есть) случается на первом шаге, а не через час обучения.
    Пример длиннее бюджета не выбрасывается — уходит в батч из одного.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)

    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    oversized = 0

    for idx in order:
        cost = bucketed(lengths[idx], bucket)
        if cost > max_tokens:
            if current:
                batches.append(current)
                current, current_max = [], 0
            batches.append([idx])
            oversized += 1
            continue

        new_max = max(current_max, cost)
        too_many = max_batch_size is not None and len(current) + 1 > max_batch_size
        if current and ((len(current) + 1) * new_max > max_tokens or too_many):
            batches.append(current)
            current, current_max = [idx], cost
        else:
            current.append(idx)
            current_max = new_max

    if current:
        batches.append(current)

    if oversized:
        logger.warning(
            "%d примеров длиннее бюджета %d — каждый пошёл в батч из одного. "
            "Проверьте max_tokens_per_batch против max_length.",
            oversized,
            max_tokens,
        )
    return batches


def batch_plan_report(
    batches: list[list[int]], lengths: list[int], max_tokens: int, bucket: int
) -> str:
    """Строка для лога — печатать до загрузки модели, как LengthReport."""
    sizes = [len(b) for b in batches]
    fill = [
        len(b) * bucketed(max(lengths[i] for i in b), bucket) / max_tokens
        for b in batches
    ]
    return (
        f"Батчи по бюджету: {len(batches)} шт. при {len(lengths)} примерах "
        f"(бюджет={max_tokens} ток., корзина={bucket}).\n"
        f"Примеров в батче: мин={min(sizes)}, медиана={median(sizes):.0f}, "
        f"макс={max(sizes)}. Заполнение бюджета: медиана={median(fill):.0%}, "
        f"мин={min(fill):.0%}."
    )


class TokenBudgetBatchSampler(Sampler[list[int]]):
    """Состав батчей фиксирован, между эпохами меняется только их порядок.

    Фиксированный состав даёт стабильный `__len__`, а значит воспроизводимое
    число шагов и корректное LR-расписание.

    Сэмплер намеренно НЕ знает про ранги: `accelerator.prepare` оборачивает его
    в `BatchSamplerShard`, который сам раздаёт батчи по процессам. Ручной шардинг
    привёл бы к двойному делению.

    Эпохи считаются внутренним счётчиком: `BatchSamplerShard` не пробрасывает
    `set_epoch`, а каждый ранг вызывает `__iter__` ровно раз за эпоху, так что
    счётчики остаются синхронными.
    """

    def __init__(self, batches: list[list[int]], seed: int = 0):
        self._batches = batches
        self._seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self):
        order = list(range(len(self._batches)))
        random.Random(self._seed + self._epoch).shuffle(order)
        self._epoch += 1
        for i in order:
            yield self._batches[i]


class TokenBudgetSFTTrainer(SFTTrainer):
    """SFTTrainer с батчами переменного размера.

    `Trainer._get_dataloader` умеет пробрасывать только `sampler`, ручки для
    `batch_sampler` там нет — поэтому переопределяем весь `get_train_dataloader`.
    """

    def __init__(self, *args, batch_sampler: Sampler | None = None, **kwargs):
        self._batch_sampler = batch_sampler
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        if self._batch_sampler is None:
            return super().get_train_dataloader()
        if self.train_dataset is None:
            raise ValueError("Нет train_dataset.")

        args = self.args
        params = {
            "batch_sampler": self._batch_sampler,
            "collate_fn": self.data_collator,
            "num_workers": args.dataloader_num_workers,
            "pin_memory": args.dataloader_pin_memory,
            "persistent_workers": args.dataloader_persistent_workers,
            "worker_init_fn": partial(
                seed_worker,
                num_workers=args.dataloader_num_workers,
                rank=args.process_index,
            ),
        }
        if args.dataloader_num_workers > 0:
            params["prefetch_factor"] = args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(self.train_dataset, **params))
