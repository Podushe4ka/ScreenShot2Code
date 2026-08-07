import logging
import math
import os

from datasets import concatenate_datasets

logger = logging.getLogger(__name__)

DRAFTING_PROMPT = (
    "You are an expert front-end developer. Look at this webpage screenshot and "
    "write a SINGLE self-contained HTML file (inline <style>, no external CSS/JS/fonts, "
    "no network requests) that reproduces the layout, text, and colors as closely as "
    "possible. Use plain gray placeholder boxes instead of any real images. "
    "Output ONLY the raw HTML code, with no explanation and no markdown code fences."
)

# Второй стиль вывода — React + Tailwind с CDN. Он появился вместе с синтетическим
# набором (Data/converters/synth), где часть страниц написана именно так: это стиль
# выхлопа UI2Code^N (React+JSX+Tailwind в 40/40 замеренных сэмплов, docs/RESULTS.md).
#
# ЗАЧЕМ ОТДЕЛЬНЫЙ ПРОМПТ. В наборе смешаны два стиля вывода, и без разделяющей
# инструкции один и тот же скриншот отображался бы в два разных валидных таргета —
# противоречивый супервижн, на котором модель усредняет и портит оба стиля.
# Стиль берётся из колонки `impl`; её отсутствие означает старый набор
# (WebSight/WebCode2M) и трактуется как static — иначе такие наборы сломались бы.
DRAFTING_PROMPT_STATIC = DRAFTING_PROMPT
DRAFTING_PROMPT_REACT = (
    "You are an expert front-end developer. Look at this webpage screenshot and "
    "write a SINGLE HTML file that reproduces the layout, text, and colors as closely "
    "as possible using React and Tailwind CSS from a CDN: load react, react-dom and "
    "@babel/standalone, put the components in a <script type=\"text/babel\"> block, "
    "style with Tailwind utility classes, and mount into <div id=\"root\">. "
    "Use plain gray placeholder boxes instead of any real images and draw icons as "
    "inline <svg>. "
    "Output ONLY the raw HTML code, with no explanation and no markdown code fences."
)

_DRAFTING_BY_IMPL = {
    "static_inline": DRAFTING_PROMPT_STATIC,
    "react_cdn": DRAFTING_PROMPT_REACT,
}


def drafting_prompt_for(example) -> str:
    return _DRAFTING_BY_IMPL.get(example.get("impl") or "", DRAFTING_PROMPT_STATIC)

POLISHING_PROMPT = (
    "You are an expert front-end developer. You are given TWO screenshots: the "
    "FIRST is the TARGET design the page should match; the SECOND is the CURRENT "
    "rendering produced by the HTML below. The current HTML is:\n"
)

POLISHING_SUFFIX = (
    "\n\nFix the HTML so its rendering matches the target screenshot as closely as "
    "possible in layout, text, and colors. Keep it a SINGLE self-contained HTML file "
    "(inline <style>, no external CSS/JS/fonts, no network requests) and use plain "
    "gray placeholder boxes instead of any real images. "
    "Output ONLY the raw corrected HTML, with no explanation and no markdown code fences."
)

EDITING_PROMPT = (
    "You are an expert front-end developer. You are given a screenshot of a webpage "
    "and its current HTML. Apply the requested edit and return the full updated page. "
    "The current HTML is:\n"
)
EDITING_INSTRUCTION = "\n\nEdit to apply:\n"
EDITING_SUFFIX = (
    "\n\nReturn the COMPLETE modified HTML as a SINGLE self-contained file "
    "(inline <style>, no external CSS/JS/fonts, no network requests), using plain gray "
    "placeholder boxes instead of real images. "
    "Output ONLY the raw HTML, with no explanation and no markdown code fences."
)

# Пиксель-бюджет картинки. Переопределяется через окружение, чтобы свипать ось A
# плана, не плодя конфиги: SFT_MAX_PIXELS=3932160 (3.93 Мп) даёт странице 1280x3072
# натуральный масштаб, тогда как при 2.10 Мп она ужимается до 928x2240 (текст в
# 1.38 раза мельче). Значение уезжает в meta -> ClearML (тег pxN.NNMp), так что ран
# самоописателен. Тот же бюджет обязан стоять на бенче (--max-pixels), иначе
# чекпоинт меряется вне своего трейн-распределения.
MIN_PIXELS = int(os.environ.get("SFT_MIN_PIXELS", 262_144))
MAX_PIXELS = int(os.environ.get("SFT_MAX_PIXELS", 2_097_152))

RESPONSE_TEMPLATE = "<|im_start|>assistant\n"
TURN_END = "<|im_end|>"
THINK_END = "</think>"

_MAX_WHITESPACE_SKIP = 4


def visual_token_budget(processor) -> int:
    """
    Верхняя оценка числа визуальных токенов на одну картинку.
    """
    ip = processor.image_processor
    factor = ip.patch_size * ip.merge_size
    return math.ceil(MAX_PIXELS / factor**2)


def to_message(example):
    html_gt = example["target_html"]

    if example["task_type"] == "drafting":
        example["messages"] = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": drafting_prompt_for(example)},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": html_gt}],
            },
        ]

    elif example["task_type"] == "polishing":
        html_cur = example["current_html"]
        example["messages"] = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": POLISHING_PROMPT + html_cur + POLISHING_SUFFIX,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": html_gt}],
            },
        ]
    elif example["task_type"] == "editing":
        html_cur = example["current_html"]
        instruction = example["instruction"]

        example["messages"] = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": EDITING_PROMPT
                        + html_cur
                        + EDITING_INSTRUCTION
                        + instruction
                        + EDITING_SUFFIX,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": html_gt}],
            },
        ]
    return example


def build_messages(dataset):
    """
    Плоская таблица с единственной колонкой messages.

    Отдельно от `add_messages`, потому что склейка по axis=1 даёт
    `ConcatenationTable`, а её нельзя нарезать на шарды и передать воркерам
    `map(num_proc>1)` — распаковка на стороне воркера падает в
    `ConcatenationTable.__setstate__`. Замер длины ходит по этой таблице.
    """
    text_columns = [name for name in dataset.column_names if name != "images"]
    return dataset.select_columns(text_columns).map(
        to_message, remove_columns=text_columns, desc="Разбор сэмплов в messages"
    )


def add_messages(dataset, messages=None):
    """
    Приклеивает messages к датасету, не притрагиваясь к колонке images.
    """
    if messages is None:
        messages = build_messages(dataset)
    return concatenate_datasets([dataset, messages], axis=1)


def _find_subsequence(seq: list[int], sub: list[int], start: int = 0) -> int | None:
    """Первое вхождение `sub` в `seq` начиная с позиции `start`."""
    if not sub:
        return None
    for i in range(start, len(seq) - len(sub) + 1):
        if seq[i : i + len(sub)] == sub:
            return i
    return None


def _assistant_spans(
    ids: list[int],
    tokenizer,
    response_ids: list[int],
    turn_end_ids: list[int],
    think_end_ids: list[int],
) -> list[tuple[int, int]]:
    """
    Интервалы [start, end) в лосс.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    while True:
        marker = _find_subsequence(ids, response_ids, pos)
        if marker is None:
            break
        start = marker + len(response_ids)

        turn_end = _find_subsequence(ids, turn_end_ids, start)
        think_end = _find_subsequence(ids, think_end_ids, start)
        if think_end is not None and (turn_end is None or think_end < turn_end):
            start = think_end + len(think_end_ids)
            skipped = 0
            while (
                start < len(ids)
                and skipped < _MAX_WHITESPACE_SKIP
                and not tokenizer.decode([ids[start]]).strip()
            ):
                start += 1
                skipped += 1
            turn_end = _find_subsequence(ids, turn_end_ids, start)

        end = turn_end + len(turn_end_ids) if turn_end is not None else len(ids)
        if end > start:
            spans.append((start, end))
        pos = end
    return spans


def make_collate_fn(processor, pad_to_multiple_of: int | None = None):
    """
    Коллатор с маскированием лосса по ходам ассистента.

    `pad_to_multiple_of` округляет длину батча вверх. Профиль показал 272
    загрузки CUDA-ядер посреди прогона: `fla` компилирует своё ядро под каждую
    новую форму, а при паддинге до максимума в батче формы почти не повторяются.
    Округление до корзины сводит их к десятку.
    """
    tokenizer = processor.tokenizer
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    response_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
    turn_end_ids = tokenizer.encode(TURN_END, add_special_tokens=False)
    think_end_ids = tokenizer.encode(THINK_END, add_special_tokens=False)

    if image_token_id is None or image_token_id == tokenizer.unk_token_id:
        raise ValueError(
            "'<|image_pad|>' отсутствует в словаре токенайзера — процессор не "
            "мультимодальный либо использует другой image-токен."
        )

    def collate_fn(examples: list) -> dict:
        texts = [
            processor.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
            for ex in examples
        ]
        images = [ex["images"] for ex in examples]
        pad_kwargs = {}
        if pad_to_multiple_of:
            pad_kwargs["pad_to_multiple_of"] = pad_to_multiple_of
        batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
            **pad_kwargs,
        )

        input_ids = batch["input_ids"]
        labels = input_ids.clone().fill_(-100)

        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()
            spans = _assistant_spans(
                ids, tokenizer, response_ids, turn_end_ids, think_end_ids
            )
            if not spans:
                logger.warning(
                    "В сэмпле %d не найден ход ассистента — сэмпл не даёт лосса.", i
                )
                continue
            for start, end in spans:
                labels[i, start:end] = input_ids[i, start:end]

        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100
        labels[input_ids == image_token_id] = -100

        batch["labels"] = labels
        return batch

    return collate_fn
