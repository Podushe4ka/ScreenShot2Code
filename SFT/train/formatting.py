import logging
import math

logger = logging.getLogger(__name__)

DRAFTING_PROMPT = (
    "You are an expert front-end developer. Look at this webpage screenshot and "
    "write a SINGLE self-contained HTML file (inline <style>, no external CSS/JS/fonts, "
    "no network requests) that reproduces the layout, text, and colors as closely as "
    "possible. Use plain gray placeholder boxes instead of any real images. "
    "Output ONLY the raw HTML code, with no explanation and no markdown code fences."
)

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

MIN_PIXELS = 262_144
MAX_PIXELS = 1_310_720

RESPONSE_TEMPLATE = "<|im_start|>assistant\n"
TURN_END = "<|im_end|>"
THINK_END = "</think>"

_MAX_WHITESPACE_SKIP = 4


def visual_token_budget(processor) -> int:
    """
    Верхняя оценка числа визуальных токенов на одну картинку.

    Именно оценка сверху: smart_resize округляет стороны вниз до кратного
    factor, поэтому фактическое число на несколько процентов меньше (1280x1280
    при потолке 1.31 Мп -> 35x35 = 1225 токенов). Для фильтрации по бюджету
    ошибка в эту сторону безопасна.
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
                    {"type": "text", "text": DRAFTING_PROMPT},
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
    Интервалы [start, end) ответов ассистента — то, что попадает в лосс.
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


def make_collate_fn(processor):
    """
    Коллатор с маскированием лосса по ходам ассистента.
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
        batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
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
