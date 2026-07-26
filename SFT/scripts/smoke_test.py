"""Самопроверка форматирования, маскирования лосса и бюджета токенов.

Запуск (оба способа работают):
    uv run python -m scripts.smoke_test [--model_name_or_path ...] [--revision ...]
    uv run python scripts/smoke_test.py

Это тестовая обвязка, поэтому здесь названа конкретная модель — библиотечный код
(`train/formatting.py`) остаётся модельно-агностичным.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from transformers import AutoProcessor

from configs.gen import MODELS, visual_tokens
from train.formatting import (
    MAX_PIXELS,
    MIN_PIXELS,
    RESPONSE_TEMPLATE,
    THINK_END,
    TURN_END,
    _assistant_spans,
    _find_subsequence,
    make_collate_fn,
    to_message,
    visual_token_budget,
)

# Заведомо больше MAX_PIXELS — картинка обязана упереться в потолок.
LARGE_IMAGE = (1280, 1280)


def _drafting_example(html="<!DOCTYPE html><html><body><h1>Hi</h1></body></html>"):
    return to_message(
        {
            "task_type": "drafting",
            "target_html": html,
            "current_html": "",
            "instruction": "",
            "images": [Image.new("RGB", LARGE_IMAGE, (200, 200, 200))],
        }
    )


def _polishing_example():
    return to_message(
        {
            "task_type": "polishing",
            "target_html": "<html><body>fixed</body></html>",
            "current_html": "<html><body>broken</body></html>",
            "instruction": "",
            "images": [
                Image.new("RGB", LARGE_IMAGE, (200, 200, 200)),
                Image.new("RGB", LARGE_IMAGE, (100, 100, 100)),
            ],
        }
    )


def check_masking(processor, collate_fn, image_token_id, response_ids):
    """Промпт замаскирован, ответ ассистента — нет, картинки не в лоссе."""
    batch = collate_fn([_drafting_example()])
    input_ids, labels = batch["input_ids"][0], batch["labels"][0]

    assert labels.shape == input_ids.shape
    img_pos = input_ids == image_token_id
    assert img_pos.any(), "в батче нет image-токенов"
    assert (labels[img_pos] == -100).all(), "image-токены должны быть замаскированы"
    assert (labels != -100).any(), "ответ ассистента должен давать лосс"

    start = _find_subsequence(input_ids.tolist(), response_ids)
    assert start is not None, "маркер ответа ассистента не найден"
    assert (
        labels[: start + len(response_ids)] == -100
    ).all(), "промпт должен быть замаскирован"

    kept, total = int((labels != -100).sum()), int(labels.numel())
    print(f"  маскирование: в лоссе {kept}/{total} токенов")
    return batch


def check_visual_budget(processor, batch, image_token_id):
    """Фактическое число визуальных токенов не превышает расчётный потолок.

    Проверка односторонняя. `visual_token_budget` — верхняя граница: smart_resize
    округляет стороны ВНИЗ до кратного factor, поэтому реально выходит на
    несколько процентов меньше (1280x1280 при потолке 1.31 Мп даёт 35x35=1225,
    а не 1280). Для фильтрации оценка сверху и нужна.

    Превышение означает, что min_pixels/max_pixels не доехали до процессора: у
    Qwen3.5 в preprocessor_config.json их нет, и без наших констант он взял бы
    свой longest_edge (16384 токена) — скриншот уехал бы в модель целиком.
    """
    expected = visual_token_budget(processor)
    merge = processor.image_processor.merge_size
    actual_grid = int(batch["image_grid_thw"].prod(dim=-1).sum()) // merge**2
    actual_tokens = int((batch["input_ids"] == image_token_id).sum())

    assert actual_grid == actual_tokens, (
        f"image_grid_thw даёт {actual_grid} токенов, а в input_ids их "
        f"{actual_tokens} — рассинхрон сетки и плейсхолдеров"
    )
    assert actual_tokens <= expected, (
        f"картинка заняла {actual_tokens} токенов при потолке {expected}. "
        "min_pixels/max_pixels не доехали до процессора."
    )
    assert actual_tokens >= 0.8 * expected, (
        f"картинка заняла всего {actual_tokens} токенов при потолке {expected} — "
        "слишком мало для скриншота, упирающегося в MAX_PIXELS."
    )
    print(
        f"  визуальный бюджет: {actual_tokens} токенов "
        f"({actual_tokens / expected:.0%} от потолка {expected})"
    )


def check_factor_matches_configs(processor, model_id):
    """`factor` в configs/gen.py совпадает с фактическим у процессора."""
    ip = processor.image_processor
    actual = ip.patch_size * ip.merge_size
    declared = {m["id"]: m.get("factor") for m in MODELS.values()}.get(model_id)
    if declared is None:
        print(f"  factor: {actual} (модели нет в configs/gen.py — сверка пропущена)")
        return
    assert declared == actual, (
        f"configs/gen.py объявляет factor={declared} для {model_id}, "
        f"а процессор даёт {actual}. Из-за этого max_length будет посчитан неверно."
    )
    print(f"  factor: {actual}, визуальных токенов {visual_tokens(actual)} — сходится")


def check_prompt_overhead(processor, collate_fn, image_token_id, response_ids):
    """Сколько токенов съедает промпт с обвязкой — сверка PROMPT_OVERHEAD_TOKENS."""
    batch = collate_fn([_drafting_example(html="x")])
    ids = batch["input_ids"][0].tolist()
    start = _find_subsequence(ids, response_ids) + len(response_ids)
    n_image = sum(1 for i in ids[:start] if i == image_token_id)
    print(f"  накладные расходы промпта: {start - n_image} токенов (без картинки)")


def check_polishing(processor, collate_fn, image_token_id):
    """Две картинки не схлопываются: список списков доезжает до процессора."""
    batch = collate_fn([_polishing_example()])
    n_images = int(batch["image_grid_thw"].size(0))
    assert n_images == 2, f"ожидались 2 картинки, процессор увидел {n_images}"
    n_tokens = int((batch["input_ids"] == image_token_id).sum())
    budget = visual_token_budget(processor)
    # Границы те же, что в check_visual_budget: потолок сверху, разумный низ.
    assert 2 * 0.8 * budget <= n_tokens <= 2 * budget, (
        f"две картинки заняли {n_tokens} токенов при потолке {2 * budget}"
    )
    assert (batch["labels"][batch["input_ids"] == image_token_id] == -100).all()
    print(f"  polishing: 2 картинки, {n_tokens} визуальных токенов")


def check_multi_turn(processor):
    """Маска собирает ВСЕ ходы ассистента, а не только первый."""
    tokenizer = processor.tokenizer
    response_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
    turn_end_ids = tokenizer.encode(TURN_END, add_special_tokens=False)
    think_end_ids = tokenizer.encode(THINK_END, add_special_tokens=False)

    text = (
        f"<|im_start|>user\nfirst{TURN_END}\n"
        f"{RESPONSE_TEMPLATE}answer one{TURN_END}\n"
        f"<|im_start|>user\nfeedback{TURN_END}\n"
        f"{RESPONSE_TEMPLATE}answer two{TURN_END}\n"
    )
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    spans = _assistant_spans(ids, tokenizer, response_ids, turn_end_ids, think_end_ids)

    assert len(spans) == 2, f"ожидались 2 хода ассистента, найдено {len(spans)}"
    covered = tokenizer.decode([t for s, e in spans for t in ids[s:e]])
    assert "answer one" in covered and "answer two" in covered
    assert "feedback" not in covered, "фидбек пользователя попал в лосс"
    print(f"  multi-turn: {len(spans)} хода ассистента, фидбек вне лосса")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--revision", default="c202236235762e1c871ad0ccb60c8ee5ba337b9a")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        revision=args.revision,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    collate_fn = make_collate_fn(processor)
    tokenizer = processor.tokenizer
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    response_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)

    print(f"модель: {args.model_name_or_path}")
    print(f"словарь токенайзера: {len(tokenizer)}")
    print(f"image token id: {image_token_id}")

    batch = check_masking(processor, collate_fn, image_token_id, response_ids)
    check_visual_budget(processor, batch, image_token_id)
    check_factor_matches_configs(processor, args.model_name_or_path)
    check_prompt_overhead(processor, collate_fn, image_token_id, response_ids)
    check_polishing(processor, collate_fn, image_token_id)
    check_multi_turn(processor)

    print("OK: форматирование, маскирование и бюджет токенов в порядке")


if __name__ == "__main__":
    main()
