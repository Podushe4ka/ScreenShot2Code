"""Self-test for train/formatting.py assistant-only label masking.

Run:
    uv run python -m scripts.smoke_test [--model_name_or_path ...] [--revision ...]

This is a test harness, so it names a concrete model to instantiate the
processor. Library code (train/formatting.py) stays model-agnostic.
"""
import argparse

from PIL import Image
from transformers import AutoProcessor

from train.formatting import(
    MAX_PIXELS,
    MIN_PIXELS,
    RESPONSE_TEMPLATE,
    _find_subsequence,
    make_collate_fn,
    to_message,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        revision=args.revision,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    collate_fn = make_collate_fn(processor)

    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    response_token_ids = processor.tokenizer.encode(
        RESPONSE_TEMPLATE, add_special_tokens=False
    )

    dummy = Image.new("RGB", (768, 512), (200, 200, 200))
    example = to_message(
        {
            "task_type": "drafting",
            "target_html": "<!DOCTYPE html><html><body><h1>Hi</h1></body></html>",
            "images": [dummy],
        }
    )

    batch = collate_fn([example])
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

    assert labels.shape == input_ids.shape
    img_pos = input_ids == image_token_id
    assert img_pos.any(), "expected image tokens in the batch"
    assert (labels[img_pos] == -100).all(), "image tokens must be masked"
    assert (labels != -100).any(), "assistant tokens must contribute to the loss"

    start = _find_subsequence(input_ids.tolist(), response_token_ids)
    assert start is not None, "assistant response marker not found in rendered chat"
    assert (
        labels[: start + len(response_token_ids)] == -100
    ).all(), "prompt must be masked"

    kept = int((labels != -100).sum())
    total = int(labels.numel())
    chat_template = getattr(processor, "chat_template", None) or ""
    rendered = processor.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )

    print(f"IMAGE_TOKEN_ID: {image_token_id}")
    print(f"RESPONSE_TEMPLATE token ids: {response_token_ids}")
    print(
        f"chat template exposes a {{% generation %}} block: "
        f"{'{% generation %}' in chat_template}"
    )
    print(f"labels kept (unmasked)/total: {kept}/{total}")
    print("--- rendered chat ---")
    print(rendered)
    print("OK: assistant-only masking looks correct")


if __name__ == "__main__":
    main()
