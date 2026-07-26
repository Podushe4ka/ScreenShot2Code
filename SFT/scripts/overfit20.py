"""Overfit sanity-check: 20 WebSight examples, LoRA, loss must collapse.

Run (either way works):
    CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.overfit20 [--model_name_or_path ...]
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/overfit20.py

If the loss does not drop by an order of magnitude on 20 examples, the bug is
in the pipeline (label masking, collation, target modules) — not in the data.
Fix it here before starting a real run.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import (
    Dataset,
    Features,
    Image,
    Sequence,
    Value,
    load_dataset,
)
from trl import ModelConfig, ScriptArguments, SFTConfig

from configs.gen import MODELS, max_length_for
from train.train_sft import build_trainer

DATA_PATH = Path("data/websight20")
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

def preparing():
    if (DATA_PATH / "dataset_info.json").exists():
        return

    ds = load_dataset("HuggingFaceM4/WebSight", "v0.2", split="train", streaming=True)
    first20 = ds.take(20)
    first20 = (
        {
            "task_type": "drafting",
            "images": [ex["image"]],
            "current_html": "",
            "target_html": ex["text"],
            "instruction": "",
        }
        for ex in first20
    )
    features = Features(
        {
            "task_type": Value("string"),
            "images": Sequence(Image()),
            "current_html": Value("string"),
            "target_html": Value("string"),
            "instruction": Value("string"),
        }
    )
    dataset = Dataset.from_list(list(first20), features=features)
    dataset.save_to_disk(DATA_PATH)


def main():
    preparing()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default="c202236235762e1c871ad0ccb60c8ee5ba337b9a")
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="по умолчанию берётся расчётный бюджет модели из configs/gen.py",
    )
    args = parser.parse_args()

    max_length = args.max_length
    if max_length is None:
        entry = next(
            (m for m in MODELS.values() if m["id"] == args.model_name_or_path), None
        )
        if entry is None:
            parser.error(
                f"{args.model_name_or_path} нет в configs/gen.py — "
                "задайте --max_length явно"
            )
        max_length = max_length_for(entry)
    print(f"max_length={max_length}")

    # Датасет сохранён одним куском, без сплитов: eval здесь и не нужен —
    # цель проверки в том, чтобы train loss схлопнулся.
    script_args = ScriptArguments(dataset_name=str(DATA_PATH))
    training_args = SFTConfig(
        output_dir="./train_res",
        num_train_epochs=15,
        learning_rate=2e-4,
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        logging_steps=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        save_strategy="no",
        bf16=True,
        remove_unused_columns=False,
        max_length=max_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model_args = ModelConfig(
        model_name_or_path=args.model_name_or_path,
        model_revision=args.revision,
        dtype="bfloat16",
        use_peft=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        lora_target_modules=LORA_TARGET_MODULES,
        attn_implementation="flash_attention_2"
    )

    trainer = build_trainer(script_args, training_args, model_args)
    trainer.train()

    losses = [r["loss"] for r in trainer.state.log_history if "loss" in r]
    print(
        f"loss: {losses[0]:.4f} -> {losses[-1]:.4f} "
        f"(min {min(losses):.4f} over {len(losses)} steps)"
    )

    assert losses[-1] < losses[0] * 0.1, (
        f"не переобучилось: {losses[0]:.3f} -> {losses[-1]:.3f}"
    )
    print("OK: loss collapsed, pipeline looks sane")


if __name__ == "__main__":
    main()
