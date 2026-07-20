"""Overfit sanity-check: 20 WebSight examples, LoRA, loss must collapse.

Run:
    CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.overfit20

If the loss does not drop by an order of magnitude on 20 examples, the bug is
in the pipeline (label masking, collation, target modules) — not in the data.
Fix it here before starting a real run.
"""
from pathlib import Path

from datasets import Dataset, Features, Image, Sequence, Value, load_dataset
from trl import ModelConfig, ScriptArguments, SFTConfig

from train.train_sft import build_trainer

DATA_PATH = Path("data/websight20")

MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"

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
    first20 = map(
        lambda ex: {
            "task_type": "drafting",
            "images": [ex["image"]],
            "current_html": "",
            "target_html": ex["text"],
            "instruction": "",
        },
        first20,
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

    script_args = ScriptArguments(dataset_name=str(DATA_PATH))
    training_args = SFTConfig(
        output_dir="./train_res",
        num_train_epochs=40,
        learning_rate=2e-4,
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        save_strategy="no",
        bf16=True,
        remove_unused_columns=False,
        # dataset_kwargs={"skip_prepare_dataset": True},
        max_length=8192,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model_args = ModelConfig(
        model_name_or_path=MODEL_ID,
        model_revision=MODEL_REVISION,
        torch_dtype="bfloat16",
        use_peft=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        lora_target_modules=LORA_TARGET_MODULES,
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
