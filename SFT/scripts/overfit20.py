"""Overfit sanity-check: 20 примеров WebSight, full fine-tune, лосс обязан рухнуть.

Запуск (одна карта):
    CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.overfit20
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/overfit20.py --model_name_or_path ...

Запуск как в проде (две карты, ZeRO-2) — если одна карта не тянет по памяти:
    CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --nproc_per_node=2 \
        -m scripts.overfit20 --deepspeed configs/deepspeed_zero2.json

Если на 20 примерах лосс не падает на порядок — баг в пайплайне (маскирование
меток, коллация, бюджет длины), а не в данных. Чинить здесь, до боевого рана.

Почему full FT, а не LoRA: LoRA проверяет меньше. При замороженной базе часть
ошибок в маскировании и коллации маскируется самим адаптером — он просто не
может выучить мусор. Full FT переобучается на 20 примерах гарантированно, и
если этого не произошло, поломка настоящая.
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
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"

# Совпадает с configs/full_ft_qwen3_5_4b.yaml. Отклоняться незачем: смысл
# проверки в том, чтобы прогнать боевые гиперпараметры, а не свои.
LEARNING_RATE = 2.0e-5
WEIGHT_DECAY = 0.05


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


def entry_for(model_id: str) -> dict | None:
    """Описание модели из configs/gen.py по её HF-идентификатору."""
    return next((m for m in MODELS.values() if m["id"] == model_id), None)


def main():
    preparing()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument(
        "--revision",
        default=None,
        help="по умолчанию берётся ревизия модели из configs/gen.py",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="по умолчанию берётся расчётный бюджет модели из configs/gen.py",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=None,
        help="по умолчанию full_bs модели из configs/gen.py",
    )
    parser.add_argument(
        "--deepspeed",
        default=None,
        help="путь к конфигу ZeRO, напр. configs/deepspeed_zero2.json",
    )
    args = parser.parse_args()

    # Ревизия, бюджет длины и микробатч берутся из одного источника — MODELS.
    # Раньше ревизия была захардкожена в этом файле и разъезжалась с моделью.
    entry = entry_for(args.model_name_or_path)
    if entry is None and (
        args.revision is None
        or args.max_length is None
        or args.per_device_train_batch_size is None
    ):
        parser.error(
            f"{args.model_name_or_path} нет в configs/gen.py — задайте "
            "--revision, --max_length и --per_device_train_batch_size явно"
        )

    revision = args.revision or entry["rev"]
    max_length = args.max_length or max_length_for(entry)
    # full_bs описан не у всех моделей; 2 — консервативный дефолт, при котором
    # логиты (bs, max_length, 248320) остаются в разумных пределах.
    batch_size = args.per_device_train_batch_size or entry.get("full_bs", 2)
    print(
        f"модель={args.model_name_or_path} rev={revision[:8]} "
        f"max_length={max_length} bs={batch_size} full FT"
    )

    script_args = ScriptArguments(dataset_name=str(DATA_PATH))
    training_args = SFTConfig(
        output_dir="./train_res",
        num_train_epochs=15,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        # Константный LR без прогрева: на 20 примерах нужно увидеть чистое
        # падение лосса, а косинус с warmup смазал бы картину.
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        optim="adamw_torch_fused",
        logging_steps=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        save_strategy="no",
        bf16=True,
        remove_unused_columns=False,
        max_length=max_length,
        # full_gc: без чекпоинтинга full FT 4B падает по памяти уже на bs=4.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=args.deepspeed,
    )
    model_args = ModelConfig(
        model_name_or_path=args.model_name_or_path,
        model_revision=revision,
        dtype="bfloat16",
        use_peft=False,
        attn_implementation="flash_attention_2",
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
