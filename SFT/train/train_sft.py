import logging
from dataclasses import dataclass, field
from pathlib import Path

from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
)

from data.filtering import filter_by_length
from data.loader import load_sft_dataset
from train.formatting import (
    MAX_PIXELS,
    MIN_PIXELS,
    make_collate_fn,
    to_message,
    visual_token_budget,
)
from train.run_info import format_meta, git_commit, make_run_name, save_run_info

logger = logging.getLogger(__name__)


@dataclass
class SftScriptArguments(ScriptArguments):
    """`ScriptArguments` из TRL плюс доля валидации.

    В TRL есть только имена сплитов (`dataset_train_split`), а датасет от
    data-трека приходит одним куском, поэтому held-out отрезаем сами.
    """

    val_size: float = field(
        default=0.02,
        metadata={"help": "Доля датасета под валидацию. 0 — обучение без eval."},
    )


def _split_dataset(dataset, val_size: float, seed: int):
    """Отрезать held-out. Возвращает (train, eval|None)."""
    if val_size <= 0:
        return dataset, None
    if int(len(dataset) * val_size) < 1:
        logger.warning(
            "val_size=%.3f на %d сэмплах даёт пустую валидацию — обучаемся без eval.",
            val_size,
            len(dataset),
        )
        return dataset, None
    split = dataset.train_test_split(test_size=val_size, seed=seed, shuffle=True)
    return split["train"], split["test"]


def build_trainer(script_args, training_args, model_args) -> SFTTrainer:
    set_seed(training_args.seed)
    training_args.run_name = make_run_name(training_args, model_args)
    training_args.output_dir = str(Path(training_args.output_dir) / training_args.run_name)
    print(f"run: {training_args.run_name}\nвыход: {training_args.output_dir}")

    peft_config = get_peft_config(model_args)

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    dataset = load_sft_dataset(path=script_args.dataset_name)
    dataset = dataset.map(to_message)
    report = None
    if training_args.max_length is None:
        logger.warning(
            "max_length не задан: отбраковки по длине не будет. Один слишком "
            "длинный сэмпл может уронить ран по OOM."
        )
    else:
        dataset, report = filter_by_length(
            dataset,
            processor,
            max_length=training_args.max_length,
            num_proc=training_args.dataset_num_proc,
        )
        print(report.format())

    train_dataset, eval_dataset = _split_dataset(
        dataset, script_args.val_size, training_args.seed
    )
    if eval_dataset is None and training_args.eval_strategy != "no":
        logger.warning("Валидационного сплита нет — отключаю eval_strategy.")
        training_args.eval_strategy = "no"

    meta = {
        "model": model_args.model_name_or_path,
        "revision": model_args.model_revision,
        "dataset": script_args.dataset_name,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
        "max_length": training_args.max_length,
        "image_pixels": MAX_PIXELS,
        "image_tokens": visual_token_budget(processor),
        "vision_factor": processor.image_processor.patch_size
        * processor.image_processor.merge_size,
        "dropped_by_length": report.dropped if report else 0,
        "dropped_share": round(report.dropped_share, 4) if report else 0.0,
        "peft": bool(peft_config),
        "git_commit": git_commit(),
    }
    print(f"meta: {format_meta(meta)}")
    save_run_info(
        training_args.output_dir,
        script_args=script_args,
        training_args=training_args,
        model_args=model_args,
        meta=meta,
    )

    collate_fn = make_collate_fn(processor)

    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        dtype=model_args.dtype,
        attn_implementation=model_args.attn_implementation,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
        processing_class=processor,
    )
    return trainer


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = TrlParser((SftScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config(args=argv)
    trainer = build_trainer(script_args, training_args, model_args)
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
