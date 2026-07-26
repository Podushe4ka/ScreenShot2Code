import logging
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


def _prepare_split(script_args, training_args, processor, split, required):
    """Прочитать сплит, разложить в messages и отбраковать по бюджету токенов.

    Возвращает (dataset|None, report|None).
    """
    dataset = load_sft_dataset(script_args.dataset_name, split, required=required)
    if dataset is None:
        return None, None

    dataset = dataset.map(to_message)
    if training_args.max_length is None:
        return dataset, None

    dataset, report = filter_by_length(
        dataset,
        processor,
        max_length=training_args.max_length,
        num_proc=training_args.dataset_num_proc,
    )
    return dataset, report


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

    if training_args.max_length is None:
        logger.warning(
            "max_length не задан: отбраковки по длине не будет. Один слишком "
            "длинный сэмпл может уронить ран по OOM."
        )

    # Замер бюджета и отбраковка идут ДО загрузки модели: узнать о том, что
    # половина датасета не влезает, лучше сразу, а не по OOM через двадцать
    # минут после начала скачивания весов.
    train_dataset, train_report = _prepare_split(
        script_args, training_args, processor, script_args.dataset_train_split, True
    )
    if train_report is not None:
        print(f"[{script_args.dataset_train_split}] {train_report.format()}")

    eval_dataset, eval_report = _prepare_split(
        script_args, training_args, processor, script_args.dataset_test_split, False
    )
    if eval_report is not None:
        print(f"[{script_args.dataset_test_split}] {eval_report.format()}")

    if eval_dataset is None and training_args.eval_strategy != "no":
        logger.warning(
            "Сплита '%s' нет — отключаю eval_strategy.", script_args.dataset_test_split
        )
        training_args.eval_strategy = "no"

    meta = {
        "model": model_args.model_name_or_path,
        "revision": model_args.model_revision,
        "dataset": script_args.dataset_name,
        "train_split": script_args.dataset_train_split,
        "eval_split": script_args.dataset_test_split if eval_dataset else None,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
        "max_length": training_args.max_length,
        "image_pixels": MAX_PIXELS,
        "image_tokens": visual_token_budget(processor),
        "vision_factor": processor.image_processor.patch_size
        * processor.image_processor.merge_size,
        "dropped_by_length": train_report.dropped if train_report else 0,
        "dropped_share": round(train_report.dropped_share, 4) if train_report else 0.0,
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
        # transformers v5 and TRL v1 both renamed torch_dtype -> dtype.
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
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config(args=argv)
    trainer = build_trainer(script_args, training_args, model_args)
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
