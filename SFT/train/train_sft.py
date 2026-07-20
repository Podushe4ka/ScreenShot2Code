from trl import ModelConfig, ScriptArguments, SFTConfig, TrlParser, SFTTrainer, get_peft_config
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers import set_seed
from train.formatting import to_message, make_collate_fn, MIN_PIXELS, MAX_PIXELS
from data.loader import load_sft_dataset


def build_trainer(script_args, training_args, model_args) -> SFTTrainer:
    set_seed(42)

    peft_config = get_peft_config(model_args)

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    collate_fn = make_collate_fn(processor)

    dataset = load_sft_dataset(path=script_args.dataset_name)
    dataset = dataset.map(to_message)

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
        train_dataset=dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
        processing_class=processor,
    )
    return trainer



def main(argv=None):
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config(args=argv)
    trainer = build_trainer(script_args, training_args, model_args)
    trainer.train()
    trainer.save_model(training_args.output_dir)
    

if __name__ == '__main__':
    main()
