"""Короткий прогон только ради замера скорости.

Отличие от train.train_sft: НЕ сохраняет веса в конце. Иначе матрица из
десятка экспериментов пишет на диск десяток полных моделей.

    torchrun --nproc_per_node=2 -m scripts.throughput_run --config ... --max_steps 5
"""

from trl import ModelConfig, ScriptArguments, SFTConfig, TrlParser

from train.train_sft import build_trainer


def main(argv=None):
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config(args=argv)
    trainer = build_trainer(script_args, training_args, model_args)
    trainer.train()


if __name__ == "__main__":
    main()
