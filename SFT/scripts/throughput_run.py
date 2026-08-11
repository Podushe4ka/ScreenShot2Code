"""
    torchrun --nproc_per_node=2 -m scripts.throughput_run --config ... --max_steps 5
"""

from trl import ModelConfig, ScriptArguments, SFTConfig, TrlParser

from train.batching import BatchingArguments
from train.train_sft import build_trainer


def main(argv=None):
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, BatchingArguments))
    script_args, training_args, model_args, batching_args = (
        parser.parse_args_and_config(args=argv)
    )
    trainer = build_trainer(script_args, training_args, model_args, batching_args)
    trainer.train()


if __name__ == "__main__":
    main()
