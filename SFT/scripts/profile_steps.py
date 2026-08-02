"""Профиль нескольких шагов: куда именно уходит время на GPU.

Матрица throughput_matrix.sh отвечает «что быстрее», этот скрипт — «почему».
Показывает топ CUDA-ядер по суммарному времени: видно, съедает ли его линейное
внимание (Gated DeltaNet), обычное внимание, MLP или коммуникация NCCL.

    torchrun --nproc_per_node=2 -m scripts.profile_steps \\
        --config configs/full_ft_qwen3_5_4b.yaml --dataset_name /data/webcode2m_1000_split
"""

import torch
from torch.profiler import ProfilerActivity, profile, schedule
from transformers import TrainerCallback
from trl import ModelConfig, ScriptArguments, SFTConfig, TrlParser

from train.train_sft import build_trainer

WAIT, WARMUP, ACTIVE = 1, 1, 2
TOP_N = 25


class ProfileCallback(TrainerCallback):
    """Профилировщик живёт ровно WAIT+WARMUP+ACTIVE шагов, потом печатает отчёт.

    Первые шаги пропускаются намеренно: на них компилируются Triton-ядра и
    прогревается аллокатор, и они забивают собой весь топ.
    """

    def __init__(self, trace_path: str | None):
        self.trace_path = trace_path
        self.prof = None
        self.printed = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=WAIT, warmup=WARMUP, active=ACTIVE, repeat=1),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        self.prof.start()

    def on_step_end(self, args, state, control, **kwargs):
        if self.prof is None:
            return
        self.prof.step()
        if state.global_step >= WAIT + WARMUP + ACTIVE and not self.printed:
            self._report(args)
            control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        if self.prof is not None and not self.printed:
            self._report(args)

    def _report(self, args):
        self.printed = True
        self.prof.stop()
        if not args.should_log:
            self.prof = None
            return

        table = self.prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=TOP_N
        )
        print("\n================ топ ядер по self CUDA time ================")
        print(table)

        if self.trace_path:
            self.prof.export_chrome_trace(self.trace_path)
            print(f"\ntrace: {self.trace_path} (открыть в chrome://tracing или perfetto.dev)")
        self.prof = None


def main(argv=None):
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    parser.add_argument("--trace-path", default=None,
                        help="куда выгрузить chrome trace (по умолчанию не выгружать)")
    script_args, training_args, model_args, extra = parser.parse_args_and_config(
        args=argv, return_remaining_strings=True
    )

    trace_path = None
    for i, tok in enumerate(extra):
        if tok == "--trace-path" and i + 1 < len(extra):
            trace_path = extra[i + 1]
        elif tok.startswith("--trace-path="):
            trace_path = tok.split("=", 1)[1]

    if training_args.max_steps < 0 or training_args.max_steps > 10:
        training_args.max_steps = WAIT + WARMUP + ACTIVE + 1
    training_args.save_strategy = "no"
    training_args.eval_strategy = "no"
    training_args.logging_steps = 1

    trainer = build_trainer(script_args, training_args, model_args)
    trainer.add_callback(ProfileCallback(trace_path))
    trainer.train()

    if training_args.should_log:
        print(f"\nпик памяти на карту: {torch.cuda.max_memory_allocated() / 1024**3:.1f} ГБ")


if __name__ == "__main__":
    main()
