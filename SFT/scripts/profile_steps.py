import os

import torch
from torch.profiler import ProfilerActivity, profile, schedule
from transformers import TrainerCallback
from trl import ModelConfig, ScriptArguments, SFTConfig, TrlParser

from train.batching import BatchingArguments
from train.train_sft import build_trainer

WAIT, WARMUP, ACTIVE = 1, 1, 2
TOP_N = 25


class ProfileCallback(TrainerCallback):

    def __init__(self, trace_path: str | None, mem_snapshot: str | None = None, sync_debug: bool = False):
        self.trace_path = trace_path
        self.mem_snapshot = mem_snapshot
        self.sync_debug = sync_debug
        self.prof = None
        self.printed = False

    def on_train_begin(self, args, state, control, **kwargs):
        if self.mem_snapshot:
            torch.cuda.memory._record_memory_history(max_entries=200_000)
        self.prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=WAIT, warmup=WARMUP, active=ACTIVE, repeat=1),
            record_shapes=True,
            profile_memory=False,
            with_stack=True,
        )
        self.prof.start()

    def on_step_end(self, args, state, control, **kwargs):
        if self.prof is None:
            return
        self.prof.step()
        if self.sync_debug and state.global_step == WAIT + WARMUP:
            torch.cuda.set_sync_debug_mode('warn')
        if state.global_step >= WAIT + WARMUP + ACTIVE and not self.printed:
            torch.cuda.set_sync_debug_mode('default')
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

        stats = self.prof.key_averages()
        print("\n================ топ по self CUDA time ================")
        print(stats.table(sort_by="self_cuda_time_total", row_limit=TOP_N))
        print("\n================ топ по self CPU time ================")
        print(stats.table(sort_by="self_cpu_time_total", row_limit=TOP_N))

        if self.trace_path:
            self.prof.export_chrome_trace(self.trace_path)
            print(f"\ntrace: {self.trace_path} (открыть в chrome://tracing или perfetto.dev)")
        self.prof = None

        print("\n================ память ================")
        print(f"пик аллокаций : {torch.cuda.max_memory_allocated() / 1024**3:6.1f} ГиБ")
        print(f"пик резерва   : {torch.cuda.max_memory_reserved() / 1024**3:6.1f} ГиБ")
        print(f"сейчас занято : {torch.cuda.memory_allocated() / 1024**3:6.1f} ГиБ")

        if self.mem_snapshot:
            torch.cuda.memory._dump_snapshot(self.mem_snapshot)
            torch.cuda.memory._record_memory_history(enabled=None)
            print(f"снапшот: {self.mem_snapshot}")


def main(argv=None):
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, BatchingArguments))
    script_args, training_args, model_args, batching_args = (
        parser.parse_args_and_config(args=argv)
    )

    sync_debug = bool(os.environ.get("SYNC_DEBUG")) and training_args.process_index == 0
    trace_path = os.environ.get("TRACE_PATH") or None
    mem_snapshot = os.environ.get("MEM_SNAPSHOT") or None
    if mem_snapshot and training_args.process_index != 0:
        mem_snapshot = None

    if training_args.max_steps < 0 or training_args.max_steps > 10:
        training_args.max_steps = WAIT + WARMUP + ACTIVE + 1
    training_args.save_strategy = "no"
    training_args.eval_strategy = "no"
    training_args.logging_steps = 1

    trainer = build_trainer(script_args, training_args, model_args, batching_args)
    trainer.add_callback(ProfileCallback(trace_path, mem_snapshot, sync_debug))
    trainer.train()


if __name__ == "__main__":
    main()
