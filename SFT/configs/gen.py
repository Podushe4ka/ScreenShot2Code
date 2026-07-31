"""Генератор тренировочных конфигов.

Запускать из корня SFT/:  python -m configs.gen

`max_length` здесь вычисляется, а не задаётся числом: он складывается из длины
кода, накладных расходов промпта и визуального бюджета картинки, а последний
зависит от модели. Одно и то же число на все модели неизбежно оказывается
неверным для части из них.
"""

import math
from pathlib import Path

import yaml

from train.formatting import MAX_PIXELS

OUT = Path("configs")

# Сколько карт в запуске. Нужно здесь, потому что эффективный батч —
# per_device_bs * accum * N_GPUS, и без учёта карт accumulation считается
# неверно. При смене железа поменять.
N_GPUS = 2

# Эффективный батч (примеров на шаг оптимизатора), из него выводится
# gradient_accumulation_steps.
TARGET_EFF_BATCH = 64

CODE_P99_TOKENS = 896

PROMPT_OVERHEAD_TOKENS = 160

MAX_LENGTH_ROUND_TO = 64



TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
SHARED_PARAMS = {
    "dtype": "bfloat16",
    "attn_implementation": "flash_attention_2",
    "dataset_num_proc": 16,
    "remove_unused_columns": False,
    "dataset_train_split": "train",
    "dataset_test_split": "validation",
    "dataloader_num_workers": 8,
    # Бакетинг по длине: батч собирается из сэмплов близкой длины, паддинг до
    # самого длинного в батче почти ничего не съедает. Читает колонку `length`,
    # которую проставляет filter_by_length. В transformers 5.x это пришло на
    # смену флагу group_by_length.
    "train_sampling_strategy": "group_by_length",
    "tf32": True,
    "num_train_epochs": 3,
    "warmup_ratio": 0.03,
    "optim": "adamw_torch_fused",
    "lr_scheduler_type": "cosine",
    "seed": 42,
    "logging_steps": 10,
    "eval_strategy": "steps",
    "eval_steps": 0.25,
    "save_strategy": "steps",
    "save_steps": 0.2,
    "save_total_limit": 2,
    "save_only_model": True,
    "bf16": True,
    "fp16": False,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "report_to": "none",
}

MODELS = {
    # "qwen2_5_vl_3b": {
    #     "id": "Qwen/Qwen2.5-VL-3B-Instruct",
    #     "rev": "66285546d2b821cf421d4f5eb2576359d3770cd3",
    #     "factor": 28,         
    #     "lora_bs": 16, "full_zero": 2,
    # },
    "qwen3_5_4b": {
        "id": "Qwen/Qwen3.5-4B",
        "rev": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "factor": 32,
        "lora_bs": 16, "full_zero": 2,
        # Проверено на 2xA100 80 ГБ: bs=8 и bs=4 без чекпоинтинга падают по
        # памяти. Причина не в активациях, а в словаре: 248320 токенов дают
        # логиты (bs, 2368, 248320) — 4.4 ГиБ в bf16 при bs=4, вдвое больше
        # при upcast в fp32 для лосса, плюс градиент.
        "full_bs": 2, "full_gc": True,
    },
    # "qwen3_vl_4b": {
    #     "id": "Qwen/Qwen3-VL-4B-Instruct",
    #     "rev": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
    #     "factor": 32,
    #     "lora_bs": 16, "full_zero": 2,
    #     # Замерено на 2xA100 80 ГБ (full FT, ZeRO-2, max_length 2368):
    #     #   bs=4 + чекпоинтинг         -> 20.1 с/шаг, 3.2 примера/с   (рабочий)
    #     #   bs=8 + чекпоинтинг         -> 19.9 с/шаг                  (без выигрыша)
    #     #   bs=4 без чекпоинтинга      -> OOM
    #     # GPU util 99%, то есть упор в вычисления: батч и оптимизатор скорость
    #     # не меняют. bs=4 оставлен как более безопасный по памяти.
    #     "full_bs": 4, "full_gc": True,
    # },
    # "qwen2_5_vl_7b": {
    #     "id": "Qwen/Qwen2.5-VL-7B-Instruct",
    #     "rev": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    #     "factor": 28,         
    #     "lora_bs": 8, "full_zero": 2,
    # },
    # "qwen3_vl_8b": {
    #     "id": "Qwen/Qwen3-VL-8B-Instruct",
    #     "rev": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    #     "factor": 32,
    #     "lora_bs": 8, "full_zero": 2,
    # },
    "qwen3_5_9b": {
        "id": "Qwen/Qwen3.5-9B",
        "rev": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "factor": 32,
        "lora_bs": 8, "full_zero": 2,
    },
    "qwen3_5_27b": {
        "id": "Qwen/Qwen3.5-27B",
        "rev": "fc05daec18b0a78c049392ed2e771dde82bdf654",
        "factor": 32,
        "lora_bs": 2, "full_zero": 3,
    },
}


DEEPSPEED = {2: "configs/deepspeed_zero2.json", 3: "configs/deepspeed_zero3.json"}


def _accum(microbatch: int) -> int:
    return max(1, TARGET_EFF_BATCH // (microbatch * N_GPUS))


def visual_tokens(factor: int) -> int:
    """Визуальные токены на картинку при бюджете MAX_PIXELS."""
    return math.ceil(MAX_PIXELS / factor**2)


def max_length_for(m: dict) -> int:
    """Бюджет контекста: код + промпт + картинка, округлённый вверх.

    Явный `max_length` в описании модели перекрывает расчёт.
    """
    if "max_length" in m:
        return m["max_length"]
    total = CODE_P99_TOKENS + PROMPT_OVERHEAD_TOKENS + visual_tokens(m["factor"])
    return math.ceil(total / MAX_LENGTH_ROUND_TO) * MAX_LENGTH_ROUND_TO


def lora_cfg(name: str, m: dict) -> dict:
    bs = m["lora_bs"]
    return {
        "model_name_or_path": m["id"],
        "model_revision": m["rev"],
        **SHARED_PARAMS,
        "max_length": max_length_for(m),
        "output_dir": f"sft-output/lora_ft_{name}",
        "run_name": f"lora_ft_{name}",
        "per_device_train_batch_size": bs,
        "per_device_eval_batch_size": bs,
        "gradient_accumulation_steps": _accum(bs),
        "learning_rate": 2.0e-4,
        "weight_decay": 0.01,
        "use_peft": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": TARGET_MODULES,
    }


def full_cfg(name: str, m: dict) -> dict:
    lr = 1.0e-5 if m["lora_bs"] <= 2 else 2.0e-5
    bs = m.get("full_bs", 1)
    return {
        "model_name_or_path": m["id"],
        "model_revision": m["rev"],
        **SHARED_PARAMS,
        "max_length": max_length_for(m),
        "output_dir": f"sft-output/full_ft_{name}",
        "run_name": f"full_ft_{name}",
        "per_device_train_batch_size": bs,
        "per_device_eval_batch_size": bs,
        "gradient_accumulation_steps": _accum(bs),
        "gradient_checkpointing": m.get("full_gc", True),
        "learning_rate": lr,
        "weight_decay": 0.05,
        "deepspeed": DEEPSPEED[m["full_zero"]],
    }


def main():
    for name, m in MODELS.items():
        (OUT / f"lora_ft_{name}.yaml").write_text(
            yaml.safe_dump(lora_cfg(name, m), sort_keys=False)
        )
        (OUT / f"full_ft_{name}.yaml").write_text(
            yaml.safe_dump(full_cfg(name, m), sort_keys=False)
        )
        print(
            f"{name}: factor={m['factor']}, "
            f"картинка={visual_tokens(m['factor'])} ток., "
            f"max_length={max_length_for(m)}"
        )
    print(f"wrote {2 * len(MODELS)} configs to {OUT}/")


if __name__ == "__main__":
    main()
