#!/usr/bin/env python3
"""merge_lora.py — влить LoRA-адаптер в базовую модель -> полная модель для vLLM.

Бенч (Evaluation/run_benchmark_batched.py) грузит модель через vLLM как единое целое,
БЕЗ поддержки LoRA-адаптеров, поэтому адаптер надо слить в полные веса заранее.

Запуск в SFT-контейнере (там есть peft/transformers):
    /opt/venv/bin/python -m scripts.merge_lora <adapter_dir> <out_dir>

  <adapter_dir> — папка рана (adapter_config.json + adapter_model.safetensors), напр.
                  sft-output/lora_ft_qwen3_vl_4b/lora_ft_qwen3_vl_4b_s42_20260727-170137
  <out_dir>     — куда сохранить полную модель. Чтобы её увидел eval-контейнер, клади
                  в HF-кэш (тот же путь монтируется в run.sh -> /root/.cache/huggingface),
                  напр. /root/.cache/huggingface/qwen3vl4b-drafting-merged
"""
import argparse
import json
import os
import shutil

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter", help="папка LoRA-рана (adapter_config.json + *.safetensors)")
    ap.add_argument("out", help="куда сохранить слитую полную модель (клади в HF-кэш)")
    ap.add_argument("--base", default=None,
                    help="id/путь базовой модели (по умолчанию берётся из adapter_config.json)")
    args = ap.parse_args()

    base = args.base
    if base is None:
        with open(os.path.join(args.adapter, "adapter_config.json")) as f:
            base = json.load(f)["base_model_name_or_path"]
    print(f"[merge] база:    {base}")
    print(f"[merge] адаптер: {args.adapter}")
    print(f"[merge] выход:   {args.out}")

    # merge на CPU: GPU обычно заняты, а слияние весов — арифметика, GPU не нужен.
    model = AutoModelForImageTextToText.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)

    # процессор рядом — чтобы vLLM грузил всё из одной папки
    AutoProcessor.from_pretrained(base).save_pretrained(args.out)

    # Переносим ссылку на ClearML-задачу обучения к слитым весам: бенч читает её
    # оттуда и связывает свой прогон с раном, из которого взят чекпоинт.
    link = os.path.join(args.adapter, "clearml_task.json")
    if os.path.exists(link):
        shutil.copy(link, os.path.join(args.out, "clearml_task.json"))
        print(f"[merge] перенесена ссылка на ClearML-задачу обучения")

    print(f"[merge] готово -> {args.out}  (укажи его в run.sh как --model)")


if __name__ == "__main__":
    main()
