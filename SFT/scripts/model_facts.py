"""Факты об архитектуре — без них MFU считается на глазок.

    python -m scripts.model_facts --config configs/full_ft_qwen3_5_4b.yaml
"""

import argparse
import json

import torch
import yaml
from transformers import AutoConfig, AutoModelForImageTextToText


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/full_ft_qwen3_5_4b.yaml")
    ap.add_argument("--load-weights", action="store_true",
                    help="посчитать параметры точно (грузит модель на CPU, медленно)")
    return ap.parse_args(argv)


def layer_type_histogram(cfg) -> dict:
    """Сколько слоёв какого типа — у гибридных моделей это главный вопрос."""
    for attr in ("layer_types", "layers_block_type", "block_types"):
        types = getattr(cfg, attr, None)
        if types:
            hist = {}
            for t in types:
                hist[str(t)] = hist.get(str(t), 0) + 1
            return hist
    return {}


def main(argv=None):
    args = parse_args(argv)
    raw = yaml.safe_load(open(args.config))
    model_id = raw["model_name_or_path"]
    revision = raw.get("model_revision")

    cfg = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    text_cfg = getattr(cfg, "text_config", cfg)

    facts = {
        "model": model_id,
        "hidden_size": getattr(text_cfg, "hidden_size", None),
        "num_hidden_layers": getattr(text_cfg, "num_hidden_layers", None),
        "intermediate_size": getattr(text_cfg, "intermediate_size", None),
        "num_attention_heads": getattr(text_cfg, "num_attention_heads", None),
        "num_key_value_heads": getattr(text_cfg, "num_key_value_heads", None),
        "vocab_size": getattr(text_cfg, "vocab_size", None),
        "max_position_embeddings": getattr(text_cfg, "max_position_embeddings", None),
        "layer_types": layer_type_histogram(text_cfg),
    }

    if args.load_weights:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision, dtype=torch.bfloat16, device_map="cpu"
        )
        total = sum(p.numel() for p in model.parameters())
        facts["params_total"] = total
        facts["params_billions"] = round(total / 1e9, 3)

    print(json.dumps(facts, ensure_ascii=False, indent=2))

    h = facts["hidden_size"]
    n = facts["num_hidden_layers"]
    if h and n:
        for seq in (8192, 16384):
            # активации без чекпоинтинга, оценка Korthikanti без члена внимания
            # (flash-attention не материализует матрицу s x s)
            gb = 34 * seq * h * n / 1024**3
            print(f"без gradient_checkpointing при seq={seq}, bs=1: ~{gb:.1f} ГБ активаций на карту")


if __name__ == "__main__":
    main()
