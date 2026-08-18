"""Пиксельный и токенный бюджет — один на репозиторий.

⚠ ГЛАВНОЕ. `MIN_PIXELS`/`MAX_PIXELS` читаются из ТЕХ ЖЕ переменных окружения и с теми же
дефолтами, что `SFT/train/formatting.py`. Это не стиль, а починка ошибки: раньше конвертер
держал свои `MAX_PIXELS = 1280*32*32` (1.31 Мп) и оценивал картинку в 1280 токенов там, где
обучение видит 2048. Оценка едет в `tokens_total` (`webui/convert_parallel.py`), по нему
работает барьер `--max-total-tokens` в отборе — и высокая страница недобирала до 768
токенов, проходила барьер и не влезала в окно.

Хочешь другой бюджет — меняй переменные окружения, и он поменяется у ОБОИХ треков разом.
"""
import os
import sys

RENDER_WIDTH = 1280          # ширина вьюпорта ре-рендера; высота — по контенту (full_page)

MIN_PIXELS = int(os.environ.get("SFT_MIN_PIXELS", 262_144))      # 256*32*32
MAX_PIXELS = int(os.environ.get("SFT_MAX_PIXELS", 2_097_152))    # 2048*32*32 = 2.10 Мп (Tier A)

# patch_size * merge_size у Qwen3.5. Раньше стояло 28 (Qwen2.5-VL) — рассинхрон с
# MIN/MAX_PIXELS, считанными через 32: делитель 784 давал 1672 токена там, где их 1280.
# Число 1672 успело разойтись по документам; встретив его, читай как «до Tier A».
PATCH = 32

TOKENIZER_ID_DEFAULT = "Qwen/Qwen3-VL-8B-Instruct"

# Рабочее окно обучения: SFT/configs/*qwen3_5*.yaml. 8192 осталось только в легаси-конфиге
# lora_ft_qwen3_vl_4b_webcode2m.yaml.
WINDOW = 16384
PROMPT_OVERHEAD = 160        # шаблон промпта, грубая верхняя оценка

# Подсчёт токенов — ЛЕНИВЫЕ обёртки: token_len -> transformers -> torch тянутся только при
# реальном вызове, а не при импорте. Основной путь (сборка датасета, воркеры пула) их не грузит.
_EDA_TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "eda", "tools")


def _token_len():
    if _EDA_TOOLS not in sys.path:
        sys.path.append(_EDA_TOOLS)
    import token_len
    return token_len


def count_tokens(text, tokenizer):
    """Длина текста в токенах (без спец-токенов). Обёртка над eda/tools/token_len.py."""
    return _token_len().count_tokens(text, tokenizer)


def recommend_max_length(*args, **kwargs):
    """Ориентир max_length по списку длин. Обёртка над token_len.recommend_max_length."""
    return _token_len().recommend_max_length(*args, **kwargs)


def qwen_image_tokens(w, h, patch=PATCH):
    """Во сколько токенов процессор Qwen превратит картинку w x h.

    Площадь зажимается в [MIN_PIXELS, MAX_PIXELS], дальше 1 токен = patch x patch пикселей.
    При дефолтном бюджете: пол = 256 токенов, потолок = 2048.
    """
    px = min(max(w * h, MIN_PIXELS), MAX_PIXELS)
    return round(px / (patch * patch))


def image_token_ceiling(patch=PATCH):
    """Потолок визуальных токенов — сколько стоит любая достаточно большая картинка."""
    return round(MAX_PIXELS / (patch * patch))


def code_budget(window=WINDOW, overhead=PROMPT_OVERHEAD):
    """Сколько токенов окна остаётся коду в худшем случае (картинка упёрлась в потолок)."""
    return window - image_token_ceiling() - overhead
