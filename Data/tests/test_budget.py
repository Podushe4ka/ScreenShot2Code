"""Пиксельный бюджет: он общий с SFT, и разъехаться ему нельзя.

Расхождение этого бюджета — не косметика: оценка визуальных токенов едет в `tokens_total`,
по нему работает барьер `--max-total-tokens`, и заниженный потолок пропускает в набор
страницы, которые в окно не влезают.
"""
import os
import re

from common import budget


def test_defaults_match_sft():
    """Дефолты обязаны совпадать с SFT/train/formatting.py — читаем его исходник."""
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src = open(os.path.join(here, "SFT", "train", "formatting.py"), encoding="utf-8").read()
    got = dict(re.findall(r'(MIN_PIXELS|MAX_PIXELS)\s*=\s*int\(os\.environ\.get\('
                          r'"SFT_\w+",\s*([\d_]+)\)\)', src))
    assert got, "в SFT/train/formatting.py не нашлись MIN/MAX_PIXELS — тест устарел"
    assert int(got["MIN_PIXELS"].replace("_", "")) == budget.MIN_PIXELS
    assert int(got["MAX_PIXELS"].replace("_", "")) == budget.MAX_PIXELS


def test_ceiling_is_2048():
    """Потолок = MAX_PIXELS / patch^2. При 2.10 Мп и patch 32 это 2048, а НЕ 1672:
    1672 — старая конвенция patch=28 на потолке 1.31 Мп (до Tier A)."""
    assert budget.image_token_ceiling() == 2048
    assert budget.qwen_image_tokens(1280, 100_000) == 2048
    assert budget.qwen_image_tokens(99_999, 99_999) == 2048


def test_floor_is_min_pixels():
    """Маленькая картинка не бесплатна: площадь поднимается до MIN_PIXELS."""
    assert budget.qwen_image_tokens(1, 1) == budget.MIN_PIXELS // (budget.PATCH ** 2)
    assert budget.qwen_image_tokens(1, 1) == 256


def test_monotonic_between_floor_and_ceiling():
    prev = 0
    for h in (300, 600, 900, 1200, 1500, 1638, 2000):
        cur = budget.qwen_image_tokens(1280, h)
        assert cur >= prev
        prev = cur
    assert budget.qwen_image_tokens(1280, 720) == 900     # WebUI desktop
    assert budget.qwen_image_tokens(768, 1024) == 768     # WebUI tablet


def test_code_budget_matches_synth():
    """CODE_BUDGET_TOKENS синтетики выведен из этого же бюджета — числа должны сойтись."""
    assert budget.code_budget() == 14176
