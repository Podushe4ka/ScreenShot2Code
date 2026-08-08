"""
judge_client.py — pairwise LLM-judge: показывает судье ref.png +
pred_baseline.png + pred_checkpoint.png и просит выбрать, какой из двух
кандидатов ближе к эталону.

Anti-position-bias: LLM-судьи склонны предпочитать "первый вариант в
промпте" независимо от реального качества. Чтобы это не создавало
систематического перекоса в пользу baseline или checkpoint, порядок показа
рандомизируется — какая картинка физически идёт как "Candidate A", а какая
как "Candidate B", решает swapped_for_sample() из labeler.py (тот же
детерминированный хэш от sample_id — так что порядок показа судье
СОВПАДАЕТ с тем порядком, что видел человек-разметчик при исходной
разметке; это не обязательно строго необходимо, но упрощает сверку при
ручной проверке отдельных расхождений).

Формат ответа: structured outputs (extra_body={"structured_outputs": {"json": ...}})
— гарантирует, что content будет валидным JSON вида {"winner": "A"|"B"}, без
свободного текста-рассуждения, который может зациклиться и не успеть
закрыть JSON до max_tokens.
"""

import json
import random
from pathlib import Path

from vllm_client import VLLMClient, image_path_message_part

JUDGE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "enum": ["A", "B"]},
    },
    "required": ["winner"],
}


class JudgeError(RuntimeError):
    pass


def build_judge_messages(prompt_text: str, ref_path: Path, cand_a_path: Path, cand_b_path: Path) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "text", "text": "ORIGINAL (target):"},
                image_path_message_part(ref_path),
                {"type": "text", "text": "Candidate A:"},
                image_path_message_part(cand_a_path),
                {"type": "text", "text": "Candidate B:"},
                image_path_message_part(cand_b_path),
            ],
        }
    ]


def judge_one(
    client: VLLMClient,
    prompt_text: str,
    sample_dir: Path,
    swapped: bool,
    max_tokens: int = 128,
) -> dict:
    """Один запрос к судье для одного сэмпла.

    swapped решает, что физически показывается как "Candidate A": если
    swapped=False, A=baseline и B=checkpoint; если True — наоборот.
    Прокидывается вызывающим кодом (run_judge_eval.py), а не считается
    здесь заново, чтобы вызывающий код мог логировать точный порядок показа
    вместе с остальными полями результата.

    Возвращает dict:
        {"winner_model": "baseline"|"checkpoint", "raw_winner": "A"|"B"}
    Бросает JudgeError, если ответ не распарсился (вызывающий код трактует
    это как отдельный сэмпл с ошибкой, не роняя весь прогон)."""
    pred_a_path = sample_dir / ("pred_checkpoint.png" if swapped else "pred_baseline.png")
    pred_b_path = sample_dir / ("pred_baseline.png" if swapped else "pred_checkpoint.png")
    ref_path = sample_dir / "ref.png"

    messages = build_judge_messages(prompt_text, ref_path, pred_a_path, pred_b_path)
    resp = client.chat(
        messages,
        max_tokens=max_tokens,
        temperature=0.0,
        extra_body={"structured_outputs": {"json": JUDGE_JSON_SCHEMA}},
    )
    content = resp["choices"][0]["message"]["content"]

    try:
        parsed = json.loads(content)
        raw_winner = parsed["winner"]
        if raw_winner not in ("A", "B"):
            raise ValueError(f"winner вне схемы: {raw_winner!r}")
    except Exception as e:
        raise JudgeError(f"Не удалось разобрать ответ судьи: {content!r} ({e})")

    if not swapped:
        winner_model = "baseline" if raw_winner == "A" else "checkpoint"
    else:
        winner_model = "checkpoint" if raw_winner == "A" else "baseline"

    return {"winner_model": winner_model, "raw_winner": raw_winner}
