"""
judge_client.py — pairwise LLM-judge метрика (в дополнение к 5 официальным
метрикам Design2Code в metrics.py).

Идея: у нас есть ДВЕ модели-кандидата (baseline и checkpoint, см.
--model-a/--model-b в run_benchmark_batched.py), каждая генерирует свой
pred.html/pred.png для одного и того же сэмпла. Судья (третья модель, напр.
Qwen3.5-4B, отдельный vLLM-процесс на своём GPU — см. serve_models.sh)
получает три картинки — ref.png (эталон), и два предсказания — и должен
сказать, какое из двух ближе к эталону.

Anti-position-bias рандомизация: LLM-судьи известны систематическим
предпочтением к позиции "первый вариант в промпте" независимо от реального
качества (см. напр. https://arxiv.org/abs/2306.05685 про position bias у
LLM-as-judge). Чтобы это не создавало системного перекоса в пользу baseline
или checkpoint (одна из моделей ВСЕГДА была бы "первой", если бы порядок был
фиксирован), для каждого сэмпла независимо подбрасывается монетка: кто из
двух — model_a или model_b — показывается судье как "Кандидат A", а кто как
"Кандидат B". Результат судьи (A/B) потом отображается обратно на
model_a/model_b через сохранённое присвоение, так что per-model win rate
считается корректно вне зависимости от того, где физически оказался ответ
каждой модели в конкретном промпте.

Формат ответа: structured outputs на стороне vLLM (extra_body с
structured_outputs.json схемой — см. ниже про guided_json) — так ответ
судьи гарантированно парсится, без ретраев на "мусорный" текстовый ответ
вида "I think A is better because...".

Схема НЕ включает поле reasoning (было раньше) — свободный текст-объяснение
перед winner давал судье возможность зациклиться в повторах ("Candidate A's
X is blue, while original is gray. Candidate A's Y is blue, ..." десятки
раз подряд на реальном прогоне) и упереться в judge_max_new_tokens ДО того,
как модель успевала закрыть JSON-строку и дойти до поля winner — structured
outputs гарантирует синтаксическую валидность JSON только пока генерация не
оборвана лимитом токенов, обрезанный на середине ответ всё равно ломает
json.loads (Unterminated string). Схема с одним полем winner намного короче
и не даёт этому случиться, плюс дешевле по токенам на judge-GPU.

ВАЖНО: параметр называется extra_body={"structured_outputs": {"json": ...}},
не extra_body={"guided_json": ...} — последний устарел в vLLM (см.
https://github.com/vllm-project/vllm/blob/main/docs/features/structured_outputs.md,
раздел про миграцию guided_* -> structured_outputs). На версиях vLLM, где
guided_json больше не поддерживается впрямую, старый параметр тихо
игнорируется БЕЗ ошибки — модель тогда генерирует произвольный JSON со
своими полями (например {"candidate_A": {...}, "candidate_B": {...}} вместо
{"winner": "A"|"B"}), из-за чего json.loads(content) в judge_one падает с
JSONDecodeError. Если в будущем при обновлении vLLM структура снова
сменится — первый признак будет именно такой: judge_one массово падает с
"Не удалось разобрать ответ судьи", а не с ошибкой соединения/EngineDeadError.
"""

import json
import random

from vllm_client import VLLMClient, image_path_message_part

JUDGE_PROMPT = (
    "You are an expert UI/UX evaluator. You will see three screenshots: "
    "the ORIGINAL target webpage, and two AI-generated recreations of it, "
    "labeled Candidate A and Candidate B (their order is randomized). "
    "Decide which candidate more faithfully reproduces the ORIGINAL's layout, "
    "text content, colors, and overall visual structure. "
    "Respond with a JSON object matching the schema, and nothing else."
)

# JSON Schema, передаётся в vLLM как guided_json (backend outlines/xgrammar на
# сервере гарантирует, что output будет валидным JSON, соответствующим схеме
# — т.е. поле winner будет ровно одним из "A"/"B", без необходимости парсить
# произвольный текст руками и повторять запрос при мусорном ответе).
JUDGE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "enum": ["A", "B"]},
    },
    "required": ["winner"],
}


def build_judge_messages(ref_png_path: str, cand_a_png_path: str, cand_b_png_path: str) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": JUDGE_PROMPT},
                {"type": "text", "text": "ORIGINAL (target):"},
                image_path_message_part(ref_png_path),
                {"type": "text", "text": "Candidate A:"},
                image_path_message_part(cand_a_png_path),
                {"type": "text", "text": "Candidate B:"},
                image_path_message_part(cand_b_png_path),
            ],
        }
    ]


class JudgeError(RuntimeError):
    pass


def judge_one(
    judge_client: VLLMClient,
    ref_png_path: str,
    pred_a_png_path: str,
    pred_b_png_path: str,
    model_a_name: str,
    model_b_name: str,
    rng: random.Random,
    max_tokens: int = 512,
) -> dict:
    """Один pairwise-сравнение. pred_a_png_path/pred_b_png_path соответствуют
    model_a_name/model_b_name СЕМАНТИЧЕСКИ (a=baseline, b=checkpoint, скажем)
    — но какая из картинок физически показывается судье как "Candidate A", а
    какая как "Candidate B" в промпте, решается броском монетки НИЖЕ (см.
    docstring модуля про anti-position-bias). Возвращает dict:
      {"winner_model": "<model_a_name|model_b_name>",  # ties не предусмотрены
       "swapped": bool,     # True если model_b была показана судье как "A"
       "raw_winner": "A"|"B"}
    Бросает JudgeError, если ответ судьи не удалось разобрать даже после
    structured outputs (не должно случаться на практике, но вызывающий код
    (judge_sample в run_benchmark_batched.py) трактует это как обычную
    judge_error — тот же принцип, что и метрики в process_sample_scoring)."""
    swapped = rng.random() < 0.5
    if not swapped:
        shown_a_path, shown_b_path = pred_a_png_path, pred_b_png_path
    else:
        shown_a_path, shown_b_path = pred_b_png_path, pred_a_png_path

    messages = build_judge_messages(ref_png_path, shown_a_path, shown_b_path)
    resp = judge_client.chat(
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

    # raw_winner относится к тому, что физически показали судье как "A"/"B" —
    # отображаем обратно на model_a_name/model_b_name с учётом swapped.
    if not swapped:
        winner_model = model_a_name if raw_winner == "A" else model_b_name
    else:
        winner_model = model_b_name if raw_winner == "A" else model_a_name

    return {
        "winner_model": winner_model,
        "swapped": swapped,
        "raw_winner": raw_winner,
    }