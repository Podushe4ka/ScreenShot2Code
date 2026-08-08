"""
prompts.py — варианты промпта судьи. Добавляй сюда новые варианты по мере
экспериментов на train, каждый под своим ключом; run_judge_eval.py выбирает
нужный через --prompt-key.

Держать их все в одном месте (а не разбросанными по CLI-аргументам)
удобно, потому что после подбора финального промпта на train легко
посмотреть diff между версиями и понять, что именно сработало.
"""

PROMPTS: dict[str, str] = {
    "v1_baseline": (
        "You are an expert UI/UX evaluator. You will see three screenshots: "
        "the ORIGINAL target webpage, and two AI-generated recreations of it, "
        "labeled Candidate A and Candidate B (their order is randomized). "
        "Decide which candidate more faithfully reproduces the ORIGINAL's layout, "
        "text content, colors, and overall visual structure. "
        "Respond with a JSON object matching the schema, and nothing else."
    ),
    "v2_criteria_list": (
        "You are an expert front-end engineer reviewing two AI-generated HTML "
        "recreations of a target webpage screenshot.\n\n"
        "You will see: the ORIGINAL target screenshot, then Candidate A, then "
        "Candidate B (order randomized).\n\n"
        "Judge strictly by visual fidelity to the ORIGINAL, in this priority order:\n"
        "1. Layout structure (position and size of major blocks/sections)\n"
        "2. Text content accuracy (is the same text present, roughly in the same place)\n"
        "3. Color scheme (background, text, accent colors)\n"
        "4. Typography and spacing details\n\n"
        "Pick the candidate that is closer overall. Respond with a JSON object "
        "matching the schema, and nothing else."
    ),
    "v3_human": (
        "Look at the ORIGINAL webpage and the two recreations. Ask yourself: "
        "\"Which one feels more like the original?\" Choose the better match based "
        "on the overall visual impression, considering layout, content, colors, "
        "and styling together. Respond with a JSON object matching the schema, "
        "and nothing else."
    ),
}


def get_prompt(key: str) -> str:
    if key not in PROMPTS:
        raise KeyError(
            f"Промпт '{key}' не найден. Доступные ключи: {sorted(PROMPTS.keys())}"
        )
    return PROMPTS[key]
