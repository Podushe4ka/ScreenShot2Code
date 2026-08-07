"""Детектор «AI slop» в сгенерированных страницах.

Порт ключевых правил из навыка kill-ai-slop (github.com/yetone/kill-ai-slop,
`scripts/scan.mjs` + `references/taxonomy.md`) на Python: сканер там на чистом Node,
а node на машине сборки нет. Нумерация правил сохранена, чтобы можно было свериться
с `references/taxonomy.md`.

Зачем это в конвейере данных. Slop — это не «некрасиво», это КОРРЕЛЯЦИЯ: все страницы
скатываются к одному и тому же машинному дефолту (индиго-фиолетовый градиент, заголовок
с обрезкой градиента по тексту, кикер над каждым заголовком, ряд выдуманных чисел).
Для обучающего набора это прямой вред: разнообразие схлопывается, и модель заучивает
дефолт как правило. То есть slop-метрика здесь — измерение разнообразия, а не вкусовщина.

ВАЖНО: это ОТЧЁТ, а не отбраковка. Часть «тычков» справедлива для маркетингового
лендинга, но законна в приборной панели: полоски статусов и плитки с числами у нас
прямо заказаны в `features`, и там они несут настоящие данные, а не украшают пустоту.
Решение принимает человек, глядя на отчёт.
"""

import re

# (id, имя, паттерны). Отобраны те, что реально схлопывают разнообразие НАШИХ страниц.
TELLS = [
    ("01", "индиго→фиолетовый градиент", [
        r"from-(?:indigo|violet|purple|fuchsia)-\d+[\s\S]{0,60}?to-(?:purple|violet|fuchsia|pink)-\d+",
        r"(?:linear-gradient|bg-gradient)[^;\"'`]*(?:#6366f1|#8b5cf6|#a855f7|#7c3aed)",
    ]),
    ("02", "заголовок с обрезкой градиента по тексту", [
        r"bg-clip-text[\s\S]{0,40}?text-transparent|text-transparent[\s\S]{0,40}?bg-clip-text",
        r"(?:-webkit-)?background-clip:\s*text",
        r"-webkit-text-fill-color:\s*transparent",
    ]),
    ("06", "градиент как атмосфера", [
        r"radial-gradient",
        r"bg-gradient-to-[bt]\b[\s\S]{0,40}?from-",
        r"repeating-(?:linear|radial)-gradient",
    ]),
    ("10", "кикер над каждым заголовком", [
        r"\buppercase\b[\s\S]{0,40}?tracking-(?:wide|wider|widest)\b",
        r"tracking-(?:wide|wider|widest)\b[\s\S]{0,40}?\buppercase\b",
        r"text-transform:\s*uppercase[\s\S]{0,80}?letter-spacing:\s*0?\.\d+em",
    ]),
    ("11", "заголовок-предложение гигантским кеглем", [
        r"\btext-(?:5|6|7|8|9)xl\b",
        r"font-size:\s*(?:[5-9]\d(?:\.\d+)?px|[4-9](?:\.\d+)?rem)",
    ]),
    ("12", "плоская типографская иерархия", [
        r"<h[12][^>]*text-(?:sm|base|lg)\b",
    ]),
    ("14", "копирайтерский голос ИИ", [
        r"not just .{1,40}\bit(?:['’])?s\b",
        r"\b(?:say goodbye to|meet your new|supercharge|unlock the power of|in seconds,? not)\b",
        r"\b(?:blazing[- ]fast|effortless(?:ly)?|seamless(?:ly)?|game[- ]?changer|next[- ]level)\b",
        # русские кальки — из scripts/rules.ru.mjs того же навыка
        r"не просто .{1,40}?[—–-]\s*это",
        r"попрощайтесь с|забудьте о том|представьте себе мир",
        r"молниеносн|бесшовн|безупречн|революционн|беспрецедентн",
        r"раскройте (?:весь )?потенциал|в считанные секунды",
        r"выведите .{1,30} на новый уровень",
    ]),
    ("15", "эмодзи повсюду", [
        r"[\U0001F300-\U0001FAFF✨⚡✅\U0001F680]",
    ]),
    ("19", "максимальное скругление и стекло", [
        r"backdrop-blur|backdrop-filter:\s*blur",
        r"\brounded-(?:3xl|full)\b[\s\S]{0,40}?\brounded-(?:3xl|full)\b",
    ]),
    ("23", "спам бейджей и пилюль", [
        r"(?:✨|\U0001F525|β)\s*(?:New|Beta|Popular|Hot)\b",
        r"\brounded-full\b[^\"'\n]{0,50}\btext-xs\b[^\"'\n]{0,50}\b(?:px-2|px-3)\b",
    ]),
    ("30", "маркеры секций 01 / 02 / 03", [
        r">\s*0[1-9]\s*<[\s\S]{0,200}>\s*0[2-9]\s*<",
    ]),
    ("32", "один отступ на всё", [
        r"(?:\bgap-4\b[\s\S]*?){4,}",
        r"(?:\bp-4\b[\s\S]*?){6,}",
    ]),
    ("33", "Inter повсюду", [
        r"font-family:[^;]*\bInter\b",
        r"fonts\.googleapis[^\"']*Inter",
        r"\bSpace Grotesk\b",
    ]),
]

_COMPILED = [(tid, name, [re.compile(p, re.I) for p in pats]) for tid, name, pats in TELLS]


def scan(html: str) -> dict:
    """{id тычка: число совпадений}. Пусто — чисто."""
    out = {}
    for tid, name, pats in _COMPILED:
        n = sum(len(p.findall(html)) for p in pats)
        if n:
            out[tid] = n
    return out


def names() -> dict:
    return {tid: name for tid, name, _ in TELLS}
