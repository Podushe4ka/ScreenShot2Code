"""Собирает briefs.jsonl — детерминированный набор ТЗ на генерацию страниц.

Две независимые задачи, которые здесь СОЗНАТЕЛЬНО разделены:

1. ЧТО на странице (содержание) — берётся из реальных сидов, не выдумывается.
   Обоснование в плане: у Flame (Data/papers/Flame_2503.01619.pdf, стр. 7) замерено, что
   модели на структурных синтезах (Waterfall / Additive) бьют модель на Evolution-синтезе,
   потому что случайные вариации вносят "noise and unnatural variations that hinder
   learning robust coding patterns". Поэтому случайный розыгрыш содержания — плохая идея,
   и содержание приходит из готовых человеческих ТЗ.

2. КАКИМ его сделать (тир плотности, язык, impl, hard-features) — сетка ПОКРЫТИЯ.
   Это не розыгрыш: доли задаются точно и раскладываются круговым распределением,
   чтобы в наборе из 300 страниц каждая страта была представлена ровно столько раз,
   сколько заказано. При случайном розыгрыше на N=300 редкие страты выпадают целиком.

Фильтр страничности. Замер по всем 157 903 строкам Flame-Evo-React: 43% layout_description
описывают страничные структуры (header/footer/sidebar/hero/секции), 47.7% — одиночный
компонент («a single RadioGroup component», «a centered icon»). Нам нужны страницы, поэтому
компонентные строки отсеиваются: одиночный компонент даёт скриншот, по которому нечему
учиться на уровне раскладки.

Отбор разнообразия — farthest-point sampling по TF-IDF, а не k-means: при N=300 из десятков
тысяч кандидатов нам нужна МАКСИМАЛЬНАЯ взаимная непохожесть отобранного, а это ровно то,
что max-min оптимизирует напрямую. K-means оптимизирует другое (компактность кластеров) и
на длинном хвосте редких тем склонен отдавать центроиды больших однородных групп.

Использование:
    .venv/bin/python Data/generators/synth/make_briefs.py --n 300
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

SEED = 42

# --- сетка покрытия -------------------------------------------------------------

# Доли выбраны так: react_cdn — основной стиль (решение пользователя, он же стиль
# UI2Code^N), static_inline — меньшинство, но достаточное, чтобы модель не забыла
# самодостаточный HTML, которым сформулирован текущий промпт бенча.
IMPL_MIX = {"react_cdn": 0.65, "static_inline": 0.35}

# Тир плотности — главная ось «сложности». Именно тира L нет ни в одном синтетическом
# наборе: WebSight после precompile_tailwind даёт p99 3815 токенов, то есть сплошь S.
#
# `nodes` — критерий приёмки (язык- и impl-нейтрален). `bytes` — только подсказка
# генератору о масштабе: байт-на-узел зависит и от impl (на калибровке react на тире L
# дал 31 Б/узел против 51 у static — данные разворачиваются через .map()), и от языка
# (UTF-8 добавляет 8-13% на кириллице и CJK). Верхняя граница тира L подрезана с 40 000
# до 34 000: при замеренных ~2.6 Б/токен это ~13 000 токенов, что оставляет запас к
# бюджету кода 14 176 из SFT/configs/gen.py.
TIERS = {
    "S": {"share": 0.25, "nodes": (40, 80), "bytes": (3_000, 8_000)},
    "M": {"share": 0.45, "nodes": (150, 300), "bytes": (8_000, 20_000)},
    "L": {"share": 0.30, "nodes": (400, 900), "bytes": (18_000, 34_000)},
}

# Язык: метрика Design2Code на 20% состоит из text-score, а бенч содержит неанглийские
# страницы. Модель, видевшая только английский, на них проседает.
LANG_MIX = {"en": 0.55, "ru": 0.15, "de": 0.10, "zh": 0.10, "ja": 0.10}

# Визуальный стиль — ОРТОГОНАЛЬНАЯ содержанию ось, и без неё набор вырождается.
# Сиды (Flame, UIGEN) дают разнообразие ТЕМ, но не ВИДА: и те и другие описывают
# «что на странице», а не «как она выглядит», и модель-генератор по умолчанию рисует
# один и тот же светлый SaaS-интерфейс. На первых 19 страницах пилота это было видно
# на контактном листе невооружённым глазом. Тема и стиль независимы: складской учёт
# бывает и тёмной консолью, и пастельным дашбордом.
VISUAL_MIX = {
    "clean_light_saas": 0.20,
    "dense_enterprise": 0.16,
    "dark_dashboard": 0.14,
    "editorial_serif": 0.11,
    "brutalist_blocks": 0.10,
    "terminal_mono": 0.08,
    "pastel_soft": 0.08,
    "glassmorphism": 0.07,
    "high_contrast_mono": 0.06,
}

# Hard-features: то, что реально тяжело сверстать и что метрика видит.
# Иконки — ТОЛЬКО инлайновый <svg>: шрифты FontAwesome не вендорятся (см. fetch_vendor.sh),
# классы fa-* дали бы пустые квадраты в рендере.
FEATURES = [
    "nested_tables", "inline_svg_icons", "css_gradients", "layered_shadows",
    "sticky_header", "z_index_overlap", "multilevel_nav", "status_badges",
    "progress_bars", "star_ratings", "breadcrumbs", "form_validation_states",
    "zebra_table", "pagination", "avatar_stack", "timeline_dots",
    "pure_css_chart", "inline_svg_chart", "tabs_strip", "stat_tiles",
]

# Структурно тяжёлые фичи: в 40-80 узлов тира S они физически не помещаются, а
# требование «сделай многоуровневую навигацию в 60 узлов» гарантирует брак на линте.
HEAVY_FEATURES = frozenset({
    "nested_tables", "multilevel_nav", "pagination", "zebra_table",
    "sticky_header", "tabs_strip",
})

# Слова, по которым оценивается СОБСТВЕННАЯ сложность темы. Нужны, чтобы тир не
# назначался вслепую: иначе простое todo-приложение уезжает в тир L (400-900 узлов) и
# генератор вынужден раздувать его искусственно, а админка с фильтрами попадает в тир S
# и её приходится кромсать. И то и другое — те самые «unnatural variations», из-за
# которых Evolution-синтез у Flame проиграл структурным (стр. 7).
_HEAVY_WORDS = re.compile(
    r"(?i)\b(dashboard|admin|analytics|table|tables|filter|filtering|sortable|"
    r"sidebar|multi-?level|nested|report|reports|catalog|checkout|marketplace|"
    r"portal|workspace|kanban|calendar|inventory|management|permissions|"
    r"pagination|charts?|graphs?|metrics|statistics|columns?)\b"
)
_LIGHT_WORDS = re.compile(
    r"(?i)\b(simple|single|minimal|basic|small|one |card|badge|button|icon|"
    r"spinner|toggle|avatar|tooltip|snippet|widget)\b"
)


def complexity_score(text: str) -> float:
    """Грубая оценка «насколько тема сама по себе большая». Только для ранжирования."""
    heavy = len(set(m.group(0).lower() for m in _HEAVY_WORDS.finditer(text)))
    light = len(set(m.group(0).lower() for m in _LIGHT_WORDS.finditer(text)))
    return heavy - 0.8 * light

# --- фильтр страничности --------------------------------------------------------

_PAGE_RE = re.compile(
    r"(?i)\b(header|footer|navbar|navigation bar|sidebar|hero|landing|dashboard|"
    r"sections?|main content|top bar|banner|grid of|list of)\b"
)
_COMPONENT_RE = re.compile(
    r"(?i)(\bpage (consists of|features|contains) (a |an )?(single|one)\b|"
    r"\ba single [a-z]+ component\b|\bcentered icon\b|\bonly one component\b)"
)
_PLACEHOLDER_TD = "write a js code that may render a webpage like this photo"


def is_page_level(task_description: str, layout_description: str) -> bool:
    ld = layout_description or ""
    if len(ld) < 300:
        return False
    if _COMPONENT_RE.search(ld):
        return False
    # Считаем «страничным», если упомянуто хотя бы два разных структурных ориентира —
    # одного слова «section» мало, оно встречается и в описании одиночной карточки.
    return len(set(m.group(0).lower() for m in _PAGE_RE.finditer(ld))) >= 2


# --- TF-IDF + farthest-point sampling -------------------------------------------

_TOKEN_RE = re.compile(r"[a-z][a-z0-9_-]{2,}")
_STOP = frozenset("""the and for with that this from are was were has have had not but
its it's you your they them their there here which who whom whose will would can could
should page component components element elements likely overall design layout section
sections color colors text background top bottom left right center centered contains
consists featuring features displayed display appears positioned""".split())


def hashed_tfidf(docs: list[str], dim: int = 2048) -> np.ndarray:
    """TF-IDF с хешированием признаков в фиксированную размерность.

    Хеширование вместо словаря: полный словарь по десяткам тысяч описаний даёт матрицу,
    которую в плотном виде не удержать, а ради отбора 300 непохожих строк точный словарь
    не нужен — коллизии на dim=2048 размывают меру сходства пренебрежимо.
    """
    n = len(docs)
    tf = np.zeros((n, dim), dtype=np.float32)
    df = np.zeros(dim, dtype=np.float32)
    for i, doc in enumerate(docs):
        counts = Counter(
            t for t in _TOKEN_RE.findall(doc.lower()) if t not in _STOP
        )
        seen = set()
        for tok, c in counts.items():
            j = int(hashlib.blake2b(tok.encode(), digest_size=4).hexdigest(), 16) % dim
            tf[i, j] += 1.0 + np.log(c)  # сублинейный tf
            seen.add(j)
        for j in seen:
            df[j] += 1.0
    idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
    x = tf * idf
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def farthest_point_sample(x: np.ndarray, k: int, seed: int = SEED) -> list[int]:
    """Жадный max-min отбор k строк: каждая следующая максимально далека от уже взятых.

    Векторы L2-нормированы, поэтому косинусное сходство = скалярное произведение,
    а расстояние монотонно ему обратно — держим максимум сходства с выбранными
    и на каждом шаге берём точку с МИНИМАЛЬНЫМ таким максимумом.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    k = min(k, n)
    first = int(rng.integers(n))
    chosen = [first]
    max_sim = x @ x[first]
    for _ in range(k - 1):
        max_sim[chosen] = np.inf
        nxt = int(np.argmin(max_sim))
        chosen.append(nxt)
        np.maximum(max_sim, x @ x[nxt], out=max_sim)
    return chosen


# --- сетка покрытия -------------------------------------------------------------

def exact_allocation(mix: dict, n: int) -> list:
    """Раскладывает n элементов по долям ТОЧНО (метод наибольших остатков).

    Не round(share*n) по отдельности: суммы бы не сошлись с n, и последняя страта
    молча собирала бы остаток.
    """
    raw = {k: v * n for k, v in mix.items()}
    base = {k: int(np.floor(v)) for k, v in raw.items()}
    rest = n - sum(base.values())
    for k, _ in sorted(raw.items(), key=lambda kv: -(kv[1] - np.floor(kv[1]))):
        if rest <= 0:
            break
        base[k] += 1
        rest -= 1
    out = []
    for k, c in base.items():
        out.extend([k] * c)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="Data/synth_pilot/seeds")
    ap.add_argument("--out", default="Data/synth_pilot/briefs.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--pool", type=int, default=15000,
                    help="сколько кандидатов подавать в отбор разнообразия")
    args = ap.parse_args()

    seeds_dir = Path(args.seeds)
    rng = np.random.default_rng(SEED)

    # --- кандидаты из Flame: страничные, с осмысленным ТЗ ---
    flame = []
    n_total = n_page = 0
    for line in (seeds_dir / "flame_evo_react.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        n_total += 1
        td = (r.get("task_description") or "").strip()
        ld = (r.get("layout_description") or "").strip()
        if not is_page_level(td, ld):
            continue
        n_page += 1
        # Шаблон-заглушку (5.8% строк) как ТЗ не используем — она не несёт информации,
        # содержание в таких строках целиком в layout_description.
        if _PLACEHOLDER_TD in td.lower():
            td = ""
        flame.append({"seed_source": "flame_evo_react", "seed_id": str(r.get("id")),
                      "task_description": td, "layout_description": ld})
    print(f"Flame: {n_total} строк -> {n_page} страничных ({100*n_page/n_total:.1f}%)")

    # --- кандидаты из UIGEN-T3: короткие запросы, другой регистр формулировок ---
    uigen = []
    for line in (seeds_dir / "uigen_t3.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        q = (r.get("Question") or "").strip()
        if len(q) < 40:
            continue
        uigen.append({"seed_source": "uigen_t3", "seed_id": str(r.get("id")),
                      "task_description": q, "layout_description": ""})
    print(f"UIGEN-T3: {len(uigen)} запросов длиннее 40 символов")

    # Пул под отбор: Flame прореживаем (их десятки тысяч), UIGEN берём целиком —
    # он и так на порядок меньше, а формулировки в нём ценные.
    if len(flame) > args.pool:
        idx = rng.choice(len(flame), size=args.pool, replace=False)
        flame = [flame[i] for i in sorted(idx)]
    pool = flame + uigen
    print(f"пул кандидатов: {len(pool)}")

    docs = [f"{c['task_description']} {c['layout_description']}".strip() for c in pool]
    x = hashed_tfidf(docs)
    picked = farthest_point_sample(x, args.n)
    chosen = [pool[i] for i in picked]

    # --- сетка покрытия: точные доли ---
    impls = exact_allocation(IMPL_MIX, args.n)
    langs = exact_allocation(LANG_MIX, args.n)
    # impl и lang от содержания не зависят — перемешиваем, иначе они окажутся
    # скоррелированы с порядком отбора (а он идёт от «самых непохожих» к остальным).
    for arr in (impls, langs):
        rng.shuffle(arr)

    # Визуальный стиль тянем из ОТДЕЛЬНОГО генератора. Иначе добавление новой оси
    # сдвинуло бы последовательность основного rng, и у всех уже сгенерированных страниц
    # поменялись бы features — пришлось бы перегенерировать готовое.
    style_rng = np.random.default_rng(SEED + 1)
    visuals = exact_allocation(VISUAL_MIX, args.n)
    style_rng.shuffle(visuals)

    # Тир, наоборот, ПРИВЯЗЫВАЕМ к содержанию: ранжируем темы по собственной сложности
    # и режем по границам долей. Доли при этом соблюдаются точно — меняется только то,
    # какая тема какой тир получит. См. комментарий к complexity_score.
    order = sorted(
        range(len(chosen)),
        key=lambda i: complexity_score(
            f"{chosen[i]['task_description']} {chosen[i]['layout_description']}"
        ),
    )
    tier_slots = exact_allocation({k: v["share"] for k, v in TIERS.items()}, args.n)
    tier_slots.sort(key=lambda t: ["S", "M", "L"].index(t))  # от простых к сложным
    tiers = [None] * len(chosen)
    for rank, idx in enumerate(order):
        tiers[idx] = tier_slots[rank]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as w:
        for i, c in enumerate(chosen):
            tier = tiers[i]
            n_feat = int(rng.integers(2, 5))  # 2..4
            # Тир S не получает структурно тяжёлых фич (их некуда положить в 40-80
            # узлов); тир L, наоборот, обязан получить хотя бы одну — иначе «большой»
            # тир вырождается в длинную простыню одинаковых карточек.
            pool_f = [f for f in FEATURES if tier != "S" or f not in HEAVY_FEATURES]
            feats = set(rng.choice(pool_f, size=n_feat, replace=False).tolist())
            if tier == "L" and not (feats & HEAVY_FEATURES):
                feats.discard(sorted(feats)[0])
                feats.add(str(rng.choice(sorted(HEAVY_FEATURES))))
            feats = sorted(feats)
            brief = {
                "id": f"p{i:04d}",
                "seed_source": c["seed_source"],
                "seed_id": c["seed_id"],
                "task_description": c["task_description"],
                "layout_description": c["layout_description"],
                "impl": impls[i],
                "tier": tier,
                "dom_nodes": list(TIERS[tier]["nodes"]),
                "target_bytes": list(TIERS[tier]["bytes"]),
                "lang": langs[i],
                "visual": visuals[i],
                "features": feats,
            }
            w.write(json.dumps(brief, ensure_ascii=False) + "\n")

    print(f"\nзаписано {len(chosen)} брифов -> {out_path}")
    for name, arr in (("impl", impls), ("tier", tiers), ("lang", langs)):
        print(f"  {name}: {dict(sorted(Counter(arr).items()))}")
    print(f"  источники: {dict(Counter(c['seed_source'] for c in chosen))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
