#!/usr/bin/env python3
"""select_by_complexity.py — отбор страниц по перцентилям сложности.

⚠ Имя файла НЕ `select.py`: модуль лежит в каталоге, который score.py кладёт в sys.path,
и `select.py` перекрыл бы стандартный `select` — на macOS это роняет любой импорт socket
(а значит и ProcessPoolExecutor) в дочерних процессах.

РЕШЕНИЕ ПО УМОЛЧАНИЮ — НЕ «ТОЛЬКО СЛОЖНЫЕ». Снизу отрезаются тривиальные страницы (на
сниппете в десять узлов учиться нечему), сверху — монстры (за p99 начинается раздутость,
а не сложность), а внутри масса смещается в тяжёлый конец с сохранением градиента:

    --min-pct 40  --max-pct 99
    --mix "0.15:40-60, 0.35:60-85, 0.50:85-99"

то есть p0–p40 выбрасываем целиком; половина набора берётся из p85–p99, треть из p60–p85,
остаток из p40–p60. Обоснование: набор только из сложного учится хуже смешанного и
расходится с распределением Design2Code, на котором нас меряют; отсечка снизу убирает
сниппеты. Ветка A/B «весь набор из p85+» доступна как `--hard-only`.

ПЛОТНОСТЬ (`--density-min`) — отдельный барьер, и он тут главный по смыслу. «Длинный
таргет» и «сложная страница» коррелируют, поэтому наивный отбор по верхним перцентилям
натаскает раздутых страниц — ровно тех, на которых модель выучилась писать `<head>` на
25 тысяч символов. Порог «видимых блоков на токен кода» отсекает их до перцентилей.

    python select_by_complexity.py features.jsonl --out selected.jsonl -n 8000 \
        --min-pct 40 --max-pct 99 --mix "0.15:40-60, 0.35:60-85, 0.50:85-99"
"""
import argparse
import json
import random
import re
from collections import Counter

_MIX_RE = re.compile(r"^\s*([\d.]+)\s*:\s*(\d+)\s*-\s*(\d+)\s*$")


def parse_mix(text):
    """`"0.15:40-60, 0.35:60-85, 0.50:85-99"` -> [(доля, lo_pct, hi_pct), …]."""
    out = []
    for part in text.split(","):
        m = _MIX_RE.match(part)
        if not m:
            raise SystemExit(f"не разобрал долю смеси: {part!r} (ожидается «0.35:60-85»)")
        frac, lo, hi = float(m.group(1)), int(m.group(2)), int(m.group(3))
        if lo >= hi:
            raise SystemExit(f"пустой диапазон в {part!r}")
        out.append((frac, lo, hi))
    total = sum(f for f, _, _ in out)
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"доли смеси в сумме дают {total}, а не 1.0")
    return out


def pct_bounds(values, lo_pct, hi_pct):
    """Границы значений сложности по перцентилям набора."""
    vals = sorted(values)
    n = len(vals)
    lo = vals[min(n - 1, int(n * lo_pct / 100.0))]
    hi = vals[min(n - 1, int(n * hi_pct / 100.0))]
    return lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("features", nargs="+", help="features.jsonl от score.py (можно несколько)")
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", "--count", type=int, required=True, help="сколько страниц отобрать")
    ap.add_argument("--min-pct", type=float, default=40, help="ниже этого перцентиля не берём")
    ap.add_argument("--max-pct", type=float, default=99, help="выше этого перцентиля не берём")
    ap.add_argument("--mix", default="0.15:40-60, 0.35:60-85, 0.50:85-99",
                    help="раскладка набора по перцентильным полосам")
    ap.add_argument("--hard-only", action="store_true",
                    help="ветка A/B: весь набор из p85+ (перебивает --mix)")
    ap.add_argument("--density-min", type=float, default=None,
                    help="минимум видимых блоков на токен кода (порог раздутости)")
    ap.add_argument("--min-css-decls", type=int, default=None,
                    help="минимум CSS-объявлений у страницы с >=--unstyled-min-nodes узлов; "
                         "ниже -> страница фактически без стилей (стайлшита нет в источнике)")
    ap.add_argument("--unstyled-min-nodes", type=int, default=20,
                    help="с какого числа узлов страница обязана быть оформленной")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="потолок токенов кода (бюджет окна SFT)")
    ap.add_argument("--max-total-tokens", type=int, default=None,
                    help="потолок на код+картинку (окно SFT минус накладные шаблона)")
    ap.add_argument("--max-page-width", type=int, default=1280,
                    help="шире вьюпорта = горизонтальное переполнение, кадр нечитаем")
    ap.add_argument("--max-page-height", type=int, default=4096,
                    help="выше этого страница при сжатии в бюджет пикселей теряет читаемость")
    ap.add_argument("--key", default="complexity", help="по какому полю ранжировать")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    feats = []
    for path in args.features:
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if not rec.get("error") and rec.get(args.key) is not None:
                    feats.append(rec)
    print(f"[отбор] на входе {len(feats)} страниц с признаками")

    # Жёсткие барьеры — ДО перцентилей: перцентили должны считаться по тому набору,
    # из которого реально берём, иначе раздутые страницы сдвинут все границы.
    dropped = Counter()
    if args.density_min is not None:
        before = len(feats)
        feats = [f for f in feats if (f.get("density") or 0) >= args.density_min]
        dropped["density"] = before - len(feats)
    # Страница «без стилей»: узлов много, а оформления нет. У WebUI это не редкость —
    # у части источников колонка `css` не содержит стайлшита компонентов, и конвертер
    # честно отдаёт неоформленную страницу (сам он верен источнику: сырой рендер такой же).
    # Пара при этом СОГЛАСОВАНА — скриншот снят с того же кода, — но учить на ней
    # воспроизведению дизайна нечему, а сайт она представляет заведомо неверно.
    # Замер на отобранном WebUI: 21.1% страниц с >=20 узлов имели <10 объявлений
    # (у WebCode2M — 0.8%). Барьер здесь, а не в конвертере: переконвертация не нужна,
    # решение «годится ли для обучения» принимается на отборе.
    if args.min_css_decls is not None:
        before = len(feats)
        feats = [f for f in feats
                 if (f.get("nodes") or 0) < args.unstyled_min_nodes
                 or (f.get("css_decls") or 0) >= args.min_css_decls]
        dropped["unstyled"] = before - len(feats)
    if args.max_tokens is not None:
        before = len(feats)
        feats = [f for f in feats if (f.get("tokens_code") or 0) <= args.max_tokens]
        dropped["tokens"] = before - len(feats)
    # Бюджет окна честнее считать по СУММЕ «код + картинка», а не плоским потолком на код.
    # Визуальная часть зависит от реального размера скриншота: страница на один экран
    # (1280×720) стоит ~900 токенов, а полноэкранная высокая упирается в потолок ~2048.
    # Плоские 6000 на код одинаково режут и ту и другую, хотя у первой в окне остаётся
    # заметно больше места.
    if args.max_total_tokens is not None:
        before = len(feats)
        feats = [f for f in feats
                 if (f.get("tokens_total") or f.get("tokens_code") or 0) <= args.max_total_tokens]
        dropped["tokens_total"] = before - len(feats)
    # Геометрия скриншота. `render.py` снимает full_page, то есть ВСЮ прокручиваемую
    # область — включая горизонтальное переполнение. Замерено на конвертации: страница
    # 3777×9647 при вьюпорте 1280. Такой кадр процессор Qwen ужмёт в бюджет пикселей
    # целиком, и текст на нём станет нечитаемым (та же беда, что закрывали Tier A:
    # при 1.31 Мп высокие страницы давали ~10px шрифт). Пара формально корректна —
    # непригодна именно для обучения, поэтому барьер здесь, а не в конвертере.
    if args.max_page_width is not None:
        before = len(feats)
        feats = [f for f in feats
                 if (f.get("w") or f.get("page_w") or 0) <= args.max_page_width]
        dropped["width"] = before - len(feats)
    if args.max_page_height is not None:
        before = len(feats)
        feats = [f for f in feats
                 if (f.get("h") or f.get("page_h") or 0) <= args.max_page_height]
        dropped["height"] = before - len(feats)
    if dropped:
        print(f"[отбор] снято барьерами: {dict(dropped)} -> осталось {len(feats)}")
    if not feats:
        raise SystemExit("после барьеров не осталось страниц — ослабь --density-min/--max-tokens")

    vals = [float(f[args.key]) for f in feats]
    bands = [(1.0, 85, int(args.max_pct))] if args.hard_only else parse_mix(args.mix)
    if not args.hard_only:
        lo_all, hi_all = args.min_pct, args.max_pct
        bands = [(fr, max(lo, lo_all), min(hi, hi_all)) for fr, lo, hi in bands]
        bands = [b for b in bands if b[1] < b[2]]

    rng = random.Random(args.seed)
    picked, picked_ids, report = [], set(), []
    for frac, lo_pct, hi_pct in bands:
        lo, hi = pct_bounds(vals, lo_pct, hi_pct)
        pool = [f for f in feats if lo <= float(f[args.key]) <= hi and f["id"] not in picked_ids]
        want = int(round(args.count * frac))
        rng.shuffle(pool)
        take = pool[:want]
        picked.extend(take)
        picked_ids.update(f["id"] for f in take)
        report.append((f"p{lo_pct}-p{hi_pct}", frac, want, len(pool), len(take)))

    # Недобор в одной полосе (её пул мельче запрошенной доли) добираем из соседних
    # ВНУТРИ [min_pct, max_pct] — иначе набор молча выйдет меньше заказанного.
    if len(picked) < args.count:
        lo, hi = pct_bounds(vals, args.min_pct, args.max_pct)
        rest = [f for f in feats
                if lo <= float(f[args.key]) <= hi and f["id"] not in picked_ids]
        rng.shuffle(rest)
        add = rest[:args.count - len(picked)]
        picked.extend(add)
        picked_ids.update(f["id"] for f in add)
        if add:
            print(f"[отбор] добор из всего коридора p{args.min_pct}-p{args.max_pct}: {len(add)}")

    with open(args.out, "w", encoding="utf-8") as f:
        for rec in picked:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n=== ОТБОР: {len(picked)} из {len(feats)} (заказано {args.count}) ===")
    print(f"  {'полоса':12s} {'доля':>6s} {'нужно':>7s} {'в пуле':>7s} {'взято':>7s}")
    for band, frac, want, pool_n, took in report:
        print(f"  {band:12s} {frac:6.2f} {want:7d} {pool_n:7d} {took:7d}")
    if picked:
        summarize(picked, args.key)


def summarize(picked, key):
    def q(k, p):
        vals = sorted(float(f.get(k) or 0) for f in picked)
        return vals[min(len(vals) - 1, int(len(vals) * p))]

    print("\n  что отобрано (p50 / p90 / p99):")
    for k in ("nodes", "depth", "css_decls", "columns", "tokens_code"):
        print(f"    {k:14s} {q(k,.5):8.0f} {q(k,.9):8.0f} {q(k,.99):8.0f}")
    dens = [f.get("density") for f in picked if f.get("density")]
    if dens:
        dens.sort()
        print(f"    {'density':14s} {dens[len(dens)//2]:8.4f}")
    src = Counter(f.get("source") or f.get("source_name") for f in picked)
    print(f"  по источникам: {src.most_common(8)}")
    n400 = sum(1 for f in picked if (f.get("nodes") or 0) >= 400)
    print(f"  страниц >=400 узлов: {n400} ({100*n400/len(picked):.1f}%)")


if __name__ == "__main__":
    main()
