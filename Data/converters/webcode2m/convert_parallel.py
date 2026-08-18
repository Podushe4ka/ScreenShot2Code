#!/usr/bin/env python3
"""convert_parallel.py (WebCode2M) — параллельный (многопроцессный) конвертер
WebCode2M -> формат drafting-контракта.

Зеркалит `Data/converters/websight/convert_parallel.py`, но:
  • источник и поля берутся из `convert_lib` этой папки (WebCode2M, поле `text`);
  • `process_one` НЕ делает Tailwind-precompile (реальный CSS), зато санитайзит внешние ресурсы.

Логика — в `convert_lib.py`. Здесь только оркестрация:
  фаза 1: стрим + лёгкие фильтры -> список HTML;
  фаза 2: ProcessPoolExecutor(spawn) -> convert_lib.process_one;
  фаза 3: save_to_disk + приёмка + токен-отчёт.

    python convert_parallel.py --target 5000 --n-workers 32

Тестовый набор берётся из ХВОСТА корпуса, чтобы не пересечься с обучающим:
`--skip` выбрасывает первые N записей стрима до всякой фильтрации, а `--exclude-html`
дополнительно выкидывает страницы, чей HTML уже лежит в html-кэше обучающего набора.

    python convert_parallel.py --target 500 --skip 200000 --out ./webcode2m_test_500 \
        --exclude-html ./webcode2m_15k_htmls.jsonl.gz
"""
import argparse
import gzip
import hashlib
import json
import multiprocessing
import os
import shutil
import struct
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk

# Сколько сэмплов складывать в один кусок при записи. Все картинки куска лежат
# в ОДНОМ бинарном массиве Arrow, а у него 32-битные оффсеты — предел 2 ГБ на
# массив. 15k сэмплов это ~7.5 ГБ, и `Dataset.from_list` на всём наборе сразу
# падает с «offset overflow while concatenating arrays» (на 3k проходило
# впритык, ~1.5 ГБ). Кусок в 500 страниц — это сотни МБ, с запасом.
CHUNK_ROWS = 500

from convert_lib import (DATASET_ID, HTML_FIELD, IMAGE_FIELD, FEATURES, RENDER_WIDTH,
                         SPLIT, TOKENIZER_ID_DEFAULT, ahash, count_tokens, hamming,
                         process_one, qwen_image_tokens)

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def png_size(data):
    """(width, height) из заголовка PNG — без декодирования пикселей (IHDR за 8-байтной сигнатурой)."""
    assert data[:8] == _PNG_SIG, "не PNG — png_size рассчитан на скриншоты Playwright"
    return struct.unpack(">II", data[16:24])


# фаза 1
def _meta_path(path):
    return path + ".meta.json"


def load_html_cache(path, target, skip):
    """Кандидаты из кэша прошлого прогона, если их там хватает.

    Фаза 1 стримит шарды источника целиком (для 15k это ~4 ГБ и ~40 минут), а
    нужен из них один HTML — картинку мы всё равно рендерим сами. Падение любой
    следующей фазы без кэша означало бы повторную скачку с нуля.

    Кэш привязан к участку корпуса: рядом лежит `<кэш>.meta.json` со `skip`. Иначе
    тестовый прогон (`--skip 200000`) молча подобрал бы кэш обучающего набора
    (`skip=0`) и склеил бы train с test. У кэшей, сделанных до появления `--skip`,
    сайдкара нет — они с начала корпуса, т.е. skip=0.
    """
    if not path or not os.path.exists(path):
        return None
    meta = {}
    if os.path.exists(_meta_path(path)):
        with open(_meta_path(path), encoding="utf-8") as f:
            meta = json.load(f)
    cached_skip = meta.get("skip", 0)
    if cached_skip != skip:
        print(f"[фаза 1] кэш {path} собран со skip={cached_skip}, а нужен skip={skip} — "
              f"стримлю заново (кэш не перезаписываю: укажи свой --html-cache)")
        return None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        htmls = [json.loads(line)["html"] for line in f]
    if len(htmls) < target:
        print(f"[фаза 1] в кэше {len(htmls)} < target {target} — стримлю заново")
        return None
    print(f"[фаза 1] кандидаты из кэша {path}: беру {target} из {len(htmls)} (skip={skip})")
    return htmls[:target]


def save_html_cache(path, htmls, skip):
    if not path:
        return
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for h in htmls:
            f.write(json.dumps({"html": h}, ensure_ascii=False) + "\n")
    os.replace(tmp, path)   # атомарно: недописанный кэш не должен выглядеть готовым
    with open(_meta_path(path), "w", encoding="utf-8") as f:
        json.dump({"dataset": DATASET_ID, "split": SPLIT, "skip": skip, "n": len(htmls)}, f)
    print(f"[фаза 1] кандидаты сохранены в кэш: {path} (skip={skip})")


def load_exclude_hashes(paths):
    """sha1 исходных HTML, которые брать нельзя (страницы обучающего набора).

    Ждём html-кэши фазы 1 (`<out>_htmls.jsonl.gz`): там лежит СЫРОЙ HTML корпуса,
    поэтому хэши сравнимы с тем, что видит `collect_candidates`. `target_html`
    готового набора для этого не годится — он уже прошёл sanitize и не совпадёт.
    """
    excl = set()
    for p in paths or []:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            n = 0
            for line in f:
                html = (json.loads(line)["html"] or "").strip()
                excl.add(hashlib.sha1(html.encode("utf-8")).hexdigest())
                n += 1
        print(f"[фаза 1] исключаю {n} страниц из {p}")
    return excl


def collect_candidates(target, max_scan, near_dup, skip=0, exclude=frozenset()):
    """Собрать `target` HTML, пропустив первые `skip` записей стрима.

    `skip` считается по СЫРЫМ записям источника, до фильтров, а `max_scan` — уже
    после пропуска: так «взять 500 штук начиная с 200k» описывается парой чисел,
    не зависящей от того, сколько записей отсеется.
    """
    # Без near-dup картинка источника не нужна вовсе (рендерим свою), а на пропуске
    # 200k записей это разница между парой сотен МБ текста и десятками ГБ PNG.
    stream = need_image = None
    if near_dup is None:
        try:
            s = load_dataset(DATASET_ID, split=SPLIT, streaming=True, columns=[HTML_FIELD])
            # Проба одной записью: `columns=` поддержан только parquet-билдером, и отказ
            # может прийти как на загрузке, так и лениво на первой итерации. Одна запись
            # дешёвая, а полный прогон терять на этом нельзя.
            assert HTML_FIELD in next(iter(s)), f"нет колонки {HTML_FIELD}"
            stream, need_image = s, False
            print(f"[фаза 1] читаю только колонку {HTML_FIELD!r} (скриншоты источника не качаются)")
        except Exception as e:
            print(f"[фаза 1] колоночная выборка недоступна ({type(e).__name__}: {e}) — читаю всё")
    if stream is None:
        stream = load_dataset(DATASET_ID, split=SPLIT, streaming=True)
        need_image = True

    it = iter(stream)
    if skip:
        # Прогресс печатаем строками, а не прогресс-баром: пропуск занимает часы, а
        # `docker run -d` пишет лог построчно — бар на `\r` в нём попросту не виден.
        import time
        t0 = time.time()
        for i in range(skip):
            try:
                next(it)
            except StopIteration:
                raise SystemExit(f"в {DATASET_ID}/{SPLIT} меньше {skip} записей — уменьши --skip")
            if (i + 1) % 10000 == 0:
                el = time.time() - t0
                print(f"[фаза 1] пропуск {i + 1}/{skip} ({el / 60:.1f} мин, "
                      f"{(i + 1) / el:.0f} зап/с, осталось ~{(skip - i - 1) / ((i + 1) / el) / 60:.0f} мин)",
                      flush=True)
        print(f"[фаза 1] пропуск {skip} записей завершён за {(time.time() - t0) / 60:.1f} мин", flush=True)

    htmls, seen, hashes = [], set(), []
    scanned = skipped_empty = skipped_dup = skipped_nd = skipped_excl = 0
    for r in it:
        if len(htmls) >= target or scanned >= max_scan:
            break
        scanned += 1
        html = (r.get(HTML_FIELD) or "").strip()
        img = r.get(IMAGE_FIELD)
        if not html or (need_image and img is None):
            skipped_empty += 1
            continue
        if near_dup is not None:
            hsh = ahash(img)
            if any(hamming(hsh, o) <= near_dup for o in hashes):
                skipped_nd += 1
                continue
            hashes.append(hsh)
        h = hashlib.sha1(html.encode("utf-8")).hexdigest()
        if h in seen:
            skipped_dup += 1
            continue
        if h in exclude:
            skipped_excl += 1
            continue
        seen.add(h)
        htmls.append(html)
    print(f"[фаза 1] кандидатов: {len(htmls)} (пропущено с начала {skip}, просмотрено {scanned}; "
          f"пустых={skipped_empty}, дублей={skipped_dup}, near-dup={skipped_nd}, "
          f"из чужого набора={skipped_excl})")
    if len(htmls) < target:
        print(f"[фаза 1] ⚠ набралось {len(htmls)} < target {target}: упёрлись в --max-scan "
              f"({max_scan}) или в конец корпуса — поднимай --max-scan")
    return htmls


# токен-отчёт
def token_report(rows, sizes):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID_DEFAULT)
    pairs = [(count_tokens(r["target_html"], tok), qwen_image_tokens(w, h))
             for r, (w, h) in zip(rows, sizes)]
    code = sorted(c for c, _ in pairs)
    img = sorted(m for _, m in pairs)
    total = sorted(c + m for c, m in pairs)
    def q(a, x): return a[min(len(a) - 1, int(len(a) * x))]
    p99_code, p99_img = q(code, .99), q(img, .99)
    ml = ((p99_code + 63) // 64) * 64 + p99_img
    print(f"[токены] код:      median={code[len(code)//2]}, p99={p99_code}, max={code[-1]}")
    print(f"[токены] картинка: median={img[len(img)//2]}, p99={p99_img}, max={img[-1]}")
    print(f"[токены] всего:    median={total[len(total)//2]}, p99={q(total,.99)}, max={total[-1]}")
    print(f"[токены] рекомендуемый max_length (код p99 + картинка p99): {ml}")
    if p99_code > 8192:
        print("[токены] ⚠ WebCode2M p99 кода ВЫШЕ SFT-окна 8192 — нужен фильтр по бюджету "
              "(отсечь длинный хвост) перед SFT. Для pretrain ограничение мягче.")


# main
def main():
    ap = argparse.ArgumentParser(description="WebCode2M -> формат контракта (параллельно).")
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--max-scan", type=int, default=50000,
                    help="предел просмотренных записей ПОСЛЕ --skip")
    ap.add_argument("--skip", type=int, default=0,
                    help="пропустить первые N записей стрима (до фильтров). Так тестовый "
                         "набор берётся из хвоста корпуса и не пересекается с обучающим: "
                         "трейн собран из первых 50k -> для теста --skip 200000")
    ap.add_argument("--exclude-html", action="append", default=None, metavar="КЭШ.jsonl.gz",
                    help="html-кэш фазы 1 другого набора: его страницы не берём (сравнение по "
                         "sha1 сырого HTML). Можно повторять. Страховка от повторов страниц "
                         "в самом корпусе — сверх позиционного --skip")
    ap.add_argument("--n-workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="./webcode2m_pilot")
    ap.add_argument("--near-dup", type=int, default=None)
    ap.add_argument("--html-cache", default=None,
                    help="кэш кандидатов фазы 1 (jsonl.gz); по умолчанию <out>_htmls.jsonl.gz")
    ap.add_argument("--token-report", action="store_true",
                    help="пересчитать токены (тянет transformers/torch).")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    cache_path = os.path.abspath(args.html_cache or out_dir + "_htmls.jsonl.gz")
    print(f"источник: {DATASET_ID} | target: {args.target} | skip: {args.skip} | "
          f"воркеров: {args.n_workers} | out: {out_dir}")

    htmls = load_html_cache(cache_path, args.target, args.skip)
    if htmls is None:
        htmls = collect_candidates(args.target, args.max_scan, args.near_dup, args.skip,
                                   load_exclude_hashes(args.exclude_html))
        save_html_cache(cache_path, htmls, args.skip)
    if not htmls:
        raise SystemExit("фаза 1 не набрала ни одного кандидата — рендерить нечего")

    # preflight: тест рендера в главном процессе (реальная страница WebCode2M — без Tailwind)
    print("[preflight] тест sanitize + render...")
    _t = process_one("<html><head><style>.b{background:#2563eb;color:#fff;padding:20px}</style>"
                     "</head><body><div class='b'>test</div><img src='x.png'></body></html>")
    if _t[0] != "ok":
        print("[preflight] ПРОВАЛ:\n" + _t[2])
        raise SystemExit("render не работает — почини окружение (chromium/playwright).")
    print("[preflight] OK")

    # фаза 2: параллельно. spawn (не fork!) — Playwright ломается через fork после старта браузера.
    # Готовые сэмплы копятся не до конца прогона, а до CHUNK_ROWS, после чего кусок
    # уходит на диск: и в предел 2 ГБ на массив Arrow не упираемся, и падение на
    # последнем сэмпле не уносит с собой весь отрендеренный набор.
    parts_dir = out_dir + "_parts"
    shutil.rmtree(parts_dir, ignore_errors=True)
    os.makedirs(parts_dir, exist_ok=True)
    buf, parts, errs, sizes = [], [], [], []

    def flush():
        if not buf:
            return
        path = os.path.join(parts_dir, f"part-{len(parts):05d}")
        Dataset.from_list(buf, features=FEATURES).save_to_disk(path)
        parts.append(path)
        buf.clear()

    ctx = multiprocessing.get_context("spawn")
    from tqdm import tqdm
    with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=ctx) as ex:
        for res in tqdm(ex.map(process_one, htmls, chunksize=4), total=len(htmls), desc="[фаза 2] render"):
            if res[0] == "ok":
                _, html, png = res
                sizes.append(png_size(png))
                buf.append({"task_type": "drafting", "images": [{"bytes": png, "path": None}],
                            "current_html": "", "target_html": html, "instruction": ""})
                if len(buf) >= CHUNK_ROWS:
                    flush()
            else:
                errs.append((res[1], res[2]))
    flush()
    n_rows = sum(len(load_from_disk(p)) for p in parts) if parts else 0
    print(f"[фаза 2] готово сэмплов: {n_rows} | ошибок: {len(errs)} | кусков: {len(parts)}")
    for msg, cnt in Counter(m for m, _ in errs).most_common(3):
        print(f"  ✗ {cnt}×  {msg}")
    if not parts:
        raise SystemExit("Все воркеры упали. Первый traceback:\n" + errs[0][1])

    # фаза 3: куски memory-mapped, поэтому склейка не поднимает набор в память целиком.
    ds = concatenate_datasets([load_from_disk(p) for p in parts])
    ds.save_to_disk(out_dir, max_shard_size="500MB")
    del ds
    shutil.rmtree(parts_dir, ignore_errors=True)
    ds2 = load_from_disk(out_dir)
    assert len(ds2) == len(sizes), f"рассинхрон: ds2={len(ds2)} sizes={len(sizes)}"
    # только нужная колонка: иначе приёмка тянет с сетевого диска все ~8 ГБ картинок
    assert all(s["target_html"] and "<img" not in s["target_html"].lower()
               for s in ds2.select_columns(["target_html"]))
    widths = {w for w, _ in sizes}
    assert widths == {RENDER_WIDTH}, f"ширины разные: {widths}"
    heights = sorted(h for _, h in sizes)
    p95 = heights[min(len(heights) - 1, int(len(heights) * .95))]
    print(f"[приёмка] OK: {len(ds2)} сэмплов, ширина {RENDER_WIDTH}, нет <img>, load_from_disk ✓")
    print(f"[приёмка] высота: min={heights[0]}, median={heights[len(heights)//2]}, "
          f"p95={p95}, max={heights[-1]}")
    if args.token_report:
        token_report(list(ds2), sizes)
    else:
        print("[токены] отчёт пропущен (--token-report для пересчёта). WebCode2M (EDA):")
        print("[токены]   код p99≈9920 (ВЫШЕ SFT-окна 8192 — нужен фильтр по длине для SFT).")
    print(f"\n=== ПЕРЕДАЧА ===\nпуть: {out_dir}\nзагрузка: load_from_disk(<путь>)\n"
          f"⚠ WebCode2M — реальные страницы: проверь долю ошибок рендера выше и "
          f"фильтр по токенам (p99 кода ~9920 > 8192).")


if __name__ == "__main__":
    main()
