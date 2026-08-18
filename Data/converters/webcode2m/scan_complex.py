#!/usr/bin/env python3
"""scan_complex.py (WebCode2M) — просеять корпус и вытащить кандидатов в «сложные страницы».

ЗАЧЕМ ОТДЕЛЬНЫЙ СКАН. Сложность считается по ОТРЕНДЕРЕННОЙ странице
(`../complexity/features.py`), а рендер стоит ~0.3–1 с. Кандидатов в WebCode2M порядка
185 тысяч — отрендерить их все нельзя ни на каком доступном железе. Поэтому здесь дешёвая
статика без браузера: она отбирает шортлист, и уже он идёт в конвертер и в рендер-скоринг.

ПОРОГ 300, А НЕ 400. План задавал ориентир «>=400 узлов у 7.9% корпуса», но перемер на
самом корпусе (2 777 страниц, по row-group из 30 шардов) даёт другое: p50 177 / p90 281 /
p99 414, и `>=400` набирает лишь **1.22%**. Заявленный планом выход ~8% даёт порог
**300 узлов** (7.20%) — он и стоит по умолчанию. То есть воспроизводится ВЫХОД из плана,
а не абсолютный порог; замер и таблица — в `../complexity/README.md` («Калибровка порогов»).
Отсюда и оценка кандидатов: 2.56M строк x 7.20% ~ 185k при пороге 300 и всего ~31k при 400.

ЧИТАЕМ ТОЛЬКО КОЛОНКУ `text`. В parquet 93% веса — колонка `image` (44.8 МБ против 0.55 МБ
на текст в замеренной row-group). Картинку мы всё равно рендерим свою, поэтому проекция
колонок превращает 51 ГБ корпуса в ~600 МБ трафика на 190 тысяч строк.

Локальный кэш (`~/.cache/huggingface/.../webcode2m_purified`) содержит только 30 шардов из
100 — 37 168 строк. Этого мало (7.20% от 37k = ~2.7k страниц), поэтому по умолчанию
досканиваем с хаба: `--source auto` берёт сначала локальные шарды, потом хабовые.

    python scan_complex.py --out candidates.jsonl.gz --scan 190000
"""
import argparse
import glob
import gzip
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

# Пролог доступа к общему ядру и соседним пакетам конвертеров (см. common/__init__.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
_CONV = os.path.dirname(_HERE)
for _p in (_CONV, os.path.join(_CONV, "complexity")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REPO = "datasets/xcodemind/webcode2m_purified"
LOCAL_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--xcodemind--webcode2m_purified/snapshots/*/data/*.parquet")
COLUMNS = ["text", "lang", "hash"]

# Дешёвый предэкран перед разбором. BeautifulSoup на странице WebCode2M — это ~10 мс, на
# 190 тысячах строк полчаса чистого CPU; счёт открывающих тегов регуляркой — микросекунды.
# Условие одностороннее: тегов в разметке не может быть МЕНЬШЕ, чем элементов в дереве,
# поэтому страница, не набравшая порога по тегам, порога по узлам не наберёт тем более.
_TAG_RE = re.compile(r"<[a-zA-Z][\w:-]*")


def _screen_and_score(args):
    """Предэкран -> разбор -> статические признаки. Возвращает dict или None."""
    html, lang, h, min_nodes = args
    if len(_TAG_RE.findall(html)) < min_nodes:
        return None
    import features
    try:
        f = features.static_features(html)
    except Exception:
        return None
    if f["nodes"] < min_nodes:
        return None
    f.update(id=h, lang=lang, html=html)
    return f


def iter_rows(paths, columns, use_fs):
    import pyarrow.parquet as pq
    fs = None
    if use_fs:
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
    for path in paths:
        try:
            fh = fs.open(path, "rb") if fs else open(path, "rb")
            with fh:
                pf = pq.ParquetFile(fh)
                for gi in range(pf.metadata.num_row_groups):
                    tbl = pf.read_row_group(gi, columns=columns)
                    for row in tbl.to_pylist():
                        yield row
        except Exception as e:
            print(f"[скан] шард пропущен ({os.path.basename(path)}): {type(e).__name__}: {e}",
                  flush=True)


def hub_shards(limit=100):
    import requests
    r = requests.get(
        f"https://huggingface.co/api/datasets/xcodemind/webcode2m_purified/tree/main/data",
        params={"limit": limit}, timeout=60)
    r.raise_for_status()
    return [f"{REPO}/{f['path']}" for f in r.json() if f["path"].endswith(".parquet")]


def _flush(ex, batch, out, kept, args, scanned, t0):
    """Посчитать пачку в пуле и дописать кандидатов. -> (kept, хватит_ли)."""
    for feats in ex.map(_screen_and_score, batch, chunksize=32):
        if feats is None:
            continue
        kept += 1
        out.write(json.dumps(feats, ensure_ascii=False) + "\n")
        if args.max_candidates and kept >= args.max_candidates:
            out.flush()
            return kept, True
    out.flush()
    rate = scanned / max(1e-9, time.time() - t0)
    print(f"[скан] просмотрено {scanned}, кандидатов {kept} "
          f"({100*kept/max(1,scanned):.2f}%), {rate:.0f} строк/с", flush=True)
    return kept, False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="куда писать кандидатов (.jsonl.gz)")
    ap.add_argument("--scan", type=int, default=190000, help="сколько строк просмотреть")
    ap.add_argument("--min-nodes", type=int, default=300,
                    help="порог сложности по узлам. 300 = перемеренный порог с выходом 7.20%%; "
                         "плановые >=400 дают на этом корпусе лишь 1.22%% (см. complexity/README.md)")
    ap.add_argument("--max-candidates", type=int, default=0, help="хватит кандидатов (0 = без предела)")
    ap.add_argument("--source", choices=["auto", "local", "hub"], default="auto")
    ap.add_argument("-j", "--n-workers", type=int, default=6)
    args = ap.parse_args()

    local = sorted(glob.glob(LOCAL_GLOB))
    plans = []
    if args.source in ("auto", "local") and local:
        plans.append((local, False))
    if args.source in ("auto", "hub"):
        local_names = {os.path.basename(p) for p in local}
        remote = [p for p in hub_shards() if os.path.basename(p) not in local_names]
        if remote:
            plans.append((remote, True))
    if not plans:
        raise SystemExit("нет ни локальных шардов, ни доступа к хабу")
    print(f"[скан] шардов локально {len(local)}, планов чтения {len(plans)}; "
          f"цель {args.scan} строк, порог {args.min_nodes} узлов")

    t0 = time.time()
    scanned = kept = 0
    tmp = args.out + ".tmp"

    def source_rows():
        nonlocal scanned
        for paths, use_fs in plans:
            for row in iter_rows(paths, COLUMNS, use_fs):
                if scanned >= args.scan:
                    return
                scanned += 1
                yield ((row.get("text") or ""), row.get("lang"),
                       row.get("hash") or f"wc2m{scanned:09d}", args.min_nodes)

    # ⚠ Работаем ПАЧКАМИ, а не одним `ex.map` по генератору. `Executor.map` вычерпывает
    # входной итератор ЦЕЛИКОМ до первого результата — на 250 тысяч строк это значит
    # «скачай и удержи в памяти весь корпус» (~3.75 ГБ HTML при 8 ГБ на машине), и ни одна
    # строка не долетает до диска, пока скан не кончится. Проверено: за несколько минут
    # прогона выходной файл остался нулевым. Пачками память постоянна, а результат пишется
    # по ходу — прогон переживает остановку.
    BATCH = 2000
    with gzip.open(tmp, "wt", encoding="utf-8") as out, \
            ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        batch, done = [], False
        for item in source_rows():
            batch.append(item)
            if len(batch) < BATCH:
                continue
            kept, done = _flush(ex, batch, out, kept, args, scanned, t0)
            batch = []
            if done:
                break
        if batch and not done:
            kept, done = _flush(ex, batch, out, kept, args, scanned, t0)
    os.replace(tmp, args.out)
    print(f"[скан] итог: просмотрено {scanned}, кандидатов {kept} "
          f"({100*kept/max(1,scanned):.2f}%) -> {args.out}  ({(time.time()-t0)/60:.1f} мин)")


if __name__ == "__main__":
    main()
