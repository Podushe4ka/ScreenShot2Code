"""Режет briefs.jsonl на пачки под сабагентов Claude Code.

Пачка = один сабагент = один файл `batch_XXX.jsonl`. Размер пачки подобран так, чтобы
агент удержал в контексте и промпт генерации, и сами ТЗ, и написанные страницы: на тире L
одна страница это 18-40 КБ, поэтому пачки с тяжёлыми ТЗ делаются меньше.

Пачки формируются ОДНОРОДНЫМИ по impl: смешивать в одном агенте react_cdn и
static_inline — верный способ получить static-страницу с классами Tailwind.

Использование:
    .venv/bin/python Data/generators/synth/make_batches.py
    .venv/bin/python Data/generators/synth/make_batches.py --only p0000 p0001
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Страниц на пачку по тиру — обратно пропорционально их весу.
BATCH_SIZE = {"S": 12, "M": 8, "L": 4}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--briefs", default="Data/synth_pilot/briefs.jsonl")
    ap.add_argument("--out", default="Data/synth_pilot/batches")
    ap.add_argument("--only", nargs="*", help="собрать пачку только из этих id")
    args = ap.parse_args()

    briefs = [json.loads(l) for l in Path(args.briefs).open(encoding="utf-8")]
    if args.only:
        keep = set(args.only)
        briefs = [b for b in briefs if b["id"] in keep]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("batch_*.jsonl"):
        old.unlink()

    groups = defaultdict(list)
    for b in briefs:
        groups[(b["impl"], b["tier"])].append(b)

    n_batch = 0
    for (impl, tier), items in sorted(groups.items()):
        size = BATCH_SIZE[tier]
        for i in range(0, len(items), size):
            chunk = items[i:i + size]
            path = out_dir / f"batch_{n_batch:03d}_{impl}_{tier}.jsonl"
            with path.open("w", encoding="utf-8") as w:
                for b in chunk:
                    w.write(json.dumps(b, ensure_ascii=False) + "\n")
            print(f"  {path.name}: {len(chunk)} ТЗ")
            n_batch += 1

    print(f"\n{n_batch} пачек на {len(briefs)} ТЗ -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
