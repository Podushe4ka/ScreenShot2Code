"""Собирает финальный датасет под контракт SFT/DATA_FORMAT_CONTRACT.md.

Вход — три источника, все построенные из одних и тех же чистых страниц:
  build/manifest.jsonl  -> drafting  (скриншот страницы -> её исходник)
  editing.jsonl         -> editing   (рендер текущей + инструкция -> исправленный код)
  polishing.jsonl       -> polishing (эталонный скриншот + рендер испорченной -> эталон)

Схема — ровно пять полей `FEATURES` из §2 контракта плюс служебные колонки
(`page_id`, `impl`, `brief_id`, `op`). Тренировочный код служебные не читает: `to_message`
в SFT/train/formatting.py обращается только к task_type / images / current_html /
target_html / instruction, поэтому лишние колонки безопасны, а разрез и отладка без них
невозможны.

Порядок картинок для polishing задан контрактом и промптом: ПЕРВОЙ идёт эталон
(«the FIRST is the TARGET design»), ВТОРЫМ — текущий рендер. Перепутать их значит
обучать модель ровно наоборот.

Использование:
    .venv/bin/python Data/converters/synth/pack.py --out Data/synth_pilot/dataset
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import Dataset, Features, Image, Sequence, Value  # noqa: E402

from make_split_grouped import assert_no_leak, grouped_split  # noqa: E402

# Схема — дословно из контракта §2, плюс служебные колонки в конце.
FEATURES = Features({
    "task_type":    Value("string"),
    "images":       Sequence(Image()),
    "current_html": Value("string"),
    "target_html":  Value("string"),
    "instruction":  Value("string"),
    # служебное
    "page_id":      Value("string"),
    "impl":         Value("string"),
    "op":           Value("string"),
})


def _img(path: str) -> dict:
    """Картинка как встроенные байты, а не путь (§1 контракта)."""
    return {"bytes": Path(path).read_bytes(), "path": None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="Data/synth_pilot/build/manifest.jsonl")
    ap.add_argument("--editing", default="Data/synth_pilot/editing.jsonl")
    ap.add_argument("--polishing", default="Data/synth_pilot/polishing.jsonl")
    ap.add_argument("--raw", default="Data/synth_pilot/raw")
    ap.add_argument("--out", default="Data/synth_pilot/dataset")
    args = ap.parse_args()

    raw_dir = Path(args.raw)
    rows = []

    # --- drafting ---
    pages = {}
    for line in Path(args.manifest).open(encoding="utf-8"):
        r = json.loads(line)
        if r["status"] != "ok":
            continue
        pages[r["id"]] = r
        rows.append({
            "task_type": "drafting",
            "images": [_img(r["png"])],
            "current_html": "",
            "target_html": (raw_dir / f"{r['id']}.html").read_text(encoding="utf-8"),
            "instruction": "",
            "page_id": r["id"], "impl": r["impl"], "op": "",
        })

    # --- editing ---
    for path, task in ((args.editing, "editing"), (args.polishing, "polishing")):
        p = Path(path)
        if not p.exists():
            print(f"  ! {path} нет — пропускаю задачу {task}", file=sys.stderr)
            continue
        for line in p.open(encoding="utf-8"):
            s = json.loads(line)
            if s["page_id"] not in pages:
                continue          # страница отбракована — её производные тоже
            if task == "editing":
                images = [_img(s["current_png"])]
            else:
                # эталон первым, текущий рендер вторым — см. докстринг
                images = [_img(pages[s["page_id"]]["png"]), _img(s["current_png"])]
            rows.append({
                "task_type": task,
                "images": images,
                "current_html": s["current_html"],
                "target_html": s["target_html"],
                "instruction": s.get("instruction", ""),
                "page_id": s["page_id"], "impl": s["impl"],
                "op": s["meta"].get("op") or s["meta"].get("degradation", ""),
            })

    if not rows:
        print("нечего паковать", file=sys.stderr)
        return 1

    ds = Dataset.from_list(rows, features=FEATURES)
    dd = grouped_split(ds)
    assert_no_leak(dd)

    out = Path(args.out)
    dd.save_to_disk(str(out))

    print(f"собрано {len(ds)} сэмплов из {len(pages)} страниц -> {out}")
    print(f"  train={len(dd['train'])}  validation={len(dd['validation'])}")
    for name in ("train", "validation"):
        print(f"  {name}: задачи {dict(sorted(Counter(dd[name]['task_type']).items()))}, "
              f"стилей {dict(sorted(Counter(dd[name]['impl']).items()))}, "
              f"страниц {len(set(dd[name]['page_id']))}")
    print("утечки страниц между сплитами нет")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
