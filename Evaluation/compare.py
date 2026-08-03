#!/usr/bin/env python3
"""compare.py — визуальное сравнение генераций разных моделей на одном датасете.

Две подкоманды:
  gen    — одной моделью сгенерировать HTML по первым N сэмплам и отрендерить в PNG.
  stitch — склеить для каждого сэмпла ленту колонок: orig | base | ckpt94 | ... .

Итог stitch: <root>/compare_out/sample_XXXX.png — 5 картинок в ряд
(оригинал, генерация базовой модели, генерации чекпоинтов дообучения).

------------------------------------------------------------------------------
Почему падал прежний вариант
------------------------------------------------------------------------------
RuntimeError: "An attempt has been made to start a new process before the
current process has finished its bootstrapping phase" — это классика
multiprocessing со start-методом "spawn". В нашем стеке рендер раньше
распараллеливался через ProcessPoolExecutor + spawn (см. run_benchmark_batched.py,
там VLLM_WORKER_MULTIPROC_METHOD=spawn и явный mp.get_context("spawn")). При spawn
дочерний воркер ЗАНОВО ИМПОРТИРУЕТ главный модуль (сам скрипт). Если создание
пула / тяжёлая работа лежит на верхнем уровне модуля, а не под
`if __name__ == "__main__"`, то при этом повторном импорте воркер снова пытается
поднять пул — ещё до конца бутстрапа — и multiprocessing валит ровно этой ошибкой.

Здесь это исключено по двум причинам:
1. Вся работа — внутри функций, единственная точка входа под
   `if __name__ == "__main__"` (см. низ файла).
2. N маленькое (единицы сэмплов), поэтому рендер идёт ПОСЛЕДОВАТЕЛЬНО в одном
   процессе с переиспользуемым Chromium из render.py. Пул процессов не
   создаётся вовсе — спавнить нечего, падать нечему.

------------------------------------------------------------------------------
Запуск (внутри docker-образа design2code-bench, как в исходном скрипте)
------------------------------------------------------------------------------
  # генерация каждой моделью в свою папку:
  python3 /storage/compare.py gen --model <hf_id_или_путь> \
      --dataset /storage/data/webcode2m_le3072 --n 8 --out /storage/compare/<label>

  # склейка (orig берётся из самого датасета, остальные — из папок gen):
  python3 /storage/compare.py stitch --dataset /storage/data/webcode2m_le3072 \
      --n 8 --root /storage/compare --labels orig,base,ckpt94,ckpt188,ckpt282
"""

import argparse
import os
import re
import sys
from pathlib import Path

# --- render.py в образе лежит в /app (COPY в Dockerfile), а мы запускаемся как
#     `python3 /storage/compare.py`, т.е. sys.path[0] == /storage. Добавляем и
#     /app, и папку самого скрипта, чтобы `import render` находился в обоих
#     сценариях (в контейнере и локально рядом с render.py). --------------------
for _p in ("/app", str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- промпт дообучения. Чекпоинты учились ровно на нём (SFT/train/formatting.py:
#     DRAFTING_PROMPT); базовой модели он тоже подходит — так сравнение честное. -
DRAFTING_PROMPT = (
    "You are an expert front-end developer. Look at this webpage screenshot and "
    "write a SINGLE self-contained HTML file (inline <style>, no external CSS/JS/fonts, "
    "no network requests) that reproduces the layout, text, and colors as closely as "
    "possible. Use plain gray placeholder boxes instead of any real images. "
    "Output ONLY the raw HTML code, with no explanation and no markdown code fences."
)


def extract_html(text: str) -> str:
    """Достаёт HTML из ответа модели: срезает <think>, markdown-заборы и всё до
    <!doctype>/<html>. Дообученные модели отдают сырой HTML, база иногда — в
    ```html ... ```; покрываем оба случая (идентично run_benchmark.py)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    m = re.search(r"```(?:html)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1)
    low = text.lower()
    start = low.find("<!doctype")
    if start == -1:
        start = low.find("<html")
    if start != -1:
        text = text[start:]
    return text.strip()


def load_samples(dataset_path: str, n: int):
    """Первые N сэмплов (детерминированно, БЕЗ shuffle — чтобы `orig` и все
    модели ссылались на одни и те же сэмплы при раздельных запусках gen)."""
    from datasets import load_from_disk

    ds = load_from_disk(dataset_path)
    # save_to_disk мог сохранить как один Dataset, так и DatasetDict со сплитами.
    if hasattr(ds, "keys") and not hasattr(ds, "num_rows"):
        split = "train" if "train" in ds else list(ds.keys())[0]
        print(f"[compare] DatasetDict, беру сплит '{split}'")
        ds = ds[split]
    n = min(n, len(ds))
    return ds.select(range(n)), n


def sample_dirname(i: int) -> str:
    return f"sample_{i:04d}"


# ============================ подкоманда gen ==================================

def cmd_gen(args):
    # vLLM при tensor_parallel_size>1 форкает воркеров; spawn безопаснее fork с
    # уже инициализированной CUDA (см. комментарий в run_benchmark.py). Ставим
    # ДО импорта vllm. При TP=1 воркеры не форкаются вовсе.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from render import close_browser, prepare_and_render

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds, n = load_samples(args.dataset, args.n)
    print(f"[compare/gen] модель={args.model} | сэмплов={n} | out={out}")

    images = [ds[i]["images"][0].convert("RGB") for i in range(n)]

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
        # На единицах сэмплов захват CUDA-графов (десятки сек–минута на модель)
        # не окупается — eager-режим убирает этот старт-оверхед. Для больших
        # прогонов из run_benchmark его наоборот стоит оставить включённым.
        enforce_eager=args.enforce_eager,
    )

    # temperature=0 → детерминированный жадный декодинг: для визуального
    # сравнения моделей воспроизводимость важнее разнообразия.
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens)
    conversations = [
        [{"role": "user", "content": [
            {"type": "image_pil", "image_pil": img},
            {"type": "text", "text": DRAFTING_PROMPT},
        ]}]
        for img in images
    ]

    print(f"[compare/gen] генерирую {n} сэмплов одним батчем...")
    outputs = llm.chat(
        conversations,
        sampling_params=sampling_params,
        chat_template_kwargs={"enable_thinking": args.enable_thinking},
    )
    pred_html = [extract_html(o.outputs[0].text) for o in outputs]

    print("[compare/gen] рендерю предсказания...")
    for i in range(n):
        sdir = out / sample_dirname(i)
        sdir.mkdir(exist_ok=True)
        try:
            info = prepare_and_render(pred_html[i], str(sdir / "pred.html"), str(sdir / "pred.png"))
            print(f"  sample {i}: img_replaced={info['n_images_replaced']} render_ok={info['render_ok']}")
        except Exception as e:  # рендер не должен ронять весь прогон
            print(f"  sample {i}: ОШИБКА рендера: {e}")

    close_browser()
    print(f"[compare/gen] готово -> {out}")


# ============================ подкоманда stitch ===============================

def _load_font(size: int):
    from PIL import ImageFont
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _column_image(label: str, i: int, args, ds, col_w: int, col_h: int):
    """Одна колонка: картинка нужной модели/оригинала, вписанная в col_w x col_h."""
    from PIL import Image

    if label == "orig":
        # оригинал — это ровно тот скриншот, что модель видела на входе
        # (в датасете он уже с серыми плейсхолдерами — см. DATA_FORMAT_CONTRACT).
        img = ds[i]["images"][0].convert("RGB")
    else:
        png = Path(args.root) / label / sample_dirname(i) / "pred.png"
        if png.exists():
            img = Image.open(png).convert("RGB")
        else:
            img = Image.new("RGB", (col_w, col_h), color=(245, 245, 245))
            from PIL import ImageDraw
            ImageDraw.Draw(img).text((10, 10), "MISSING", fill=(200, 0, 0), font=_load_font(20))

    # вписываем по ширине, высоту режем/паддим до col_h (единый вьюпорт колонок)
    scale = col_w / img.width
    new_h = max(1, int(round(img.height * scale)))
    img = img.resize((col_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (col_w, col_h), color=(255, 255, 255))
    canvas.paste(img, (0, 0))  # выравнивание по верху
    return canvas


def cmd_stitch(args):
    from PIL import Image, ImageDraw

    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    ds, n = load_samples(args.dataset, args.n)

    out_dir = Path(args.root) / "compare_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    col_w = args.col_width
    col_h = args.col_height
    header_h = 40
    gap = 8
    font = _load_font(22)

    print(f"[compare/stitch] сэмплов={n} | колонки={labels} | out={out_dir}")
    for i in range(n):
        cols = [_column_image(lab, i, args, ds, col_w, col_h) for lab in labels]

        total_w = len(cols) * col_w + (len(cols) - 1) * gap
        strip = Image.new("RGB", (total_w, header_h + col_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(strip)

        x = 0
        for lab, col in zip(labels, cols):
            # шапка-подпись колонки
            draw.rectangle([x, 0, x + col_w, header_h], fill=(30, 30, 30))
            bbox = draw.textbbox((0, 0), lab, font=font)
            tw = bbox[2] - bbox[0]
            draw.text((x + (col_w - tw) // 2, 8), lab, fill=(255, 255, 255), font=font)
            strip.paste(col, (x, header_h))
            x += col_w + gap

        out_path = out_dir / f"{sample_dirname(i)}.png"
        strip.save(out_path)
        print(f"  {out_path}")

    print(f"[compare/stitch] готово -> {out_dir}")


# ================================ CLI ========================================

def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="сгенерировать HTML одной моделью и отрендерить в PNG")
    g.add_argument("--model", required=True, help="HF id или локальный путь к весам")
    g.add_argument("--dataset", required=True, help="путь к датасету (load_from_disk)")
    g.add_argument("--n", type=int, default=8)
    g.add_argument("--out", required=True, help="папка вывода для этой модели (label)")
    g.add_argument("--max-new-tokens", type=int, default=4096,
                   help="датасет ≤3072 токенов кода → 4096 с запасом хватает")
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--enable-thinking", action="store_true")
    g.add_argument("--tensor-parallel-size", type=int, default=1)
    g.add_argument("--gpu-memory-utilization", type=float, default=0.89)
    g.add_argument("--max-model-len", type=int, default=8192)
    # Пиксельный бюджет как в обучении (SFT/train/formatting.py) — иначе orig/base/ckpt
    # сравниваются на разном разрешении входа.
    g.add_argument("--min-pixels", type=int, default=262_144, help="min_pixels (256*32*32)")
    g.add_argument("--max-pixels", type=int, default=2_097_152, help="max_pixels (Tier A, 2048*32*32)")
    g.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false",
                   help="включить захват CUDA-графов (медленнее старт, окупается только на больших N)")
    g.set_defaults(enforce_eager=True)
    g.set_defaults(func=cmd_gen)

    s = sub.add_parser("stitch", help="склеить orig|base|ckpt.. в один PNG на сэмпл")
    s.add_argument("--dataset", required=True, help="тот же датасет, что в gen (для orig)")
    s.add_argument("--n", type=int, default=8)
    s.add_argument("--root", required=True, help="корень с папками моделей и compare_out/")
    s.add_argument("--labels", default="orig,base,ckpt94,ckpt188,ckpt282",
                   help="порядок колонок; 'orig' берётся из датасета, остальные из <root>/<label>/")
    s.add_argument("--col-width", type=int, default=380, help="ширина колонки, px")
    s.add_argument("--col-height", type=int, default=1400, help="высота колонки, px (обрезка/паддинг)")
    s.set_defaults(func=cmd_stitch)
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
