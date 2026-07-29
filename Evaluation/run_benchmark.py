"""
run_benchmark.py — запуск Qwen3.5-9B на бенчмарке Design2Code, локально, через vLLM.

Что делает:
1. Загружает модель через vLLM (LLM(...) сам скачает с Hugging Face при первом
   запуске и закэширует локально как обычно).
2. Загружает датасет Design2Code-hf (datasets.load_dataset — сам скачает при
   первом запуске в ~/.cache/huggingface, дальше берёт из кэша).
3. Генерирует HTML для ВСЕХ сэмплов ОДНИМ батчем через llm.chat() (continuous
   batching vLLM — сильно быстрее, чем цикл с одним generate() за раз).
4. Для каждого сэмпла прогоняет результат через render.py (замена <img> на
   плейсхолдер + рендер) и metrics.py (официальные метрики, final_score =
   среднее геометрическое).
5. Сохраняет results.csv и сырые файлы (html/png) в --outdir.

Запуск:
    python run_benchmark.py --n-samples 10 --outdir ./results
    python run_benchmark.py --model /path/to/local/weights --n-samples 10
"""

import argparse
import os
import re
import sys
from pathlib import Path
import time

import pandas as pd
from tqdm import tqdm

# vLLM с tensor_parallel_size > 1 форкает GPU-воркеров под капотом; если CUDA
# уже была затронута в главном процессе до этого, "fork" падает с
# "Cannot re-initialize CUDA in forked subprocess". vLLM обычно определяет это
# сам и переключается на spawn, но не всегда надёжно — поэтому задаём явно
# через переменную окружения, которую vLLM реально читает (в отличие от
# multiprocessing.set_start_method на стороне Python, который здесь не
# действует, так как vLLM создаёт свой собственный mp-контекст).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3.5-9B на бенчмарке Design2Code, локально (vLLM).")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B",
                         help="HF id или локальный путь к весам (vLLM сам разберётся)")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--outdir", default="./design2code_results")
    parser.add_argument("--hf-dataset", default="SALT-NLP/Design2Code-hf")
    parser.add_argument("--enable-thinking", action="store_true",
                         help="Не отключать <think>...</think> — по умолчанию отключено")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                         help="Число GPU для tensor parallelism (vLLM)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.89,
                         help="Доля памяти GPU, отдаваемая под KV-cache и веса (vLLM)")
    parser.add_argument("--max-model-len", type=int, default=16384,
                         help="Максимальная длина контекста (vLLM)")
    return parser.parse_args()


def load_design2code_dataset(hf_dataset: str, n_samples: int, seed: int):
    """load_dataset сам скачает датасет при первом запуске и закэширует в
    ~/.cache/huggingface/datasets — при повторных запусках просто читает кэш."""
    from datasets import load_dataset

    print(f"[run_benchmark] Загружаю датасет {hf_dataset} (при первом запуске скачается)...")
    ds_full = load_dataset(hf_dataset, split="train")
    print(f"[run_benchmark] Всего примеров: {len(ds_full)}")
    ds = ds_full.shuffle(seed=seed).select(range(min(n_samples, len(ds_full))))
    print(f"[run_benchmark] Взяли в подвыборку: {len(ds)}")
    return ds


def load_model(model_id_or_path: str, tensor_parallel_size: int, gpu_memory_utilization: float, max_model_len: int):
    """Загружает модель через vLLM. limit_mm_per_prompt={"image": 1} — каждому
    сэмплу нужна ровно одна картинка-скриншот; ограничиваем явно, чтобы vLLM
    выделил под мультимодальный кэш ровно столько, сколько нужно."""
    from vllm import LLM

    print(f"[run_benchmark] Загружаю модель {model_id_or_path} через vLLM (при первом запуске скачается с HF)...")
    llm = LLM(
        model=model_id_or_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
    )
    return llm


PROMPT = (
"Generate a single self-contained HTML file with precompiled Tailwind. Replace images with gray placeholder blocks."
)


def extract_html(text: str) -> str:
    # На всякий случай убираем <think>...</think>, если thinking всё же включён
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


def generate_html_batch(llm, images, max_new_tokens: int, enable_thinking: bool) -> list[str]:
    """Генерирует HTML для всех сэмплов одним батчем через llm.chat() — вместо
    цикла с одним generate() за сэмпл. vLLM сам занимается continuous batching
    и планированием, так что сюда можно отдавать хоть весь датасет разом."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(max_tokens=max_new_tokens)

    conversations = [
        [{"role": "user", "content": [{"type": "image_pil", "image_pil": image}, {"type": "text", "text": PROMPT}]}]
        for image in images
    ]

    outputs = llm.chat(
        conversations,
        sampling_params=sampling_params,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )
    max_len = 0
    min_len = 100_000
    for i, output in enumerate(outputs):
        max_len = max(max_len, len(output.outputs[0].token_ids))
        min_len = min(min_len, len(output.outputs[0].token_ids))

    print(f"Максимальный промпт: {max_len}, минимальный: {min_len}")

    return [extract_html(output.outputs[0].text) for output in outputs]

def log_time(stage_name, duration_seconds):
    with open("time_results.txt", "a", encoding="utf-8") as f:
        f.write(f"{stage_name}: {duration_seconds:.6f} сек.\n")

def main():
    start_main = time.perf_counter()
    print("Программа запущена...")
    args = parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from render import prepare_and_render

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ds = load_design2code_dataset(args.hf_dataset, args.n_samples, args.seed)

    # metrics.py импортируется только ПОСЛЕ создания LLM: он на импорте грузит
    # CLIP и тем самым инициализирует CUDA в этом (главном) процессе. При
    # tensor_parallel_size > 1 vLLM форкает воркеров под GPU-процессы, а форк
    # процесса с уже инициализированным CUDA-контекстом падает с
    # "Cannot re-initialize CUDA in forked subprocess". Поэтому сначала
    # поднимаем vLLM (он форкает воркеров сам, аккуратно), и только потом
    # трогаем CLIP/CUDA в главном процессе.
    llm = load_model(args.model, args.tensor_parallel_size, args.gpu_memory_utilization, args.max_model_len)
    from metrics import score_pair
    print("Thinking:", args.enable_thinking)

    start_prep = time.perf_counter()
    print("Начало подготовки данных...")

    # --- эталоны: подготавливаем и рендерим заранее, до генерации ---
    ref_infos = []
    for i in tqdm(range(len(ds)), desc="Подготовка эталонов"):
        sample_dir = outdir / f"sample_{i:04d}"
        sample_dir.mkdir(exist_ok=True)
        ref_infos.append(prepare_and_render(
            ds[i]["text"],
            str(sample_dir / "ref.html"),
            str(sample_dir / "ref.png"),
        ))
    
    end_prep = time.perf_counter()
    start_generation = time.perf_counter()
    print("Начало генерации данных...")

    # --- генерация HTML для всех сэмплов одним батчем ---
    print(f"[run_benchmark] Генерирую HTML для {len(ds)} сэмплов одним батчем...")
    images = [ds[i]["image"].convert("RGB") for i in range(len(ds))]
    try:
        pred_html_list = generate_html_batch(llm, images, args.max_new_tokens, args.enable_thinking)
    except Exception as e:
        print(f"[run_benchmark] Ошибка батчевой генерации: {e}")
        pred_html_list = [None] * len(ds)

    end_generation = time.perf_counter()
    start_render = time.perf_counter()
    print("Начало рендера...")

    # --- рендер предсказаний + метрики, сэмпл за сэмплом ---
    rows = []
    old = time.perf_counter()
    for i in tqdm(range(len(ds)), desc="Рендер + метрики"):
        print(f"Рендер {i} семпла, {time.perf_counter() - old} сек.")
        old = time.perf_counter()
        sample_dir = outdir / f"sample_{i:04d}"
        row = {"idx": i, "ref_n_img_replaced": ref_infos[i]["n_images_replaced"], "ref_render_ok": ref_infos[i]["render_ok"]}

        if pred_html_list[i] is None:
            row["status"] = "generation_error"
            rows.append(row)
            continue

        pred_info = prepare_and_render(
            pred_html_list[i],
            str(sample_dir / "pred.html"),
            str(sample_dir / "pred.png"),
        )
        row.update({"pred_n_img_replaced": pred_info["n_images_replaced"], "pred_render_ok": pred_info["render_ok"]})

        try:
            scores = score_pair(str(sample_dir / "pred.html"), str(sample_dir / "ref.html"))
            row.update(scores)
            row["status"] = "scored"
        except Exception as e:
            row["status"] = f"metric_error: {e}"

        rows.append(row)

    end_render = time.perf_counter()
    end_main = time.perf_counter()

    duration_main = end_main - start_main
    duration_generation = end_generation - start_generation
    duration_prep = end_prep - start_prep
    duration_render = end_render - start_render

    log_time("С начала main", duration_main)
    log_time("С начала подготовки", duration_prep)
    log_time("С начала генерации", duration_generation)
    log_time("С начала рендера", duration_render)

    with open("time_results.txt", "a", encoding="utf-8") as f:
        f.write("-" * 30 + "\n")

    from render import close_browser
    close_browser()  # общий Chromium (см. render.py) больше не нужен — закрываем явно

    df = pd.DataFrame(rows)
    results_path = outdir / "results.csv"
    df.to_csv(results_path, index=False)

    ok_df = df[df["status"] == "scored"]
    print(f"\nМодель: {args.model}")
    print(f"Успешно оценено: {len(ok_df)} / {len(df)}")
    if len(ok_df) > 0:
        print(ok_df[["block_match", "text", "position", "color", "clip", "final_score", "final_score_arithmetic"]].mean().to_string())
    print(f"\nРезультаты: {results_path}")


if __name__ == "__main__":
    main()
