"""
run_benchmark_batched.py
Что изменилось относительно run_benchmark.py и почему:

1. Датасет читается ИЗ HF STREAMING ПОБАТЧНО, фиксированными чанками
   --batch-size (по умолчанию 10000), а не куском в память целиком. На 70k
   сэмплов старый код: (а) держит 70k PIL-картинок в памяти разом, (б) отдаёт
   70k разговоров в один llm.chat() (это само по себе нормально для vLLM —
   continuous batching, — но валится по памяти на этапе подготовки/рендера
   эталонов ДО генерации, т.к. ref_infos/PNG на диске растут неограниченно),
   (в) копит ВСЕ строки results.csv в памяти и пишет один раз в конце — при
   падении где-то на 50000-м сэмпле теряется вообще всё.

2. После рендера+метрик каждого батча каталог сэмпла (html/png файлы)
   УДАЛЯЕТСЯ. Полностью, кроме заранее выбранных --n-examples-per-batch
   (по умолчанию 5) случайных сэмплов батча — их pred.png/ref.png/pred.html/
   ref.html копируются в examples/batch_XXXX/ и остаются на диске как
   иллюстрация "как выглядит генерация". Это единственное, что оставляем от
   каждого батча в виде файлов.

3. results.csv больше не содержит по строке на сэмпл (при 70k сэмплах это
   бессмысленно раздувает файл ради данных, которые всё равно тут же
   агрегируются). Вместо этого туда пишутся ТОЛЬКО строки тех же ~5
   example-сэмплов на батч (с пометкой batch_idx) — как читаемая иллюстрация,
   а не как источник итоговой статистики.

4. Итоговая оценка (среднее по всем 5 метрикам + final_score/
   final_score_arithmetic) считается ТОЧНО, а не как "среднее средних
   батчей". Если бы мы просто усредняли per-batch means, то:
     - последний батч почти наверняка неполный (70000 % 10000 == 0 тут
       совпадает, но n_samples может быть любым, напр. 70123);
     - часть сэмплов в батче падает с generation_error / metric_error и не
       участвует в среднем — то есть "полные" батчи по факту дают разное
       число оценённых строк.
   При простом среднем средних маленький/неполный батч получает такой же вес,
   как батч из 10000 оценённых строк — это смещает итоговую оценку. Вместо
   этого копится РАННИНГ SUM и RANNING COUNT по каждой метрике (см.
   MetricAccumulator ниже): sum += batch_mean * batch_count, count +=
   batch_count. В конце final_mean = sum / count. Это математически то же
   самое, что честное среднее по всем оценённым сэмплам сразу (взвешенное по
   размеру батча), но не требует держать все 70k строк ни в памяти, ни в CSV.

5. ЧЕКПОИНТ / RESUME. Каждый обработанный батч фиксируется в
   {outdir}/progress.json: {"next_batch_idx": N, "dataset_offset": M}. Плюс
   там же сохраняется текущее состояние аккумулятора метрик (суммы и counts),
   чтобы после падения на батче №23 из 7 перезапуск:
     - пропустил уже обработанные батчи 0..22 (offset в streaming-датасете
       уже известен, повторно их читать/генерировать/рендерить не нужно);
     - продолжил аккумулировать метрики с того состояния, на котором
       остановился, а не с нуля.
   Это даёт точный (не приближённый) финальный результат даже после
   рестартов, т.к. сам аккумулятор и есть источник истины для среднего —
   никакие строки для его пересчёта заново не нужны.

НЕ переиспользуется как импорт из run_benchmark.py специально — вместо этого
скопированы load_model/generate_html_batch/extract_html/render_and_score_one/
PROMPT без изменений (тот же контракт), чтобы не тащить сюда всю функцию
main() старого файла и не разбираться, что в ней резать. render.py и
metrics.py используются как есть, без изменений.

Запуск:
    python run_benchmark_batched.py --n-samples 70000 --outdir ./results
    # после падения на середине — просто перезапустить ту же команду:
    python run_benchmark_batched.py --n-samples 70000 --outdir ./results
"""

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# См. run_benchmark.py — та же причина (tensor_parallel_size > 1 форкает
# GPU-воркеров, форк после CUDA-инициализации в главном процессе падает).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

METRIC_KEYS = ["block_match", "text", "position", "color", "clip",
               "final_score", "final_score_arithmetic"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3.5-9B на Design2Code, батчами по HF-streaming (для 10k-100k+ сэмплов).")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--n-samples", type=int, default=70_000,
                         help="Сколько сэмплов всего обработать (по всем батчам).")
    parser.add_argument("--batch-size", type=int, default=10_000,
                         help="Размер одного динамического батча из HF (генерация+рендер+метрики "
                              "прогоняются на нём целиком, потом файлы батча удаляются).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--outdir", default="./design2code_results")
    parser.add_argument("--hf-dataset", default="HuggingFaceM4/WebSight")
    parser.add_argument("--hf-config", default="v0.2",
                         help="имя конфига датасета (WebSight -> v0.2; Design2Code-hf -> default).")
    parser.add_argument("--hf-split", default="train",
                         help="сплит датасета (у WebSight и Design2Code-hf -> train).")
    parser.add_argument("--shuffle-buffer-size", type=int, default=10_000,
                         help="buffer_size для ds.shuffle() в streaming-режиме HF datasets.")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=24384)
    parser.add_argument("--num-workers", type=int, default=16,
                         help="Параллельные процессы для рендера+метрик внутри одного батча.")
    parser.add_argument("--n-examples-per-batch", type=int, default=5,
                         help="Сколько случайных сэмплов батча сохранить как пример "
                              "(html+png) и как строку в results.csv.")
    parser.add_argument("--no-resume", action="store_true",
                         help="Игнорировать progress.json и начать с батча 0 "
                              "(старые examples/ и results.csv будут перезаписаны).")
    # Пиксельный бюджет картинки — ДОЛЖЕН совпадать с обучением
    # (SFT/train/formatting.py). Без него vLLM берёт родное разрешение картинки,
    # и чекпоинт бенчится вне своего трейн-распределения.
    parser.add_argument("--min-pixels", type=int, default=262_144,
                         help="min_pixels процессора (как в SFT). 262144 = 256*32*32")
    parser.add_argument("--max-pixels", type=int, default=2_097_152,
                         help="max_pixels процессора (как в SFT, Tier A). 2097152 = 2048*32*32")
    return parser.parse_args()


# =============================================================================
# Загрузка модели / датасета / генерация — без изменений по сути
# относительно run_benchmark.py, скопировано, чтобы не импортировать main()
# целиком из старого файла.
# =============================================================================

def load_model(model_id_or_path: str, tensor_parallel_size: int, gpu_memory_utilization: float, max_model_len: int, min_pixels: int, max_pixels: int):
    from vllm import LLM

    print(f"[run_benchmark] Загружаю модель {model_id_or_path} через vLLM...")
    return LLM(
        model=model_id_or_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
        # тот же пиксельный бюджет, что в обучении (см. комментарий в parse_args)
        mm_processor_kwargs={"min_pixels": min_pixels, "max_pixels": max_pixels},
    )

PROMPT = (
    "You are an expert front-end developer. Look at this webpage screenshot and "
    "write a SINGLE self-contained HTML file (inline <style>, no external CSS/JS/fonts, "
    "no network requests) that reproduces the layout, text, and colors as closely as "
    "possible. Use plain gray placeholder boxes instead of any real images. "
    "Output ONLY the raw HTML code, with no explanation and no markdown code fences."
)

def extract_html(text: str) -> str:
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
    # (html, finish_reason): finish_reason="length" = упёрлись в max_new_tokens
    # (HTML оборван), "stop" = модель сама закрыла ход.
    return [
        (extract_html(output.outputs[0].text), output.outputs[0].finish_reason)
        for output in outputs
    ]


def prepare_ref_one(idx: int, ref_html: str, sample_dir_str: str) -> dict:
    """Воркер для ProcessPoolExecutor — только рендер эталона, без импорта
    metrics.py (CLIP не нужен для подготовки эталонов, поэтому здесь нет
    смысла грузить его на GPU в каждом процессе, в отличие от
    render_and_score_one ниже, которому score_pair нужен)."""
    from pathlib import Path as _Path
    from render import prepare_and_render

    sample_dir = _Path(sample_dir_str)
    sample_dir.mkdir(exist_ok=True)
    info = prepare_and_render(
        ref_html,
        str(sample_dir / "ref.html"),
        str(sample_dir / "ref.png"),
    )
    info["idx"] = idx
    return info


def render_and_score_one(idx: int, pred_html: str, sample_dir_str: str, finish_reason: str = None) -> dict:
    """Идентично run_benchmark.py — воркер для ProcessPoolExecutor (или прямой
    вызов при num_workers<=1). Импорт render/metrics внутри функции (см.
    комментарий в оригинале про spawn + CUDA)."""
    from pathlib import Path as _Path
    from render import prepare_and_render
    from metrics import score_pair

    sample_dir = _Path(sample_dir_str)
    row = {"idx": idx, "finish_reason": finish_reason}

    if pred_html is None:
        row["status"] = "generation_error"
        return row

    pred_info = prepare_and_render(
        pred_html,
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

    return row


# =============================================================================
# Потоковая загрузка датасета батчами фиксированного размера
# =============================================================================

def iter_dataset_batches(hf_dataset: str, n_samples: int, batch_size: int, seed: int,
                          shuffle_buffer_size: int, skip: int = 0,
                          hf_config: str = "v0.2", hf_split: str = "train"):
    """Генератор: отдаёт список HF-сэмплов (dict с ключами 'text'/'image') по
    batch_size штук за раз, пока не наберётся n_samples суммарно.

    skip: сколько сэмплов уже обработано в предыдущих запусках (для resume) -
    именно столько записей streaming-итератора пропускаем перед тем, как
    начать копить первый батч этого запуска. Порядок стабилен, потому что
    shuffle(seed=...) с тем же seed всегда даёт одну и ту же перестановку
    (при том же buffer_size) - см. https://huggingface.co/docs/datasets - так
    что skip=N всегда пропускает те же самые N сэмплов, что уже обработаны.
    """
    from datasets import load_dataset

    print(f"[run_benchmark] Открываю {hf_dataset} в streaming-режиме (skip={skip})...")
    ds_stream = load_dataset(hf_dataset, name=hf_config, split=hf_split, streaming=True)
    ds_stream = ds_stream.shuffle(seed=seed, buffer_size=shuffle_buffer_size)

    it = iter(ds_stream)

    for _ in range(skip):
        try:
            next(it)
        except StopIteration:
            print("[run_benchmark] Датасет закончился раньше, чем ожидал skip — нечего резюмировать.")
            return

    remaining = n_samples - skip
    while remaining > 0:
        take_n = min(batch_size, remaining)
        batch = []
        for _ in range(take_n):
            try:
                batch.append(next(it))
            except StopIteration:
                break
        if not batch:
            return
        yield batch
        remaining -= len(batch)
        if len(batch) < take_n:
            return


# =============================================================================
# Точное взвешенное усреднение метрик по батчам (см. пункт 4 в docstring файла)
# =============================================================================

class MetricAccumulator:
    """running sum/count на метрику. mean() = sum/count всегда равен честному
    среднему по всем виденным до сих пор оценённым сэмплам, независимо от
    того, как они были разбиты на батчи (в т.ч. неполный последний батч,
    в т.ч. разное число упавших сэмплов в разных батчах)."""

    def __init__(self):
        self.sums = {k: 0.0 for k in METRIC_KEYS}
        self.count = 0

    def add_batch(self, ok_df: pd.DataFrame):
        n = len(ok_df)
        if n == 0:
            return
        for k in METRIC_KEYS:
            self.sums[k] += float(ok_df[k].sum())
        self.count += n

    def means(self) -> dict:
        if self.count == 0:
            return {k: float("nan") for k in METRIC_KEYS}
        return {k: self.sums[k] / self.count for k in METRIC_KEYS}

    def to_state(self) -> dict:
        return {"sums": self.sums, "count": self.count}

    @classmethod
    def from_state(cls, state: dict) -> "MetricAccumulator":
        acc = cls()
        if state:
            acc.sums = {k: float(state["sums"].get(k, 0.0)) for k in METRIC_KEYS}
            acc.count = int(state["count"])
        return acc


# =============================================================================
# Прогресс / чекпоинт
# =============================================================================

def load_progress(progress_path: Path) -> dict:
    if progress_path.exists():
        return json.loads(progress_path.read_text(encoding="utf-8"))
    return {"next_batch_idx": 0, "dataset_offset": 0, "n_generation_errors": 0,
            "n_metric_errors": 0, "n_other_errors": 0, "n_length_truncated": 0,
            "accumulator": None}


def save_progress(progress_path: Path, progress: dict):
    # Пишем во временный файл и переименовываем атомарно (os.replace) - чтобы
    # падение процесса ровно в момент записи не оставило progress.json в
    # битом полусохранённом состоянии, из-за которого resume был бы невозможен.
    tmp_path = progress_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, progress_path)


def log_time(outdir: Path, stage_name, duration_seconds):
    with open(outdir / "time_results.txt", "a", encoding="utf-8") as f:
        f.write(f"{stage_name}: {duration_seconds:.6f} сек.\n")


# =============================================================================
# Обработка одного батча: генерация -> рендер+метрики -> примеры -> очистка
# =============================================================================

def process_one_batch(llm, batch_samples, batch_idx: int, work_root: Path, examples_root: Path,
                       args, rng: random.Random):
    """Возвращает (examples_df, ok_df, n_generation_errors, n_metric_errors, n_other_errors).
    work_root: временный каталог этого батча (html/png сэмплов) - целиком
    удаляется в конце функции, кроме файлов, скопированных в examples_root.
    """
    batch_dir = work_root / f"batch_{batch_idx:05d}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    n = len(batch_samples)

    # --- эталоны ---
    # CPU/IO-bound (Playwright), CLIP/GPU тут не участвует вообще (см.
    # prepare_ref_one) — поэтому в отличие от render_and_score_one здесь нет
    # ограничения по VRAM, параллелим на общих правах с --num-workers.
    if args.num_workers <= 1:
        from render import prepare_and_render
        ref_infos_by_idx = {}
        for i in tqdm(range(n), desc=f"[batch {batch_idx}] Подготовка эталонов"):
            sample_dir = batch_dir / f"sample_{i:05d}"
            sample_dir.mkdir(exist_ok=True)
            ref_infos_by_idx[i] = prepare_and_render(
                batch_samples[i]["text"],
                str(sample_dir / "ref.html"),
                str(sample_dir / "ref.png"),
            )
    else:
        from render import close_browser
        close_browser()  # браузер главного процесса не нужен — рендерят воркеры

        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        ctx = mp.get_context("spawn")
        ref_infos_by_idx = {}
        print(f"[batch {batch_idx}] Подготовка эталонов на {args.num_workers} процессах...")
        with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(prepare_ref_one, i, batch_samples[i]["text"],
                                 str(batch_dir / f"sample_{i:05d}")): i
                for i in range(n)
            }
            for future in tqdm(as_completed(futures), total=len(futures),
                                desc=f"[batch {batch_idx}] Подготовка эталонов"):
                i = futures[future]
                try:
                    ref_infos_by_idx[i] = future.result()
                except Exception as e:
                    # рендер эталона упал целиком (не просто плохая картинка,
                    # а исключение в воркере) — считаем эталон "не готов",
                    # ref_render_ok=False, скор по этому сэмплу всё равно
                    # посчитается (score_pair сам разберётся с плохим PNG),
                    # но явно фиксируем причину для отладки.
                    print(f"[batch {batch_idx}] sample {i}: ошибка подготовки эталона: {e}")
                    ref_infos_by_idx[i] = {"n_images_replaced": 0, "render_ok": False}

    ref_infos = [ref_infos_by_idx[i] for i in range(n)]

    # --- генерация ---
    print(f"[batch {batch_idx}] Генерирую HTML для {n} сэмплов...")
    images = [batch_samples[i]["image"].convert("RGB") for i in range(n)]
    try:
        pred_pairs = generate_html_batch(llm, images, args.max_new_tokens, args.enable_thinking)
    except Exception as e:
        print(f"[batch {batch_idx}] Ошибка батчевой генерации: {e}")
        pred_pairs = [(None, "batch_error")] * n
    del images

    # --- рендер + метрики ---
    if args.num_workers <= 1:
        rows = []
        for i in tqdm(range(n), desc=f"[batch {batch_idx}] Рендер + метрики"):
            sample_dir = batch_dir / f"sample_{i:05d}"
            row = render_and_score_one(i, pred_pairs[i][0], str(sample_dir), pred_pairs[i][1])
            row["ref_n_img_replaced"] = ref_infos[i]["n_images_replaced"]
            row["ref_render_ok"] = ref_infos[i]["render_ok"]
            rows.append(row)
    else:
        from render import close_browser
        close_browser()  # см. run_benchmark.py: браузер главного процесса не нужен воркерам

        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        ctx = mp.get_context("spawn")
        rows_by_idx = {}
        print(f"[batch {batch_idx}] Рендер+метрики на {args.num_workers} процессах...")
        with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(render_and_score_one, i, pred_pairs[i][0],
                                 str(batch_dir / f"sample_{i:05d}"), pred_pairs[i][1]): i
                for i in range(n)
            }
            for future in tqdm(as_completed(futures), total=len(futures),
                                desc=f"[batch {batch_idx}] Рендер + метрики"):
                i = futures[future]
                try:
                    row = future.result()
                except Exception as e:
                    row = {"idx": i, "status": f"worker_error: {e}", "finish_reason": pred_pairs[i][1]}
                row["ref_n_img_replaced"] = ref_infos[i]["n_images_replaced"]
                row["ref_render_ok"] = ref_infos[i]["render_ok"]
                rows_by_idx[i] = row
        rows = [rows_by_idx[i] for i in range(n)]

    df = pd.DataFrame(rows)
    n_generation_errors = int((df["status"] == "generation_error").sum())
    n_metric_errors = int(df["status"].str.startswith("metric_error", na=False).sum()) \
        if "status" in df else 0
    # "Прочее": worker_error (сам процесс-воркер упал в ProcessPoolExecutor,
    # см. except Exception выше) или любой другой статус, который не
    # "scored"/"generation_error"/"metric_error: ...". Без этой строки такие
    # сэмплы молча пропадали бы из сводки - не попадали ни в один из
    # счётчиков ошибок, ни в n_scored, и n_samples_processed - n_scored
    # переставало бы совпадать с суммой известных причин.
    n_known_bad = n_generation_errors + n_metric_errors
    n_other_errors = int(len(df) - (df["status"] == "scored").sum() - n_known_bad) \
        if "status" in df else 0

    # Обрезка по лимиту токенов: finish_reason="length" = модель упёрлась в
    # max_new_tokens и HTML оборван. Это НЕ generation_error (сэмпл всё равно
    # отрендерится и посчитается), но метрики по нему занижены — считаем отдельно,
    # чтобы отличить «модель пишет слишком длинно» от упавших батчей.
    n_length_truncated = int((df["finish_reason"] == "length").sum()) \
        if "finish_reason" in df else 0

    # --- выбираем случайные примеры ДО удаления файлов батча ---
    scored_idx = df.index[df["status"] == "scored"].tolist()
    n_examples = min(args.n_examples_per_batch, len(scored_idx))
    example_indices = rng.sample(scored_idx, n_examples) if n_examples > 0 else []

    example_out_dir = examples_root / f"batch_{batch_idx:05d}"
    if example_indices:
        example_out_dir.mkdir(parents=True, exist_ok=True)

    example_rows = []
    for i in example_indices:
        sample_dir = batch_dir / f"sample_{i:05d}"
        dest_dir = example_out_dir / f"sample_{i:05d}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("pred.html", "pred.png", "ref.html", "ref.png"):
            src = sample_dir / fname
            if src.exists():
                shutil.copy2(src, dest_dir / fname)
        row = df.loc[i].to_dict()
        row["batch_idx"] = batch_idx
        row["example_dir"] = str(dest_dir)
        example_rows.append(row)

    examples_df = pd.DataFrame(example_rows)

    # --- аккумулируем метрики по ВСЕМ scored-сэмплам батча (не только examples) ---
    ok_df = df[df["status"] == "scored"]

    # --- полная очистка временных файлов батча ---
    shutil.rmtree(batch_dir, ignore_errors=True)

    return examples_df, ok_df, n_generation_errors, n_metric_errors, n_other_errors, n_length_truncated


def append_examples_csv(results_path: Path, examples_df: pd.DataFrame):
    if examples_df.empty:
        return
    write_header = not results_path.exists()
    examples_df.to_csv(results_path, mode="a", header=write_header, index=False)


def main():
    args = parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from tracking import log_benchmark_summary, start_benchmark_task

    # ClearML: одна задача на прогон бенча (сетап × final_score). No-op без clearml.
    cml_task = start_benchmark_task(args)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    work_root = outdir / "_work"       # временные файлы текущего батча (удаляются после каждого батча)
    examples_root = outdir / "examples"  # то немногое, что остаётся на диске
    results_path = outdir / "results.csv"
    progress_path = outdir / "progress.json"

    if args.no_resume:
        for p in (work_root, examples_root, results_path, progress_path):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink()

    work_root.mkdir(parents=True, exist_ok=True)
    examples_root.mkdir(parents=True, exist_ok=True)

    progress = load_progress(progress_path)
    start_batch_idx = progress["next_batch_idx"]
    dataset_offset = progress["dataset_offset"]
    accumulator = MetricAccumulator.from_state(progress.get("accumulator"))
    n_generation_errors = progress.get("n_generation_errors", 0)
    n_metric_errors = progress.get("n_metric_errors", 0)
    n_other_errors = progress.get("n_other_errors", 0)
    n_length_truncated = progress.get("n_length_truncated", 0)

    if start_batch_idx > 0:
        print(f"[run_benchmark] Резюмирую с батча {start_batch_idx} "
              f"(уже обработано сэмплов: {dataset_offset}, накоплено метрик: {accumulator.count}).")

    llm = load_model(args.model, args.tensor_parallel_size, args.gpu_memory_utilization, args.max_model_len, args.min_pixels, args.max_pixels)

    # # metrics.py импортируется только после LLM (см. run_benchmark.py — CUDA/fork).
    # import metrics  # noqa: F401  (гарантирует, что CLIP грузится один раз здесь)

    rng = random.Random(args.seed + 1)  # отдельный seed для выбора examples, не путать с shuffle датасета

    batch_gen = iter_dataset_batches(
        args.hf_dataset, args.n_samples, args.batch_size, args.seed,
        args.shuffle_buffer_size, skip=dataset_offset,
        hf_config=args.hf_config, hf_split=args.hf_split,
    )

    batch_idx = start_batch_idx
    for batch_samples in batch_gen:
        t0 = time.perf_counter()
        print(f"\n=== Батч {batch_idx} ({len(batch_samples)} сэмплов, "
              f"offset {dataset_offset}..{dataset_offset + len(batch_samples)}) ===")

        examples_df, ok_df, n_gen_err, n_met_err, n_other_err, n_len_trunc = process_one_batch(
            llm, batch_samples, batch_idx, work_root, examples_root, args, rng,
        )

        append_examples_csv(results_path, examples_df)
        accumulator.add_batch(ok_df)
        n_generation_errors += n_gen_err
        n_metric_errors += n_met_err
        n_other_errors += n_other_err
        n_length_truncated += n_len_trunc
        dataset_offset += len(batch_samples)
        batch_idx += 1

        if n_len_trunc > 0:
            print(f"[batch {batch_idx - 1}] Обрезано по лимиту токенов "
                  f"(finish_reason=length): {n_len_trunc}/{len(batch_samples)} "
                  f"({n_len_trunc / len(batch_samples):.1%}) — метрики по ним занижены.")

        progress = {
            "next_batch_idx": batch_idx,
            "dataset_offset": dataset_offset,
            "n_generation_errors": n_generation_errors,
            "n_metric_errors": n_metric_errors,
            "n_other_errors": n_other_errors,
            "n_length_truncated": n_length_truncated,
            "accumulator": accumulator.to_state(),
        }
        save_progress(progress_path, progress)

        if n_other_err > 0:
            print(f"[batch {batch_idx - 1}] Внимание: {n_other_err} сэмпл(ов) с "
                  f"неучтённым статусом (например worker_error) — исключены из среднего.")

        duration = time.perf_counter() - t0
        log_time(outdir, f"Батч {batch_idx - 1}", duration)
        means_so_far = accumulator.means()
        print(f"[batch {batch_idx - 1}] {duration:.1f} сек. Накопленное среднее "
              f"({accumulator.count} оценённых сэмплов): "
              + ", ".join(f"{k}={v:.4f}" for k, v in means_so_far.items()))

    from render import close_browser
    close_browser()

    final_means = accumulator.means()
    # n_excluded_total — сумма ВСЕХ причин, по которым сэмпл не попал в
    # среднее (generation_error + metric_error + прочее/worker_error), и
    # отдельно сверка dataset_offset == n_scored + n_excluded_total: если эта
    # сверка когда-нибудь разойдётся, значит появился ещё не учтённый статус
    # в render_and_score_one, и это будет видно сразу, а не тихо потеряется.
    n_excluded_total = n_generation_errors + n_metric_errors + n_other_errors
    assert dataset_offset == accumulator.count + n_excluded_total, (
        f"Расхождение в подсчёте: обработано {dataset_offset}, но "
        f"scored({accumulator.count}) + excluded({n_excluded_total}) != обработано. "
        f"Значит render_and_score_one вернул статус, который нигде не учтён."
    )
    summary = {
        "n_samples_requested": args.n_samples,
        "n_samples_processed": dataset_offset,
        "n_scored": accumulator.count,
        "n_excluded_total": n_excluded_total,
        "n_generation_errors": n_generation_errors,
        "n_metric_errors": n_metric_errors,
        "n_other_errors": n_other_errors,
        # ортогонально n_excluded_total: обрезанные по токенам сэмплы обычно
        # всё равно scored, но с заниженными метриками — поэтому отдельным полем.
        "n_length_truncated": n_length_truncated,
        **final_means,
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log_benchmark_summary(cml_task, summary, results_path)

    print(f"\nМодель: {args.model}")
    print(f"Обработано сэмплов: {dataset_offset} / {args.n_samples}")
    print(f"Успешно оценено: {accumulator.count}")
    print(f"Исключено из среднего: {n_excluded_total} "
          f"(generation_error: {n_generation_errors}, metric_error: {n_metric_errors}, "
          f"прочее: {n_other_errors})")
    if dataset_offset:
        print(f"Обрезано по лимиту токенов (finish_reason=length): {n_length_truncated} "
              f"({n_length_truncated / dataset_offset:.1%} от обработанных) — "
              f"такие сэмплы обычно scored, но с заниженными метриками.")
    print("Итоговые средние по всем оценённым сэмплам (взвешенно, не среднее средних батчей):")
    for k, v in final_means.items():
        print(f"  {k}: {v:.4f}" if not math.isnan(v) else f"  {k}: nan")
    print(f"\nПримеры сохранены в: {examples_root}")
    print(f"Строки-примеры (results.csv): {results_path}")
    print(f"Итоговая сводка: {summary_path}")


if __name__ == "__main__":
    main()
