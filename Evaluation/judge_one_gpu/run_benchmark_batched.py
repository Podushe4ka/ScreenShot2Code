"""
run_benchmark_batched.py — Design2Code: 5 официальных метрик для
модели-чекпоинта + pairwise LLM-judge (чекпоинт vs baseline), батчами по
HF-streaming, на одной GPU.

Модель, которую реально оцениваем — --model-checkpoint (5 официальных метрик
Design2Code: block_match/text/position/color/clip/final_score/
final_score_arithmetic, через metrics.score_pair против ref).
--model-baseline существует только чтобы дать судье второй скриншот для
сравнения — по ней официальные метрики не считаются вовсе. --judge-model
(третья модель) получает ref.png + pred чекпоинта + pred baseline и решает,
какой ближе к ref — итог: winrate чекпоинта относительно baseline.

Одна физическая GPU — три модели на ней одновременно не помещаются, вес
модели (checkpoint/baseline/judge) грузится и выгружается по очереди на
каждом батче (см. vllm_server_manager.py). Но там, где два соседних шага не
делят GPU-VRAM модели (browser/CPU-рендер против vLLM-деплоя), они идут
параллельно в двух потоках оркестрации, а не строго последовательно:

    1. Поднять checkpoint  -> сгенерировать HTML по всему батчу -> погасить
    2. || Рендер эталона + рендер и 5 официальных метрик чекпоинта (Playwright
         + CLIP — GPU от vLLM в этот момент уже свободен, см. ниже про CLIP)
       || Поднять baseline -> сгенерировать HTML по всему батчу
       (не пересекаются по ресурсу — идут одновременно)
    4. || Рендер baseline (только PNG, без метрик)
       || Поднять judge (только загрузка весов — сам judge-инференс ждёт PNG)
    5. Judge: pairwise-сравнение по всему батчу -> погасить
    6. Следующий батч -> снова с шага 1 (запись results.csv/progress.json
       предыдущего батча уходит в фоновый поток и может продолжаться, пока
       уже идёт генерация checkpoint следующего батча — см. main())

Один HTTP-порт (--vllm-port, дефолт 8001) переиспользуется для всех трёх
моделей по очереди — vllm_client.VLLMClient не знает о переключениях, просто
шлёт запросы на тот же base_url; какая модель там сейчас отвечает,
определяется VLLMServerManager.switch_to() до того как клиент используется.

CLIP-сервер (clip_server.py, для метрики clip у чекпоинта) — единственное
исключение из "по очереди": маленький (~350MB VRAM) отдельный процесс,
стартует один раз в начале всего прогона и живёт до конца, деля GPU с
текущей vLLM-моделью — в том числе в моменты, когда сама vLLM-модель уже
грузится параллельно с рендером (шаги 2 и 4 выше).

Этап рендер+метрики чекпоинта — единственный CPU/browser-bound этап (на
сэмпл до 6 скриншотов, см. metrics.visual_eval_v3_multi), поэтому его
конкурентность настраивается отдельно от --num-workers через
--render-workers — иначе высокий --num-workers, разумный для I/O-bound
judge-этапа, создаёт слишком много одновременных Chromium-процессов именно
там, где рендер и так самый тяжёлый (см. render.py, DEFAULT_RENDER_TIMEOUT_MS).

Anti-position-bias у судьи: см. judge_client.py — для каждого сэмпла порядок
показа выбирается монеткой независимо.

Запуск:
    python run_benchmark_batched.py --n-samples 70000 --outdir ./results \\
        --model-baseline Qwen/Qwen3.5-4B \\
        --model-checkpoint /mnt/storage-1/checkpoints/my-run \\
        --judge-model Qwen/Qwen3.5-4B
    # (сам поднимает/гасит vllm serve по ходу дела, см. vllm_server_manager.py)
    # после падения на середине — просто перезапустить ту же команду.
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

from vllm_client import VLLMClient
from vllm_server_manager import VLLMServerManager, VLLMServerError

METRIC_KEYS = ["block_match", "text", "position", "color", "clip",
               "final_score", "final_score_arithmetic"]


# =============================================================================
# Инициализация воркера ProcessPoolExecutor — один раз на процесс
# =============================================================================
# CLIP: считается только для чекпоинта (score_pair вызывается один раз на
# сэмпл, не для baseline), через общий батчующий GPU-процесс (см. docstring
# в clip_server.py) вместо копии модели в каждом воркере.
#
# Judge-клиент: создаётся один раз на процесс (одна requests.Session, одно
# переиспользуемое TCP-соединение) вместо нового VLLMClient на каждый
# сэмпл — тот же анти-паттерн, которого CLIP избегает через set_clip_client.
# judge_sample просто читает модульную глобальную переменную _judge_client.
#
# _worker_init_judge вызывается для судейского этапа батча, когда judge уже
# поднят на --vllm-port (см. VLLMServerManager.switch_to в process_one_batch,
# шаг "judge", до создания пула воркеров). Тот же URL переиспользуется для
# всех трёх моделей по очереди — воркеру достаточно знать текущее имя модели.
_judge_client = None
_judge_error_print_count = 0
_JUDGE_ERROR_PRINT_LIMIT = 1  # на процесс — при массовом падении судьи (все
                              # сэмплы этого воркера) не дублируем один и тот
                              # же traceback повторно, первого раза достаточно
                              # для диагностики; статус ошибки (row["judge_status"])
                              # по-прежнему сохраняется для КАЖДОГО сэмпла, печать
                              # ограничена только чтобы не заспамить стдаут при
                              # --num-workers=90+ и массовом краше судьи.


def _worker_init_scoring(clip_request_queue):
    """Инициализация воркера для ЭТАПА МЕТРИК ЧЕКПОИНТА (рендер эталона +
    score_pair) — в этот момент на GPU уже нет vLLM (checkpoint только что
    отгенерировал HTML и был погашен), судья ещё не поднят, поэтому
    judge-клиент здесь не нужен вовсе."""
    from clip_server import ClipClient
    import metrics
    metrics.set_clip_client(ClipClient(clip_request_queue))


def _worker_init_judge(judge_url: str, judge_model_name: str):
    """Инициализация воркера для ЭТАПА СУДЬИ — CLIP тут не нужен (судья не
    вызывает score_pair), нужен только judge-клиент."""
    global _judge_client
    from vllm_client import VLLMClient
    _judge_client = VLLMClient(judge_url, judge_model_name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Design2Code: 5 официальных метрик для модели-чекпоинта + "
                    "pairwise LLM-judge (чекпоинт vs baseline), батчами по HF-streaming. "
                    "1 GPU: модели грузятся и выгружаются по очереди внутри каждого батча "
                    "(см. vllm_server_manager.py) — checkpoint -> baseline -> judge.")
    parser.add_argument("--model-checkpoint", required=True,
                         help="Модель, которую реально оцениваем (напр. чекпоинт "
                              "/mnt/storage-1/checkpoints/my-run/step-12000). Для неё "
                              "считаются 5 официальных метрик + она участвует в judge.")
    parser.add_argument("--model-baseline", required=True,
                         help="Baseline-модель для pairwise-сравнения (напр. Qwen/Qwen3.5-4B "
                              "'из коробки'). По ней 5 официальных метрик НЕ считаются — только "
                              "участвует в judge как вторая картинка для сравнения.")
    parser.add_argument("--judge-model", default="Qwen/Qwen3.5-4B",
                         help="Модель-судья (pairwise LLM-as-judge).")
    parser.add_argument("--vllm-port", type=int, default=8001,
                         help="Порт, на котором поднимается vllm serve — один и тот же порт "
                              "переиспользуется последовательно для checkpoint, baseline и judge "
                              "(на одной GPU модели не работают одновременно, см. vllm_server_manager.py).")
    parser.add_argument("--gpu-index", default="0",
                         help="Индекс GPU (CUDA_VISIBLE_DEVICES) для единственного vllm serve "
                              "процесса. CLIP-сервер также использует эту же GPU (см. --clip-batch-size).")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=None,
                         help="gpu-memory-utilization для vllm serve. Оставляет запас под CLIP-сервер, "
                              "который держит часть VRAM на этой же карте постоянно (см. "
                              "vllm_server_manager.DEFAULT_GPU_MEMORY_UTILIZATION).")
    parser.add_argument("--max-model-len", type=int, default=16384,
                         help="max-model-len для vllm serve — общий для всех трёх моделей "
                              "(checkpoint/baseline/judge), т.к. это один и тот же переиспользуемый процесс.")
    parser.add_argument("--n-samples", type=int, default=70_000,
                         help="Сколько сэмплов всего обработать (по всем батчам).")
    parser.add_argument("--batch-size", type=int, default=10_000,
                         help="Размер одного динамического батча из HF (генерация checkpoint -> "
                              "рендер+метрики checkpoint -> генерация baseline -> рендер baseline -> "
                              "judge прогоняются на нём целиком, потом файлы батча удаляются).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--judge-max-new-tokens", type=int, default=512,
                         help="max_tokens для ответа судьи (JSON {winner} — короткий).")
    parser.add_argument("--generation-concurrency", type=int, default=64,
                         help="Сколько запросов на генерацию HTML слать в vLLM-сервер конкурентно "
                              "с клиента (см. VLLMClient.chat_batch) — сервер сам батчует их через "
                              "continuous batching, это верхний предел на клиентской стороне.")
    parser.add_argument("--outdir", default="./design2code_results")
    parser.add_argument("--hf-dataset", default="HuggingFaceM4/WebSight")
    # Конфиг и сплит раньше были ВШИТЫ как name="v0.2", split="train" — то есть
    # --hf-dataset формально принимался, но реально работал только WebSight, а
    # любой другой датасет падал на отсутствующем конфиге "v0.2". Дефолты здесь
    # повторяют прежнее поведение, чтобы существующие вызовы не поехали.
    parser.add_argument("--hf-config", default="v0.2",
                         help="Конфиг датасета (name= у load_dataset). Для "
                              "SALT-NLP/Design2Code-hf это 'default'.")
    parser.add_argument("--hf-split", default="train",
                         help="Сплит датасета.")
    parser.add_argument("--shuffle-buffer-size", type=int, default=10_000,
                         help="buffer_size для ds.shuffle() в streaming-режиме HF datasets.")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--num-workers", type=int, default=80,
                         help="Параллельные процессы (ProcessPoolExecutor) для judge (этап 5) — "
                              "чисто I/O-bound HTTP-запросы к судье, реальная нагрузка на CPU/GPU "
                              "минимальна, можно ставить высоко. Для рендер-этапов (2 и 4, "
                              "browser-bound) используется отдельный --render-workers.")
    parser.add_argument("--render-workers", type=int, default=None,
                         help="Параллельные процессы для рендера+метрик чекпоинта (этап 2) и "
                              "рендера baseline (этап 4) — browser-bound (каждый воркер держит "
                              "свой Chromium), в отличие от --num-workers (I/O-bound judge). "
                              "По умолчанию min(--num-workers, 8): на этапе 2 на сэмпл приходится "
                              "до 6 скриншотов (см. metrics.visual_eval_v3_multi), так что даже "
                              "умеренное число воркеров создаёт заметную нагрузку на браузер — "
                              "высокий --num-workers, разумный для judge, здесь легко приводит "
                              "к таймаутам рендера. Поднимайте явно, если CPU/RAM позволяют больше.")
    parser.add_argument("--render-timeout-ms", type=int, default=None,
                         help="Таймаут одного page.goto/page.screenshot в рендере, мс. По умолчанию "
                              "render.DEFAULT_RENDER_TIMEOUT_MS (переопределяется также через "
                              "переменную окружения D2C_RENDER_TIMEOUT_MS).")
    parser.add_argument("--render-retries", type=int, default=3,
                         help="Сколько ДОПОЛНИТЕЛЬНЫХ попыток делать на сбойном рендере (с "
                              "пересозданием браузера перед повтором) — 0 отключает повтор.")
    parser.add_argument("--clip-batch-size", type=int, default=256,
                         help="Размер под-батча для CLIP-инференса на общем GPU-сервере "
                              "(см. clip_server.py) — не путать с --batch-size (размер батча "
                              "датасета). CLIP-сервер стартует один раз в начале всего прогона "
                              "и живёт постоянно, деля GPU с текущей vLLM-моделью.")
    parser.add_argument("--n-examples-per-batch", type=int, default=5,
                         help="Сколько случайных сэмплов батча сохранить как пример "
                              "(html+png чекпоинта и baseline + ref) и как строку в results.csv.")
    parser.add_argument("--no-resume", action="store_true",
                         help="Игнорировать progress.json и начать с батча 0 "
                              "(старые examples/ и results.csv будут перезаписаны).")
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers должен быть >= 1 (всегда идёт через ProcessPoolExecutor).")
    if args.render_workers is None:
        args.render_workers = min(args.num_workers, 8)
    if args.render_workers < 1:
        parser.error("--render-workers должен быть >= 1.")
    return args


PROMPT = (
    "You are an expert front-end developer. Look at this webpage screenshot and "
    "write a SINGLE self-contained HTML file (inline <style>, no external CSS/JS/fonts, "
    "no network requests) that reproduces the layout, text, and colors as closely as "
    "possible. Use plain gray placeholder boxes instead of any real images. "
    "Output ONLY the raw HTML code, with no explanation and no markdown code fences."
)


def extract_html(text):
    if text is None:
        return None
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


def generate_html_batch(client, images, max_new_tokens: int, enable_thinking: bool,
                         concurrency: int, progress_label: str = None) -> list:
    """Генерирует HTML по батчу картинок через vLLM HTTP-сервер (client:
    VLLMClient). Возвращает список HTML-строк в том же порядке, что images;
    None на месте сэмпла, для которого запрос окончательно не удался (после
    внутреннего retry в VLLMClient.chat — см. vllm_client.py)."""
    extra_body = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    texts = client.chat_batch(
        images, PROMPT, max_tokens=max_new_tokens, temperature=0.0,
        max_concurrency=concurrency, extra_body=extra_body,
        progress_label=progress_label,
    )
    return [extract_html(t) for t in texts]


def process_sample_scoring(idx: int, ref_html: str, pred_html_checkpoint,
                            sample_dir_str: str, render_timeout_ms: int = None,
                            render_retries: int = 1) -> dict:
    """ЭТАП 2 (после генерации checkpoint, до загрузки baseline): рендер
    эталона + рендер и 5 официальных метрик чекпоинта. Выполняется, пока GPU
    свободен от vLLM — CLIP-клиент бьёт в постоянно живущий CLIP-сервер (см.
    clip_server.py), сам GPU в этот момент не занят никакой LLM.

    На сэмпл здесь до 6 скриншотов (ref + pred + OCR-free блоки для обоих,
    см. metrics.visual_eval_v3_multi) — заметно больше, чем в render_sample_
    baseline (один скриншот), поэтому таймауты рендера чувствительнее именно
    на этом этапе (см. --render-workers/--render-timeout-ms)."""
    from pathlib import Path as _Path
    from render import prepare_and_render, replace_images_with_placeholder
    from metrics import score_pair

    sample_dir = _Path(sample_dir_str)
    sample_dir.mkdir(exist_ok=True)
    row = {"idx": idx}

    # --- эталон ---
    ref_info = prepare_and_render(ref_html, str(sample_dir / "ref.html"), str(sample_dir / "ref.png"),
                                   timeout_ms=render_timeout_ms, retries=render_retries)
    row["ref_n_img_replaced"] = ref_info["n_images_replaced"]
    row["ref_render_ok"] = ref_info["render_ok"]
    row["ref_render_fail_reason"] = ref_info["render_fail_reason"]

    # --- чекпоинт: полные официальные метрики ---
    if pred_html_checkpoint is None:
        row["status"] = "generation_error"
    else:
        pred_ckpt_path = sample_dir / "pred_checkpoint.html"
        clean_html, n_replaced = replace_images_with_placeholder(pred_html_checkpoint)
        pred_ckpt_path.write_text(clean_html, encoding="utf-8")
        row["pred_n_img_replaced"] = n_replaced
        try:
            scores = score_pair(str(pred_ckpt_path), str(sample_dir / "ref.html"))
            row.update(scores)
            row["status"] = "scored"
            row["pred_render_ok"] = _Path(sample_dir / "pred_checkpoint.png").exists()
        except Exception as e:
            row["status"] = f"metric_error: {e}"

    return row


def render_sample_baseline(idx: int, pred_html_baseline, sample_dir_str: str,
                            render_timeout_ms: int = None, render_retries: int = 1) -> dict:
    """ЭТАП 3 (после генерации baseline): только рендер pred_baseline.png,
    никаких метрик, никакого CLIP — существует на диске исключительно чтобы
    судья мог на него посмотреть на этапе 4. Один скриншот на сэмпл (в
    отличие от process_sample_scoring, где их до 6), поэтому обычно не
    задача --render-workers/--render-timeout-ms, но использует те же
    параметры для единообразия."""
    from pathlib import Path as _Path
    from render import replace_images_with_placeholder, render_html_to_png

    sample_dir = _Path(sample_dir_str)
    sample_dir.mkdir(exist_ok=True)
    row = {"idx": idx}

    if pred_html_baseline is None:
        row["baseline_status"] = "generation_error"
    else:
        pred_base_path = sample_dir / "pred_baseline.html"
        clean_html, n_replaced = replace_images_with_placeholder(pred_html_baseline)
        pred_base_path.write_text(clean_html, encoding="utf-8")
        row["baseline_n_img_replaced"] = n_replaced
        pred_base_png = sample_dir / "pred_baseline.png"
        result = render_html_to_png(str(pred_base_path), str(pred_base_png), overwrite=True,
                                     timeout_ms=render_timeout_ms, retries=render_retries)
        row["baseline_render_ok"] = result.ok
        row["baseline_render_fail_reason"] = result.reason
        row["baseline_status"] = "rendered" if result.ok else "render_error"

    return row


def judge_sample(idx: int, sample_dir_str: str, judge_seed: int,
                  checkpoint_model_name: str, baseline_model_name: str,
                  checkpoint_scored: bool, baseline_rendered: bool) -> dict:
    """ЭТАП 4 (после того как judge поднят на GPU): pairwise-сравнение
    checkpoint vs baseline против эталона. Требует, чтобы этапы 2 и 3 УЖЕ
    записали ref.png/pred_checkpoint.png/pred_baseline.png на диск — сам этот
    этап их не создаёт, только читает.

    Всегда выполняется в воркере ProcessPoolExecutor (см. process_one_batch)
    — judge-клиент берётся из модульной переменной _judge_client,
    проставленной _worker_init_judge один раз на процесс.

    judge_seed: детерминированный per-sample seed для anti-position-bias
    монетки судьи (не общий rng, чтобы порядок обработки futures в
    as_completed не расходовал общий Random непредсказуемо).

    checkpoint_scored / baseline_rendered: статусы, посчитанные в главном
    процессе из результатов этапов 2/3 (а не пересчитанные здесь из файлов
    на диске) — так воркер этого этапа не обязан знать формат row из других
    этапов, только простой bool-флаг "есть ли смысл вообще звать судью"."""
    from pathlib import Path as _Path
    from judge_client import judge_one, JudgeError
    import random as _random
    import os as _os

    sample_dir = _Path(sample_dir_str)
    row = {"idx": idx}

    ref_png = sample_dir / "ref.png"
    pred_ckpt_png = sample_dir / "pred_checkpoint.png"
    pred_base_png = sample_dir / "pred_baseline.png"

    if checkpoint_scored and baseline_rendered \
            and ref_png.exists() and pred_ckpt_png.exists() and pred_base_png.exists():
        try:
            judge_max_tokens = int(_os.environ.get("D2C_JUDGE_MAX_TOKENS", "512"))
            result = judge_one(
                _judge_client, str(ref_png), str(pred_ckpt_png), str(pred_base_png),
                model_a_name=checkpoint_model_name, model_b_name=baseline_model_name,
                rng=_random.Random(judge_seed), max_tokens=judge_max_tokens,
            )
            row["judge_winner_model"] = result["winner_model"]
            row["judge_swapped"] = result["swapped"]
            row["judge_status"] = "scored"
        except Exception as e:
            # Печатаем полный repr(e) (не только str(e)) в стдаут воркера —
            # для requests.exceptions.ConnectionError (судья упал/недоступен)
            # str(e) часто короткий и малополезный, а repr показывает тип
            # исключения и urllib3-детали (Connection refused / Read timed
            # out / EOF и т.п.), которые важны для отличения "судья ещё не
            # готов" от "судья упал насмерть в середине батча" — раньше при
            # массовом падении судьи (см. RUNNING.md, инцидент с
            # EngineDeadError) в стдауте не было вообще ничего, только
            # финальный judge_status в progress.json постфактум.
            #
            # Traceback печатается максимум _JUDGE_ERROR_PRINT_LIMIT раз НА
            # ПРОЦЕСС (не на сэмпл) — при массовом крахе судьи один процесс
            # может обработать десятки сэмплов, все с одной и той же
            # ошибкой; печатать одинаковый traceback на каждый из них — чистый
            # спам. row["judge_status"] всё равно проставляется для КАЖДОГО
            # сэмпла независимо от печати, так что n_judge_errors в
            # progress.json остаётся точным.
            global _judge_error_print_count
            if _judge_error_print_count < _JUDGE_ERROR_PRINT_LIMIT:
                _judge_error_print_count += 1
                import traceback as _traceback
                print(f"[judge_sample] idx={idx}: judge_one упал: {e!r}\n"
                      f"{_traceback.format_exc()}")
            row["judge_status"] = f"judge_error: {e!r}"
    else:
        row["judge_status"] = "skipped_missing_render"

    return row


# =============================================================================
# Потоковая загрузка датасета батчами фиксированного размера
# =============================================================================

def iter_dataset_batches(hf_dataset: str, n_samples: int, batch_size: int, seed: int,
                          shuffle_buffer_size: int, skip: int = 0,
                          hf_config: str = "v0.2", hf_split: str = "train"):
    """Генератор: отдаёт список HF-сэмплов (dict с ключами 'text'/'image') по
    batch_size штук за раз, пока не наберётся n_samples суммарно.

    skip: сколько сэмплов уже обработано в предыдущих запусках (для resume)."""
    from datasets import load_dataset

    print(f"[run_benchmark] Открываю {hf_dataset} (config={hf_config}, split={hf_split}) "
          f"в streaming-режиме (skip={skip})...")
    ds_stream = load_dataset(hf_dataset, name=hf_config, split=hf_split, streaming=True)

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
# Точное взвешенное усреднение метрик чекпоинта по батчам + win/loss судьи
# =============================================================================

def count_errors_by_status(df: pd.DataFrame, n_expected: int) -> dict:
    """Единая логика подсчёта ошибок по df одного батча — используется и в
    process_one_batch (для progress.json), и при агрегации по батчам в
    main(), вместо дублирования одной и той же логики в двух местах."""
    n_generation_errors = int((df["status"] == "generation_error").sum()) if "status" in df else 0
    n_metric_errors = int(df["status"].str.startswith("metric_error", na=False).sum()) \
        if "status" in df else 0
    n_known_bad = n_generation_errors + n_metric_errors
    n_other_errors = int(len(df) - (df["status"] == "scored").sum() - n_known_bad) \
        if "status" in df else 0
    n_baseline_errors = int((df["baseline_status"] != "rendered").sum()) if "baseline_status" in df else n_expected
    n_judge_errors = int((df["judge_status"] != "scored").sum()) if "judge_status" in df else n_expected
    return {
        "n_generation_errors": n_generation_errors,
        "n_metric_errors": n_metric_errors,
        "n_other_errors": n_other_errors,
        "n_baseline_errors": n_baseline_errors,
        "n_judge_errors": n_judge_errors,
    }


class MetricAccumulator:
    """running sum/count на метрику ЧЕКПОИНТА (baseline не аккумулируется —
    для него не считается ни одна из METRIC_KEYS), плюс счётчики судьи
    (сколько раз выиграл чекпоинт / baseline). mean() =
    sum/count всегда равен честному среднему по всем виденным до сих пор
    оценённым сэмплам, независимо от разбиения на батчи."""

    def __init__(self):
        self.sums = {k: 0.0 for k in METRIC_KEYS}
        self.count = 0
        # winner схемы судьи — строго "checkpoint" или "baseline" (см.
        # JUDGE_JSON_SCHEMA в judge_client.py, enum без "tie"), так что
        # wins_checkpoint + wins_baseline == judge_count всегда.
        self.judge_wins_checkpoint = 0
        self.judge_wins_baseline = 0
        self.judge_count = 0

    def add_batch(self, df: "pd.DataFrame"):
        ok = df[df["status"] == "scored"]
        n = len(ok)
        if n > 0:
            for k in METRIC_KEYS:
                self.sums[k] += float(ok[k].sum())
            self.count += n

        if "judge_status" in df:
            judged = df[df["judge_status"] == "scored"]
            self.judge_count += len(judged)
            if len(judged) > 0 and "checkpoint_model_name" in df and "baseline_model_name" in df:
                ckpt_name = df["checkpoint_model_name"].iloc[0]
                base_name = df["baseline_model_name"].iloc[0]
                self.judge_wins_checkpoint += int((judged["judge_winner_model"] == ckpt_name).sum())
                self.judge_wins_baseline += int((judged["judge_winner_model"] == base_name).sum())

    def means(self) -> dict:
        if self.count == 0:
            out = {k: float("nan") for k in METRIC_KEYS}
        else:
            out = {k: self.sums[k] / self.count for k in METRIC_KEYS}
        if self.judge_count == 0:
            out["judge_winrate_checkpoint"] = float("nan")
            out["judge_winrate_baseline"] = float("nan")
        else:
            out["judge_winrate_checkpoint"] = self.judge_wins_checkpoint / self.judge_count
            out["judge_winrate_baseline"] = self.judge_wins_baseline / self.judge_count
        return out

    def to_state(self) -> dict:
        return {
            "sums": self.sums, "count": self.count,
            "judge_wins_checkpoint": self.judge_wins_checkpoint,
            "judge_wins_baseline": self.judge_wins_baseline,
            "judge_count": self.judge_count,
        }

    @classmethod
    def from_state(cls, state: dict) -> "MetricAccumulator":
        acc = cls()
        if state:
            acc.sums = {k: float(state["sums"].get(k, 0.0)) for k in METRIC_KEYS}
            acc.count = int(state["count"])
            acc.judge_wins_checkpoint = int(state.get("judge_wins_checkpoint", 0))
            acc.judge_wins_baseline = int(state.get("judge_wins_baseline", 0))
            acc.judge_count = int(state.get("judge_count", 0))
        return acc


# =============================================================================
# Прогресс / чекпоинт для resume
# =============================================================================

def load_progress(progress_path: Path) -> dict:
    if progress_path.exists():
        return json.loads(progress_path.read_text(encoding="utf-8"))
    return {"next_batch_idx": 0, "dataset_offset": 0,
            "n_generation_errors": 0, "n_metric_errors": 0, "n_other_errors": 0,
            "n_baseline_errors": 0, "n_judge_errors": 0,
            "accumulator": None}


def save_progress(progress_path: Path, progress: dict):
    tmp_path = progress_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, progress_path)


def log_time(outdir: Path, stage_name, duration_seconds):
    with open(outdir / "time_results.txt", "a", encoding="utf-8") as f:
        f.write(f"{stage_name}: {duration_seconds:.6f} сек.\n")


# =============================================================================
# Обработка одного батча: генерация (checkpoint + baseline) -> рендер+метрики+judge
# -> примеры -> очистка
# =============================================================================

def process_one_batch(server_manager, batch_samples, batch_idx: int,
                       work_root: Path, examples_root: Path, args, rng: random.Random,
                       clip_request_queue=None):
    """Возвращает (examples_df, df, batch_error_counts, timing). work_root: временный
    каталог этого батча — целиком удаляется в конце, кроме файлов,
    скопированных в examples_root.

    server_manager (VLLMServerManager, один и тот же объект переиспользуется
    между батчами) переключает модель на GPU последовательно — на ней
    физически может быть поднята только одна модель за раз. Но этапы,
    которые НЕ делят GPU-VRAM модели (browser/CPU-рендер, деплой следующей
    модели), выполняются параллельно в отдельных потоках оркестрации
    (ThreadPoolExecutor(max_workers=2) вокруг каждой пары):

        1. switch_to(checkpoint) -> generate_html_batch (чекпоинт)
        2. || рендер эталона + рендер/метрики чекпоинта (browser-bound,
           --render-workers, CLIP как отдельный постоянный GPU-процесс)
           || switch_to(baseline) -> generate_html_batch (baseline, vLLM VRAM)
           — не пересекаются по ресурсу, идут одновременно в двух потоках
        4. || рендер baseline (browser-bound, --render-workers)
           || switch_to(judge) (только загрузка весов — деплой)
           — сам judge-инференс ждёт PNG с диска, поэтому не может начаться
           раньше конца рендера, но деплой (VRAM) от рендера не зависит
        5. judge: pairwise-сравнение (ProcessPoolExecutor, --num-workers —
           чисто I/O-bound HTTP на тот же порт, модель уже поднята шагом выше)

    Время каждого этапа печатается сразу по его завершении (не одним блоком
    в конце батча) и разбито на deploy_sec (switch_to — загрузка весов) и
    work_sec (сама генерация/рендер/judge), чтобы длительный деплой большого
    чекпоинта не терялся внутри общего "generation_sec"."""
    batch_dir = work_root / f"batch_{batch_idx:05d}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    n = len(batch_samples)
    images = [batch_samples[i]["image"].convert("RGB") for i in range(n)]

    timing = {}

    def _log_stage(label: str, deploy_sec: float, work_sec: float):
        timing[f"{label}_deploy_sec"] = deploy_sec
        timing[f"{label}_work_sec"] = work_sec
        total = deploy_sec + work_sec
        print(f"[batch {batch_idx}] {label}: {total:.1f}с всего "
              f"(деплой модели {deploy_sec:.1f}с + работа {work_sec:.1f}с)")

    # =====================================================================
    # ЭТАП 1: checkpoint — генерация HTML
    # =====================================================================
    _t0 = time.perf_counter()
    server_manager.switch_to(args.model_checkpoint, label="checkpoint",
                              log_level=os.environ.get("D2C_CHECKPOINT_LOG_LEVEL", "INFO"))
    _deploy_sec = time.perf_counter() - _t0
    checkpoint_client = VLLMClient(server_manager.base_url, args.model_checkpoint)  # switch_to уже дождался /health
    print(f"[batch {batch_idx}] Генерирую HTML для {n} сэмплов: чекпоинт ({args.model_checkpoint})...")
    _t1 = time.perf_counter()
    try:
        pred_html_checkpoint = generate_html_batch(
            checkpoint_client, images, args.max_new_tokens, args.enable_thinking,
            concurrency=args.generation_concurrency,
            progress_label=f"batch {batch_idx} checkpoint",
        )
    except Exception as e:
        print(f"[batch {batch_idx}] Ошибка батчевой генерации (чекпоинт): {e}")
        pred_html_checkpoint = [None] * n
    _log_stage("checkpoint_generation", _deploy_sec, time.perf_counter() - _t1)

    # =====================================================================
    # ЭТАП 2 || ЭТАП 3: рендер+метрики checkpoint ПАРАЛЛЕЛЬНО с деплоем и
    # генерацией baseline.
    #
    # Ресурсы не пересекаются: этап 2 — CPU/browser (свой ProcessPoolExecutor,
    # свои воркер-процессы с собственным браузером-синглтоном, см. render.py)
    # + CLIP через отдельный постоянный GPU-процесс; этап 3 — vLLM VRAM
    # (baseline) в ГЛАВНОМ процессе. GPU в этот момент делят CLIP (уже
    # рассчитан на сосуществование с vLLM, см. --gpu-memory-utilization) и
    # baseline — они физически разные процессы/выделения VRAM, конфликта по
    # устройству нет. Единственная общая зависимость — оба читают из
    # batch_samples/pred_html_checkpoint, которые уже готовы к этому моменту;
    # этап 3 не читает НИЧЕГО из результатов этапа 2, так что зависимости
    # по данным нет вообще, только независимая работа на разных ресурсах.
    #
    # Синхронизация: этап 2 идёт в отдельном потоке (ThreadPoolExecutor на
    # оркестрацию, не путать с ProcessPoolExecutor внутри самого этапа) —
    # GIL это не проблема, поток почти всё время блокируется на
    # ProcessPoolExecutor.submit/as_completed или на HTTP-ожидании, не на
    # CPU-работе самого потока. Этап 3 выполняется в главном потоке как
    # раньше. process_one_batch ждёт оба перед этапом 4.
    # =====================================================================
    def _run_stage2_scoring():
        _t0 = time.perf_counter()
        rows_by_idx = {}
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = mp.get_context("spawn")
        print(f"[batch {batch_idx}] Эталон+метрики чекпоинта: {n} сэмплов на "
              f"{args.render_workers} browser-воркерах (CLIP через общий сервер)...")
        with ProcessPoolExecutor(max_workers=args.render_workers, mp_context=ctx,
                                  initializer=_worker_init_scoring,
                                  initargs=(clip_request_queue,)) as executor:
            futures = {}
            for i in range(n):
                fut = executor.submit(
                    process_sample_scoring, i, batch_samples[i]["text"],
                    pred_html_checkpoint[i], str(batch_dir / f"sample_{i:05d}"),
                    args.render_timeout_ms, args.render_retries)
                futures[fut] = i
            for future in tqdm(as_completed(futures), total=len(futures),
                                desc=f"[batch {batch_idx}] Эталон + метрики чекпоинта"):
                i = futures[future]
                try:
                    rows_by_idx[i] = future.result()
                except Exception as e:
                    rows_by_idx[i] = {"idx": i, "status": f"worker_error: {e}"}
        _work_sec = time.perf_counter() - _t0
        n_render_timeouts = sum(
            1 for r in rows_by_idx.values() if r.get("ref_render_fail_reason") == "timeout"
        )
        print(f"[batch {batch_idx}] Эталон+метрики чекпоинта: {_work_sec:.1f}с "
              f"({n_render_timeouts} таймаутов рендера эталона)")
        return rows_by_idx, _work_sec

    def _run_stage3_baseline_generation():
        _t0 = time.perf_counter()
        server_manager.switch_to(args.model_baseline, label="baseline",
                                  log_level=os.environ.get("D2C_BASELINE_LOG_LEVEL", "INFO"))
        _deploy_sec = time.perf_counter() - _t0
        client = VLLMClient(server_manager.base_url, args.model_baseline)  # switch_to уже дождался /health
        print(f"[batch {batch_idx}] Генерирую HTML для {n} сэмплов: baseline ({args.model_baseline})...")
        _t1 = time.perf_counter()
        try:
            html = generate_html_batch(
                client, images, args.max_new_tokens, args.enable_thinking,
                concurrency=args.generation_concurrency,
                progress_label=f"batch {batch_idx} baseline",
            )
        except Exception as e:
            print(f"[batch {batch_idx}] Ошибка батчевой генерации (baseline): {e}")
            html = [None] * n
        return html, _deploy_sec, time.perf_counter() - _t1

    from render import close_browser
    from concurrent.futures import ThreadPoolExecutor
    close_browser()  # закрываем браузер главного процесса до форка воркеров этапа 2

    with ThreadPoolExecutor(max_workers=2) as stage_pool:
        fut_stage2 = stage_pool.submit(_run_stage2_scoring)
        fut_stage3 = stage_pool.submit(_run_stage3_baseline_generation)
        scoring_rows_by_idx, checkpoint_scoring_sec = fut_stage2.result()
        pred_html_baseline, baseline_deploy_sec, baseline_work_sec = fut_stage3.result()

    del images
    timing["checkpoint_scoring_sec"] = checkpoint_scoring_sec
    _log_stage("baseline_generation", baseline_deploy_sec, baseline_work_sec)

    # =====================================================================
    # ЭТАП 4 || деплой judge: рендер baseline ПАРАЛЛЕЛЬНО с загрузкой весов
    # судьи (только switch_to — сам judge-инференс читает pred_baseline.png
    # с диска, поэтому не может начаться раньше, чем рендер этапа 4
    # завершится; но деплой (загрузка весов в VRAM) от этого рендера не
    # зависит, поэтому его можно начинать сразу).
    # =====================================================================
    def _run_stage4_baseline_render():
        _t0 = time.perf_counter()
        rows_by_idx = {}
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = mp.get_context("spawn")
        print(f"[batch {batch_idx}] Рендер baseline: {n} сэмплов на {args.render_workers} browser-воркерах...")
        with ProcessPoolExecutor(max_workers=args.render_workers, mp_context=ctx) as executor:
            futures = {}
            for i in range(n):
                fut = executor.submit(render_sample_baseline, i, pred_html_baseline[i],
                                       str(batch_dir / f"sample_{i:05d}"),
                                       args.render_timeout_ms, args.render_retries)
                futures[fut] = i
            for future in tqdm(as_completed(futures), total=len(futures),
                                desc=f"[batch {batch_idx}] Рендер baseline"):
                i = futures[future]
                try:
                    rows_by_idx[i] = future.result()
                except Exception as e:
                    rows_by_idx[i] = {"idx": i, "baseline_status": f"worker_error: {e}"}
        _work_sec = time.perf_counter() - _t0
        print(f"[batch {batch_idx}] Рендер baseline: {_work_sec:.1f}с")
        return rows_by_idx, _work_sec

    def _run_judge_deploy():
        _t0 = time.perf_counter()
        server_manager.switch_to(args.judge_model, label="judge",
                                  log_level=os.environ.get("D2C_JUDGE_LOG_LEVEL", "DEBUG"))
        return time.perf_counter() - _t0

    close_browser()  # снова закрываем браузер главного процесса до форка воркеров этапа 4

    with ThreadPoolExecutor(max_workers=2) as stage_pool:
        fut_stage4 = stage_pool.submit(_run_stage4_baseline_render)
        fut_judge_deploy = stage_pool.submit(_run_judge_deploy)
        baseline_rows_by_idx, baseline_render_sec = fut_stage4.result()
        judge_deploy_sec = fut_judge_deploy.result()

    timing["baseline_render_sec"] = baseline_render_sec

    # =====================================================================
    # ЭТАП 5: judge (модель уже поднята — деплой пришёлся на этап 4 выше)
    # =====================================================================
    judge_url = server_manager.base_url
    os.environ["D2C_JUDGE_URL"] = judge_url
    os.environ["D2C_JUDGE_MODEL"] = args.judge_model
    os.environ["D2C_JUDGE_MAX_TOKENS"] = str(args.judge_max_new_tokens)

    judge_rows_by_idx = {}
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ctx = mp.get_context("spawn")
    # Judge — чисто I/O-bound HTTP-запросы к vLLM (не локальная работа на
    # процессе): --num-workers здесь только верхний предел пула, реальную
    # нагрузку создаёт vLLM continuous batching на сервере, не число
    # процессов клиента.
    print(f"[batch {batch_idx}] Judge: {n} сэмплов, пул до {args.num_workers} "
          f"процессов (I/O-bound HTTP на {judge_url})...")
    _t1 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=ctx,
                              initializer=_worker_init_judge,
                              initargs=(judge_url, args.judge_model)) as executor:
        futures = {}
        for i in range(n):
            judge_seed = hash((args.seed, batch_idx, i)) & 0xFFFFFFFF
            fut = executor.submit(
                judge_sample, i, str(batch_dir / f"sample_{i:05d}"), judge_seed,
                args.model_checkpoint, args.model_baseline,
                scoring_rows_by_idx.get(i, {}).get("status") == "scored",
                baseline_rows_by_idx.get(i, {}).get("baseline_status") == "rendered",
            )
            futures[fut] = i
        for future in tqdm(as_completed(futures), total=len(futures),
                            desc=f"[batch {batch_idx}] Judge"):
            i = futures[future]
            try:
                judge_rows_by_idx[i] = future.result()
            except Exception as e:
                judge_rows_by_idx[i] = {"idx": i, "judge_status": f"worker_error: {e}"}
    _log_stage("judge", judge_deploy_sec, time.perf_counter() - _t1)

    # --- склеиваем три набора построчных результатов в один df ---
    rows = []
    for i in range(n):
        row = {}
        row.update(scoring_rows_by_idx.get(i, {"idx": i, "status": "missing"}))
        row.update(baseline_rows_by_idx.get(i, {"baseline_status": "missing"}))
        row.update(judge_rows_by_idx.get(i, {"judge_status": "missing"}))
        row["idx"] = i
        row["checkpoint_model_name"] = args.model_checkpoint
        row["baseline_model_name"] = args.model_baseline
        rows.append(row)

    df = pd.DataFrame(rows)
    _gen_duration = (timing["checkpoint_generation_deploy_sec"] + timing["checkpoint_generation_work_sec"]
                     + timing["baseline_generation_deploy_sec"] + timing["baseline_generation_work_sec"])
    _render_duration = (timing["checkpoint_scoring_sec"] + timing["baseline_render_sec"]
                         + timing["judge_deploy_sec"] + timing["judge_work_sec"])

    batch_error_counts = count_errors_by_status(df, n)

    # --- выбираем случайные примеры ДО удаления файлов батча ---
    # "хороший" пример — там, где чекпоинт оценён И judge реально ответил
    # (иначе pred_baseline.png мог не сохраниться до удаления батча зря).
    scored_idx = df.index[(df["status"] == "scored") & (df["judge_status"] == "scored")].tolist()
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
        for fname in ("pred_checkpoint.html", "pred_checkpoint.png",
                      "pred_baseline.html", "pred_baseline.png", "ref.html", "ref.png"):
            src = sample_dir / fname
            if src.exists():
                shutil.copy2(src, dest_dir / fname)
        row = df.loc[i].to_dict()
        row["batch_idx"] = batch_idx
        row["example_dir"] = str(dest_dir)
        example_rows.append(row)

    examples_df = pd.DataFrame(example_rows)

    # Сводные суммы поверх подробного per-stage timing (см. выше) — под
    # именами generation_sec/render_metrics_judge_sec, которые читает
    # log_time() ниже; подробная разбивка по этапам и по deploy/work
    # доступна в том же словаре timing.
    timing["generation_sec"] = _gen_duration
    timing["render_metrics_judge_sec"] = _render_duration

    # --- аккумулируем метрики чекпоинта + judge по ВСЕМ сэмплам батча ---
    # (не только examples — accumulator.add_batch сам фильтрует по status/judge_status)

    # --- полная очистка временных файлов батча ---
    shutil.rmtree(batch_dir, ignore_errors=True)

    return examples_df, df, batch_error_counts, timing


def append_examples_csv(results_path: Path, examples_df: pd.DataFrame):
    if examples_df.empty:
        return
    write_header = not results_path.exists()
    examples_df.to_csv(results_path, mode="a", header=write_header, index=False)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    work_root = outdir / "_work"
    examples_root = outdir / "examples"
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
    n_baseline_errors = progress.get("n_baseline_errors", 0)
    n_judge_errors = progress.get("n_judge_errors", 0)

    if start_batch_idx > 0:
        print(f"[run_benchmark] Резюмирую с батча {start_batch_idx} "
              f"(уже обработано сэмплов: {dataset_offset}, "
              f"накоплено метрик чекпоинта: {accumulator.count}, "
              f"judge count: {accumulator.judge_count}).")

    # --- 1-GPU-версия: ОДИН VLLMServerManager, модели переключаются по ходу
    # process_one_batch (checkpoint -> baseline -> judge на каждом батче).
    # Никакой предварительный старт серверов тут не нужен — первый switch_to
    # произойдёт внутри process_one_batch на первом батче.
    server_manager = VLLMServerManager(
        port=args.vllm_port,
        gpu_index=args.gpu_index,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.max_model_len,
        log_dir=str(outdir / "server_logs"),
    )

    # D2C_JUDGE_MAX_TOKENS читается в judge_sample независимо от --num-workers;
    # D2C_JUDGE_URL/D2C_JUDGE_MODEL для --num-workers<=1 выставляются заново на
    # каждом батче внутри process_one_batch (после того как judge реально
    # поднят) — здесь достаточно проставить то, что не меняется между батчами.
    os.environ["D2C_JUDGE_MAX_TOKENS"] = str(args.judge_max_new_tokens)

    # CLIP-сервер — стартует ОДИН РАЗ в начале всего прогона (не на батч) и
    # живёт до конца, деля GPU с текущей vLLM-моделью (см. docstring вверху
    # файла и vllm_server_manager.py про --vllm-gpu-memory-utilization).
    # Используется только для метрик чекпоинта (baseline никогда не
    # обращается к CLIP, судья тоже). Всегда нужен — этап 2 всегда идёт через
    # ProcessPoolExecutor (см. process_one_batch), inline-режима без пула нет.
    from clip_server import ClipServer
    clip_server = ClipServer(clip_batch_size=args.clip_batch_size, gpu_index=args.gpu_index)
    print(f"[run_benchmark] Поднимаю CLIP-сервер на GPU {args.gpu_index} "
          f"(живёт постоянно, до конца прогона)...")
    clip_server.start()
    clip_request_queue = clip_server.request_queue

    rng = random.Random(args.seed + 1)  # для выбора examples — отдельно от judge-монетки и shuffle датасета

    # Между-батчевая финализация (запись examples_df в results.csv,
    # progress.json, time_results.txt) — чистый диск-I/O, не читает и не
    # пишет ничего, что нужно следующему батчу (accumulator/счётчики
    # ошибок обновляются СИНХРОННО, до постановки в очередь — см. ниже,
    # именно они определяют resume-состояние в памяти). Поэтому саму
    # запись на диск можно делать в фоновом потоке, пока следующий батч
    # уже начал этап 1 (switch_to(checkpoint) + генерация) — GPU/vLLM не
    # ждёт, пока допишутся файлы предыдущего батча.
    #
    # Пул из ОДНОГО потока и ручной wait перед следующей постановкой в
    # очередь — не для параллельности внутри финализации (там просто
    # несколько последовательных дисковых операций), а чтобы гарантировать
    # строгий порядок записи батчей (N, затем N+1, ...) и чтобы обработка
    # следующего батча не могла обогнать запись двух батчей назад, если
    # диск медленный — это создало бы неограниченно растущую очередь
    # фоновых задач без обратного давления.
    from concurrent.futures import ThreadPoolExecutor as _TPE
    finalize_pool = _TPE(max_workers=1)
    pending_finalize = None

    def _finalize_batch_on_disk(examples_df, progress, batch_label, duration,
                                 checkpoint_gen_sec, checkpoint_scoring_sec,
                                 baseline_gen_sec, baseline_render_sec, judge_sec):
        append_examples_csv(results_path, examples_df)
        save_progress(progress_path, progress)
        log_time(outdir, batch_label, duration)
        log_time(outdir, f"{batch_label} — генерация checkpoint", checkpoint_gen_sec)
        log_time(outdir, f"{batch_label} — эталон+метрики checkpoint", checkpoint_scoring_sec)
        log_time(outdir, f"{batch_label} — генерация baseline", baseline_gen_sec)
        log_time(outdir, f"{batch_label} — рендер baseline", baseline_render_sec)
        log_time(outdir, f"{batch_label} — judge", judge_sec)

    try:
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

            examples_df, df, batch_err, timing = process_one_batch(
                server_manager, batch_samples, batch_idx,
                work_root, examples_root, args, rng,
                clip_request_queue=clip_request_queue,
            )

            # --- синхронно: только in-memory состояние, нужное resume ---
            accumulator.add_batch(df)
            n_generation_errors += batch_err["n_generation_errors"]
            n_metric_errors += batch_err["n_metric_errors"]
            n_other_errors += batch_err["n_other_errors"]
            n_baseline_errors += batch_err["n_baseline_errors"]
            n_judge_errors += batch_err["n_judge_errors"]
            dataset_offset += len(batch_samples)
            batch_idx += 1

            progress = {
                "next_batch_idx": batch_idx,
                "dataset_offset": dataset_offset,
                "n_generation_errors": n_generation_errors,
                "n_metric_errors": n_metric_errors,
                "n_other_errors": n_other_errors,
                "n_baseline_errors": n_baseline_errors,
                "n_judge_errors": n_judge_errors,
                "accumulator": accumulator.to_state(),
            }
            duration = time.perf_counter() - t0
            means_so_far = accumulator.means()
            print(f"[batch {batch_idx - 1}] {duration:.1f} сек. Накопленное среднее чекпоинта "
                  f"({accumulator.count} оценённых сэмплов): "
                  + ", ".join(f"{k}={means_so_far[k]:.4f}" for k in METRIC_KEYS))
            if accumulator.judge_count > 0:
                print(f"[batch {batch_idx - 1}] Judge (n={accumulator.judge_count}): "
                      f"winrate чекпоинта={means_so_far['judge_winrate_checkpoint']:.4f}, "
                      f"winrate baseline={means_so_far['judge_winrate_baseline']:.4f}")

            # --- дождаться записи батча N-2, прежде чем ставить в очередь N-1
            # (обратное давление — см. docstring finalize_pool выше) ---
            if pending_finalize is not None:
                pending_finalize.result()
            pending_finalize = finalize_pool.submit(
                _finalize_batch_on_disk, examples_df, progress,
                f"Батч {batch_idx - 1}", duration,
                timing["checkpoint_generation_deploy_sec"] + timing["checkpoint_generation_work_sec"],
                timing["checkpoint_scoring_sec"],
                timing["baseline_generation_deploy_sec"] + timing["baseline_generation_work_sec"],
                timing["baseline_render_sec"],
                timing["judge_deploy_sec"] + timing["judge_work_sec"],
            )
            # process_one_batch следующей итерации стартует здесь, пока
            # запись батча {batch_idx - 1} на диск ещё может идти в фоне.

        if pending_finalize is not None:
            pending_finalize.result()
        finalize_pool.shutdown(wait=True)

        from render import close_browser
        close_browser()
    finally:
        # Порядок важен: сначала гасим текущий vLLM (может быть ещё поднят,
        # если прогон упал/прервался посреди батча), потом CLIP — так VRAM
        # освобождается в предсказуемом порядке и не остаётся висящих
        # процессов ни от того, ни от другого при Ctrl+C/исключении.
        # finalize_pool.shutdown(wait=False) здесь на случай, если цикл упал
        # ДО штатного shutdown(wait=True) выше — не блокируем аварийное
        # завершение ожиданием фоновой записи, но и не оставляем поток
        # висеть незакрытым.
        finalize_pool.shutdown(wait=False)
        print("[run_benchmark] Останавливаю vLLM-сервер...")
        server_manager.stop()
        if clip_server is not None:
            print("[run_benchmark] Останавливаю CLIP-сервер...")
            clip_server.stop()

    final_means = accumulator.means()
    n_excluded_total = n_generation_errors + n_metric_errors + n_other_errors
    assert dataset_offset == accumulator.count + n_excluded_total, (
        f"Расхождение в подсчёте: обработано {dataset_offset}, но "
        f"scored({accumulator.count}) + excluded({n_excluded_total}) != обработано. "
        f"Значит process_sample_scoring вернул статус чекпоинта, который нигде не учтён."
    )

    summary = {
        "n_samples_requested": args.n_samples,
        "n_samples_processed": dataset_offset,
        "model_checkpoint": args.model_checkpoint,
        "model_baseline": args.model_baseline,
        "judge_model": args.judge_model,
        "n_scored": accumulator.count,
        "n_excluded_total": n_excluded_total,
        "n_generation_errors": n_generation_errors,
        "n_metric_errors": n_metric_errors,
        "n_other_errors": n_other_errors,
        "n_baseline_errors": n_baseline_errors,
        "judge_n_scored": accumulator.judge_count,
        "n_judge_errors": n_judge_errors,
        **final_means,
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nЧекпоинт: {args.model_checkpoint}")
    print(f"Baseline (только для judge): {args.model_baseline}")
    print(f"Судья: {args.judge_model}")
    print(f"Обработано сэмплов: {dataset_offset} / {args.n_samples}")
    print(f"Успешно оценено (чекпоинт, 5 метрик): {accumulator.count}")
    print(f"Исключено из среднего (чекпоинт): {n_excluded_total} "
          f"(generation_error: {n_generation_errors}, metric_error: {n_metric_errors}, "
          f"прочее: {n_other_errors})")
    print("Итоговые средние по чекпоинту (взвешенно, не среднее средних батчей):")
    for k in METRIC_KEYS:
        v = final_means[k]
        print(f"  {k}: {v:.4f}" if not math.isnan(v) else f"  {k}: nan")
    if accumulator.judge_count > 0:
        print(f"\nJudge (n={accumulator.judge_count}, ошибок судьи: {n_judge_errors}):")
        print(f"  winrate чекпоинта: {final_means['judge_winrate_checkpoint']:.4f}")
        print(f"  winrate baseline:  {final_means['judge_winrate_baseline']:.4f}")
    print(f"\nПримеры сохранены в: {examples_root}")
    print(f"Строки-примеры (results.csv): {results_path}")
    print(f"Итоговая сводка: {summary_path}")


if __name__ == "__main__":
    main()