"""Основной прогон бенчмарка Design2Code: батчи, resume, точные средние.

Рассчитан на десятки тысяч сэмплов, откуда и все решения ниже.

ПОБАТЧНОЕ ЧТЕНИЕ. Датасет идёт HF-стримингом чанками по --batch-size. Держать
все сэмплы в памяти нельзя: на 70k это 70k PIL-картинок разом плюс неограниченно
растущие PNG эталонов на диске ещё ДО генерации.

ФАЙЛЫ БАТЧА УДАЛЯЮТСЯ. После рендера и метрик каталог батча сносится целиком,
кроме --n-examples-per-batch случайных сэмплов: их pred/ref (html и png, плюс
pred_raw.html) копируются в examples/batch_XXXXX/. В results.csv попадают строки
только этих примеров — это иллюстрация «как выглядит генерация», а не источник
итоговой статистики.

СРЕДНЕЕ СЧИТАЕТСЯ ВЗВЕШЕННО, А НЕ КАК «СРЕДНЕЕ СРЕДНИХ БАТЧЕЙ». Батчи дают
разное число ОЦЕНЁННЫХ строк: последний батч почти всегда неполный, и в каждом
батче своё число сэмплов, упавших с generation_error/metric_error. При среднем
средних неполный батч получил бы тот же вес, что и полный, и итог сместился бы.
Поэтому MetricAccumulator копит sum и count по каждой метрике, а в конце делит
одно на другое — это ровно честное среднее по всем оценённым сэмплам, но без
хранения всех строк.

⚠ Упавшие сэмплы ИСКЛЮЧАЮТСЯ из среднего, а не считаются нулём. У модели,
которая падает чаще, знаменатель меньше, и её скор оценивается по её же лучшим
сэмплам. Сравнивая два прогона, смотреть n_scored, а не только final_score.

RESUME. Каждый батч фиксируется в {outdir}/progress.json: номер следующего
батча, offset в датасете и состояние аккумулятора. Перезапуск той же командой
пропускает уже посчитанные батчи и продолжает копить метрики с того же места,
поэтому итог после рестартов точный, а не приближённый. Начать заново —
--no-resume.

Общий код с run_benchmark.py (load_model, generate_html_batch, extract_html,
render_and_score_one, PROMPT) СКОПИРОВАН, а не импортирован: тот файл — цельная
main() под свой сценарий, и импорт из него потащил бы её целиком.
render.py и metrics.py используются как есть.

Запуск (и первый, и повторный после падения — команда одна):
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


# CLIP: один общий батчующий GPU-процесс вместо копии модели в каждом воркере.
# Раньше metrics.py грузил CLIP при импорте, а render_and_score_one
# выполнялся в ProcessPoolExecutor(--num-workers) — то есть до num_workers
# отдельных копий CLIP на GPU, каждая гоняющая encode_image с batch_size=1.
# Теперь одна копия модели живёт в отдельном процессе (clip_server.ClipServer,
# запускается один раз на весь прогон, не на батч), а CPU-воркеры шлют туда
# запросы через очередь и получают результат по приватному Pipe — сервер сам
# группирует прилетевшие запросы в под-батчи по --clip-batch-size и делает
# один forward на под-батч. См. подробный docstring в clip_server.py.
#
# _clip_request_queue передаётся в каждый воркер-процесс через initializer
# ProcessPoolExecutor (не как аргумент submit — Queue не сериализуется через
# pickle на каждый вызов, а вот один раз при старте процесса через initargs
# работает штатно для spawn-контекста).
def _worker_init(clip_request_queue):
    from clip_server import ClipClient
    import metrics
    metrics.set_clip_client(ClipClient(clip_request_queue))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3.5-9B на Design2Code, батчами по HF-streaming (для 10k-100k+ сэмплов).")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--n-samples", type=int, default=70_000,
                         help="Сколько сэмплов всего обработать (по всем батчам).")
    parser.add_argument("--batch-size", type=int, default=10_000,
                         help="Размер одного динамического батча из HF (генерация+рендер+метрики "
                              "прогоняются на нём целиком, потом файлы батча удаляются).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--outdir", default="./design2code_results")
    parser.add_argument("--hf-dataset", default="SALT-NLP/Design2Code")
    parser.add_argument("--shuffle-buffer-size", type=int, default=10_000,
                         help="НЕ ПРИМЕНЯЕТСЯ. Флаг оставлен, чтобы не ломать существующие "
                              "команды; ds.shuffle() не вызывается, выборка — префикс сплита "
                              "(см. iter_dataset_batches и docs/experiments/DIVERGENCES.md, K3).")
    parser.add_argument("--enable-thinking", action="store_true")
    # Параметры декодирования. Раньше не задавались вообще — vLLM брал свои
    # дефолты (temperature 1.0, top_p 1.0), то есть бенч мерил модель под
    # чистым сэмплингом. Для HTML это шум: см. комментарий в generate_html_batch.
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = greedy (по умолчанию). >0 включает сэмплинг.")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="1.0 = выключено; 1.05-1.1 лечит зацикливание.")
    parser.add_argument("--sampling-seed", type=int, default=0,
                        help="Seed сэмплинга (применяется только при temperature>0), "
                             "чтобы прогон был воспроизводим.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.89)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--num-workers", type=int, default=16,
                         help="Параллельные процессы для рендера+метрик внутри одного батча.")
    parser.add_argument("--clip-batch-size", type=int, default=256,
                         help="Размер под-батча для CLIP-инференса на общем GPU-сервере "
                              "(см. clip_server.py) — не путать с --batch-size (размер батча "
                              "датасета). Запросы от всех --num-workers процессов копятся здесь "
                              "до clip-batch-size ИЛИ короткого таймаута и считаются одним "
                              "forward-проходом.")
    parser.add_argument("--n-examples-per-batch", type=int, default=5,
                         help="Сколько случайных сэмплов батча сохранить как пример "
                              "(html+png) и как строку в results.csv.")
    parser.add_argument("--no-resume", action="store_true",
                         help="Игнорировать progress.json и начать с батча 0 "
                              "(старые examples/ и results.csv будут перезаписаны).")
    parser.add_argument("--hf-config", default="default",
                         help="имя конфига датасета (Design2Code-hf -> default).")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--min-pixels", type=int, default=262_144,
                         help="min_pixels процессора (как в SFT). 262144 = 256*32*32")
    parser.add_argument("--max-pixels", type=int, default=2_097_152,
                         help="max_pixels процессора (ДОЛЖЕН совпадать с обучением, иначе "
                              "чекпоинт меряется вне своего распределения — H1).")
    parser.add_argument("--materialize-dom", action="store_true",
                         help="Заменять pred.html на состояние DOM ПОСЛЕ выполнения JS. "
                              "Нужно для моделей, которые пишут React/Vue (напр. UI2Code^N): "
                              "метрика разбирает статический исходник и у таких страниц не "
                              "находит текстовых блоков вовсе. По умолчанию ВЫКЛЮЧЕНО, чтобы "
                              "ранее снятые числа остались сравнимыми.")
    parser.add_argument("--prompt-file", default=None,
                         help="файл с текстом промпта; заменяет встроенный PROMPT целиком. "
                              "Нужен для чужих моделей со своим форматом запроса "
                              "(напр. UI2Code^N: 'Please generate the corresponding html "
                              "code for the given UI screenshot.').")
    parser.add_argument("--train-task-id", default=None,
                         help="id ClearML-задачи обучения, из которой взят чекпоинт "
                              "(обычно подхватывается из clearml_task.json рядом с весами).")
    return parser.parse_args()


def load_model(model_id_or_path: str, tensor_parallel_size: int, gpu_memory_utilization: float,
               max_model_len: int, min_pixels: int = 262_144, max_pixels: int = 2_097_152):
    from vllm import LLM

    # Пиксель-бюджет процессора — ДОЛЖЕН совпадать с обучением (H1). Без него
    # vLLM берёт родное разрешение картинки, и чекпоинт меряется вне трейн-распределения.
    # НО: min_pixels/max_pixels — параметры Qwen2VL-процессора. У других семейств их
    # нет (у Glm4vImageProcessor из UI2Code^N бюджет задаётся через size.longest_edge),
    # и передача неизвестных kwargs роняет загрузку. Поэтому 0 = не передавать вовсе.
    mm_kwargs = {}
    if min_pixels > 0:
        mm_kwargs["min_pixels"] = min_pixels
    if max_pixels > 0:
        mm_kwargs["max_pixels"] = max_pixels
    print(f"[run_benchmark] Загружаю модель {model_id_or_path} через vLLM "
          f"(mm_processor_kwargs={mm_kwargs or 'не заданы, берутся из конфига модели'})...")
    return LLM(
        model=model_id_or_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
        **({"mm_processor_kwargs": mm_kwargs} if mm_kwargs else {}),
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


def generate_html_batch(llm, images, max_new_tokens: int, enable_thinking: bool,
                        temperature: float = 0.0, top_p: float = 1.0,
                        repetition_penalty: float = 1.0, seed=None) -> list[str]:
    from vllm import SamplingParams

    # ⚠ Дефолт vLLM у SamplingParams — temperature=1.0, top_p=1.0, seed=None,
    # то есть ЧИСТЫЙ сэмплинг из полного распределения. Раньше параметры не
    # задавались вовсе, и бенч молча работал именно так. Для страницы в ~8к
    # токенов это губительно: шанс сорваться копится по всем токенам, и выход
    # получается «всё или ничего» — либо почти эталон, либо обрыв посреди
    # <style> с пустым рендером и score ровно 0. Тем же объясняется разброс
    # между прогонами одной и той же модели. По умолчанию теперь greedy.
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
    )
    conversations = [
        [{"role": "user", "content": [{"type": "image_pil", "image_pil": image}, {"type": "text", "text": PROMPT}]}]
        for image in images
    ]
    outputs = llm.chat(
        conversations,
        sampling_params=sampling_params,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )
    # Вместе с HTML отдаём диагностику генерации. finish_reason различает две
    # СОВСЕМ разные поломки, которые по одному только score неотличимы:
    # "length" — упёрлись в max_new_tokens (метрики занижены незаслуженно),
    # "stop" при коротком выводе — модель сама выдала EOS раньше времени
    # (ровно так схлопывались чекпоинты жёсткого рецепта: 156-456 байт).
    # raw_len/clean_len — сколько текста выдала модель и сколько осталось после
    # extract_html. Расхождение в разы = ответ портится ДО метрики, и без этих
    # двух чисел такое не видно вовсе (ровно так молча терялось 60-90% вывода
    # UI2Code^N). Сам сырой текст кладём в info — он сохраняется в pred_raw.html
    # для тех сэмплов, что попадают в examples/.
    htmls, infos = [], []
    for output in outputs:
        o = output.outputs[0]
        clean = extract_html(o.text)
        htmls.append(clean)
        infos.append({
            "finish_reason": o.finish_reason,
            "pred_n_tokens": len(o.token_ids),
            "raw_len": len(o.text),
            "clean_len": len(clean),
            "_raw_text": o.text,
        })
    return htmls, infos


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


def render_and_score_one(idx: int, pred_html: str, sample_dir_str: str,
                          raw_text: str | None = None, materialize: bool = False) -> dict:
    """Идентично run_benchmark.py — воркер для ProcessPoolExecutor (или прямой
    вызов при num_workers<=1). Импорт render/metrics внутри функции (см.
    комментарий в оригинале про spawn + CUDA)."""
    from pathlib import Path as _Path
    from render import replace_images_with_placeholder
    from metrics import score_pair

    sample_dir = _Path(sample_dir_str)
    row = {"idx": idx}

    # Сырой ответ модели — ДО extract_html и до замены плейсхолдеров. Без него
    # порчу ответа конвейером не отследить: на диске остаётся только результат
    # обработки, и «модель выдала мусор» неотличимо от «мы его испортили».
    if raw_text is not None:
        try:
            (sample_dir / "pred_raw.html").write_text(raw_text, encoding="utf-8")
        except OSError as e:
            # Диагностика не должна ронять замер: если сырой ответ не записался
            # (нет места, права), сэмпл всё равно надо оценить. Но и молчать
            # нельзя — иначе pred_raw.html пропадёт незаметно, а именно его
            # отсутствие когда-то скрыло потерю 60-90% вывода UI2Code^N.
            row["raw_dump_error"] = f"{type(e).__name__}: {e}"

    if pred_html is None:
        row["status"] = "generation_error"
        return row

    # Только запись pred.html с плейсхолдерами — score_pair ожидает его на
    # диске. НЕ рендерим pred.png здесь: score_pair -> visual_eval_v3_multi
    # рендерит его сам (do_it_again=True) внутри одного общего render_many
    # вместе с обоими перекрашенными вариантами pred и (если нужно) ref —
    # см. metrics.visual_eval_v3_multi. Рендер здесь был бы избыточным
    # повторным скриншотом того же pred.html.
    pred_html_path = sample_dir / "pred.html"
    clean_html, n_replaced = replace_images_with_placeholder(pred_html)
    pred_html_path.write_text(clean_html, encoding="utf-8")
    row["pred_n_img_replaced"] = n_replaced

    # Страницы, которые рисует JavaScript (React/Vue/…): метрика разбирает
    # статический исходник и у них не находит ни одного текстового блока.
    # Здесь подменяем файл на состояние DOM после отрисовки — см. materialize_dom.
    if materialize:
        from render import materialize_dom
        info = materialize_dom(str(pred_html_path))
        row["materialized"] = info.get("materialized", False)
        row["len_after_materialize"] = info.get("len_after")
        if info.get("error"):
            row["materialize_error"] = info["error"]

    try:
        scores = score_pair(str(pred_html_path), str(sample_dir / "ref.html"))
        row.update(scores)
        row["status"] = "scored"
        row["pred_render_ok"] = _Path(sample_dir / "pred.png").exists()
    except Exception as e:
        row["status"] = f"metric_error: {e}"

    return row


def iter_dataset_batches(hf_dataset: str, n_samples: int, batch_size: int, seed: int,
                          shuffle_buffer_size: int, skip: int = 0,
                          hf_config: str = "default", hf_split: str = "train"):
    """Генератор: отдаёт список HF-сэмплов (dict с ключами 'text'/'image') по
    batch_size штук за раз, пока не наберётся n_samples суммарно.

    ⚠ Выборка — ПРЕФИКС сплита, а не случайные n_samples: `ds.shuffle()` здесь не
    вызывается, и аргументы `seed`/`shuffle_buffer_size` не используются вовсе.
    Все числа проекта сняты на детерминированном префиксе Design2Code. Включать
    шаффл нельзя не подумав: выборка сменится, и новые прогоны станут несравнимы
    со всеми прошлыми. Разбор — docs/experiments/DIVERGENCES.md, K3.
    """
    from datasets import load_dataset, load_from_disk

    # Локальный датасет (load_from_disk) — для sanity-тестов, где train-сет и
    # bench-сет ДОЛЖНЫ быть одними и теми же сэмплами. Колонки 'image'/'text',
    # порядок как есть, без shuffle.
    if os.path.isdir(hf_dataset) and os.path.exists(os.path.join(hf_dataset, "dataset_info.json")):
        print(f"[run_benchmark] Локальный датасет {hf_dataset} (без shuffle)...")
        it = iter(load_from_disk(hf_dataset))
        for _ in range(skip):
            next(it, None)
        remaining = n_samples - skip
        while remaining > 0:
            batch = [x for x in (next(it, None) for _ in range(min(batch_size, remaining))) if x is not None]
            if not batch:
                return
            yield batch
            remaining -= len(batch)
        return

    print(f"[run_benchmark] Открываю {hf_dataset} (config={hf_config}) в streaming-режиме (skip={skip})...")
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


def load_progress(progress_path: Path) -> dict:
    if progress_path.exists():
        return json.loads(progress_path.read_text(encoding="utf-8"))
    return {"next_batch_idx": 0, "dataset_offset": 0, "n_generation_errors": 0,
            "n_metric_errors": 0, "n_other_errors": 0, "accumulator": None}


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


def process_one_batch(llm, batch_samples, batch_idx: int, work_root: Path, examples_root: Path,
                       args, rng: random.Random, clip_request_queue=None):
    """Возвращает (examples_df, ok_df, n_generation_errors, n_metric_errors, n_other_errors).
    work_root: временный каталог этого батча (html/png сэмплов) - целиком
    удаляется в конце функции, кроме файлов, скопированных в examples_root.
    clip_request_queue: очередь общего CLIP-сервера (см. clip_server.py) — если
    задана, воркеры рендера+метрик передают её в _worker_init и НЕ грузят
    CLIP локально (см. --num-workers ветку ниже). None только если
    --num-workers<=1 — тогда рендер+метрики (в т.ч. CLIP) считаются прямо в
    этом процессе без пула, clip_server не запускается вовсе.
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
        pred_html_list, gen_infos = generate_html_batch(
            llm, images, args.max_new_tokens, args.enable_thinking,
            temperature=args.temperature, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            seed=(args.sampling_seed if args.temperature > 0 else None),
        )
    except Exception as e:
        print(f"[batch {batch_idx}] Ошибка батчевой генерации: {e}")
        pred_html_list = [None] * n
        gen_infos = [{} for _ in range(n)]
    del images

    # --- рендер + метрики ---
    if args.num_workers <= 1:
        rows = []
        for i in tqdm(range(n), desc=f"[batch {batch_idx}] Рендер + метрики"):
            sample_dir = batch_dir / f"sample_{i:05d}"
            info = dict(gen_infos[i])
            raw = info.pop("_raw_text", None)   # в CSV сырой текст не кладём
            row = render_and_score_one(i, pred_html_list[i], str(sample_dir),
                                       raw_text=raw, materialize=args.materialize_dom)
            row["ref_n_img_replaced"] = ref_infos[i]["n_images_replaced"]
            row["ref_render_ok"] = ref_infos[i]["render_ok"]
            row.update(info)
            rows.append(row)
    else:
        from render import close_browser
        close_browser()  # см. run_benchmark.py: браузер главного процесса не нужен воркерам

        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        ctx = mp.get_context("spawn")
        rows_by_idx = {}
        print(f"[batch {batch_idx}] Рендер+метрики на {args.num_workers} процессах "
              f"(CLIP через общий сервер, clip-batch-size={args.clip_batch_size})...")
        # initializer=_worker_init передаёт клиента CLIP-сервера каждому
        # процессу пула ОДИН РАЗ при его старте (не на каждый submit) — сам
        # процесс воркера переиспользуется между батчами датасета (пул
        # создаётся заново на каждый process_one_batch, но это дёшево:
        # тяжёлая часть, GPU-модель, остаётся в clip_server и не
        # пересоздаётся вместе с пулом).
        with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=ctx,
                                  initializer=_worker_init, initargs=(clip_request_queue,)) as executor:
            futures = {
                executor.submit(render_and_score_one, i, pred_html_list[i],
                                 str(batch_dir / f"sample_{i:05d}"),
                                 gen_infos[i].get("_raw_text"),
                                 args.materialize_dom): i
                for i in range(n)
            }
            for future in tqdm(as_completed(futures), total=len(futures),
                                desc=f"[batch {batch_idx}] Рендер + метрики"):
                i = futures[future]
                try:
                    row = future.result()
                except Exception as e:
                    row = {"idx": i, "status": f"worker_error: {e}"}
                row["ref_n_img_replaced"] = ref_infos[i]["n_images_replaced"]
                row["ref_render_ok"] = ref_infos[i]["render_ok"]
                info = dict(gen_infos[i])
                info.pop("_raw_text", None)   # в CSV сырой текст не кладём
                row.update(info)
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
        for fname in ("pred.html", "pred_raw.html", "pred.png", "ref.html", "ref.png"):
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

    return examples_df, ok_df, n_generation_errors, n_metric_errors, n_other_errors


def append_examples_csv(results_path: Path, examples_df: pd.DataFrame):
    if examples_df.empty:
        return
    write_header = not results_path.exists()
    examples_df.to_csv(results_path, mode="a", header=write_header, index=False)


def main():
    args = parse_args()

    # Промпт из файла заменяет встроенный: чужие модели обучены на своей формулировке,
    # и мерить их нашей — значит мерить рассогласование промпта, а не качество вёрстки.
    if args.prompt_file:
        global PROMPT
        PROMPT = Path(args.prompt_file).read_text(encoding="utf-8").strip()
        print(f"[run_benchmark] Промпт взят из {args.prompt_file} "
              f"({len(PROMPT)} символов): {PROMPT[:120]!r}")

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

    if start_batch_idx > 0:
        print(f"[run_benchmark] Резюмирую с батча {start_batch_idx} "
              f"(уже обработано сэмплов: {dataset_offset}, накоплено метрик: {accumulator.count}).")

    llm = load_model(args.model, args.tensor_parallel_size, args.gpu_memory_utilization,
                     args.max_model_len, args.min_pixels, args.max_pixels)

    # CLIP-сервер стартует ОДИН РАЗ на весь прогон (не на батч, не на воркера)
    # — после vLLM (та же причина, что и раньше: не мешаем CUDA-инициализации
    # vLLM своей). Единственная копия CLIP на GPU, все --num-workers процессов
    # рендера+метрик шлют туда запросы вместо локальной загрузки модели.
    # Если --num-workers<=1, отдельный процесс не нужен — рендер+метрики и
    # так считаются в этом же процессе, metrics.py сам поднимет локальный
    # CLIP при первом обращении (см. _ensure_local_clip_loaded).
    clip_server = None
    clip_request_queue = None
    if args.num_workers > 1:
        from clip_server import ClipServer
        clip_server = ClipServer(clip_batch_size=args.clip_batch_size)
        clip_server.start()
        clip_request_queue = clip_server.request_queue

    rng = random.Random(args.seed + 1)  # отдельный seed для выбора examples, не путать с shuffle датасета

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

            examples_df, ok_df, n_gen_err, n_met_err, n_other_err = process_one_batch(
                llm, batch_samples, batch_idx, work_root, examples_root, args, rng,
                clip_request_queue=clip_request_queue,
            )

            append_examples_csv(results_path, examples_df)
            accumulator.add_batch(ok_df)
            n_generation_errors += n_gen_err
            n_metric_errors += n_met_err
            n_other_errors += n_other_err
            dataset_offset += len(batch_samples)
            batch_idx += 1

            progress = {
                "next_batch_idx": batch_idx,
                "dataset_offset": dataset_offset,
                "n_generation_errors": n_generation_errors,
                "n_metric_errors": n_metric_errors,
                "n_other_errors": n_other_errors,
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
    finally:
        # Гарантированная остановка CLIP-сервера даже при исключении/Ctrl+C —
        # иначе daemon-процесс с моделью на GPU останется висеть и держать
        # VRAM после падения основного процесса.
        if clip_server is not None:
            print("[run_benchmark] Останавливаю CLIP-сервер...")
            clip_server.stop()

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
        # Режим декодирования пишем в сводку: без него прогоны не сравнить
        # (старые summary.json сняты чистым сэмплингом при temperature 1.0).
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "max_new_tokens": args.max_new_tokens,
        },
        "materialize_dom": args.materialize_dom,
        "prompt_file": args.prompt_file,
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
    print("Итоговые средние по всем оценённым сэмплам (взвешенно, не среднее средних батчей):")
    for k, v in final_means.items():
        print(f"  {k}: {v:.4f}" if not math.isnan(v) else f"  {k}: nan")
    print(f"\nПримеры сохранены в: {examples_root}")
    print(f"Строки-примеры (results.csv): {results_path}")
    print(f"Итоговая сводка: {summary_path}")


if __name__ == "__main__":
    main()
