"""ClearML-трекинг прогонов бенча. Опциональный и безопасный.

Главная цель плана — `final_score` на Design2Code выше базы. Именно прогон бенча,
а не обучение, даёт эту цифру, поэтому каждый прогон = отдельная ClearML-задача:
в UI их сравниваешь бок о бок (сетап × final_score), и это те самые «одна строка =
один сетап» из протокола метрик.

Безопасность: если clearml не установлен / сервер не настроен / CLEARML_DISABLE=1 —
все функции становятся no-op и возвращают None. Бенч не должен падать из-за трекера.
"""

import json
import logging
import math
import os

logger = logging.getLogger(__name__)

DEFAULT_PROJECT = "ScreenShot2Code/Bench"

# Подметрики visual_eval_v3, которые логируем как сравнимые единичные значения.
_METRIC_COLS = [
    "final_score",
    "final_score_arithmetic",
    "block_match",
    "text",
    "position",
    "color",
    "clip",
]


def _pixels_tag(pixels) -> str | None:
    if not pixels:
        return None
    return f"px{pixels / 1e6:.2f}Mp"


def start_benchmark_task(args):
    """Поднять ClearML Task для прогона бенча. Возвращает Task или None.

    `args` — argparse.Namespace из run_benchmark. Имя модели используем как имя
    задачи (для локального чекпоинта — имя каталога), гиперпараметры прогона
    (пиксель-бюджет, max_new_tokens, датасет) уходят в connect и в теги.
    """
    if os.environ.get("CLEARML_DISABLE") == "1":
        logger.info("CLEARML_DISABLE=1 — трекинг выключен")
        return None
    try:
        from clearml import Task
    except ImportError:
        logger.warning("clearml не установлен — трекинг выключен (pip install clearml)")
        return None

    model = str(args.model)
    task_name = model if "/" in model and not os.path.isdir(model) else os.path.basename(
        model.rstrip("/")
    )
    tags = ["bench", args.hf_dataset.split("/")[-1]]
    px = _pixels_tag(getattr(args, "max_pixels", None))
    if px:
        tags.append(px)
    # Доп. теги из CLEARML_TAGS (через запятую): S-ID, приоритет, напр. S5,P0.
    tags += [t.strip() for t in os.environ.get("CLEARML_TAGS", "").split(",") if t.strip()]

    try:
        task = Task.init(
            project_name=os.environ.get("CLEARML_PROJECT", DEFAULT_PROJECT),
            task_name=f"bench:{task_name}",
            tags=tags,
            reuse_last_task_id=False,
            auto_connect_frameworks=False,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("не удалось поднять clearml Task (%s) — трекинг выключен", e)
        return None

    bench_args = {
        "model": model,
        "hf_dataset": args.hf_dataset,
        "n_samples": args.n_samples,
        "max_new_tokens": args.max_new_tokens,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "max_model_len": args.max_model_len,
        "enable_thinking": args.enable_thinking,
        "seed": args.seed,
    }

    # Связь с раном обучения: из какого чекпоинта эти веса.
    train_link = _find_train_link(model, args)
    if train_link:
        bench_args.update(train_link)
        _link_to_train_task(task, train_link["train_task_id"])

    task.connect(bench_args, name="bench_args")
    return task


def _find_train_link(model, args=None) -> dict | None:
    """Найти `clearml_task.json`, положенный обучением рядом с чекпоинтом.

    Явное указание (`--train-task-id` или CLEARML_TRAIN_TASK) имеет приоритет
    над файлом: при ручном копировании весов файл мог не поехать вместе с ними."""
    explicit = getattr(args, "train_task_id", None) or os.environ.get("CLEARML_TRAIN_TASK")
    if explicit:
        return {"train_task_id": explicit}
    if not os.path.isdir(str(model)):
        return None  # модель с HF-хаба — обучения за ней нет
    path = os.path.join(str(model), "clearml_task.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        logger.warning("не удалось прочитать %s: %s", path, e)
        return None
    return {k: v for k, v in data.items() if v is not None}


def _link_to_train_task(task, train_task_id) -> None:
    """Двусторонняя связь: тег+ссылка на бенче и обратная ссылка на обучении.

    Односторонней мало: в UI обычно идёшь от рана обучения к его метрикам, а не
    наоборот."""
    try:
        from clearml import Task

        task.add_tags([f"train:{train_task_id}"])
        train = Task.get_task(task_id=train_task_id)
        if train is not None:
            train.add_tags([f"bench:{task.id}"])
    except Exception as e:  # noqa: BLE001
        logger.warning("не удалось связать с задачей обучения %s: %s", train_task_id, e)


def log_benchmark_results(task, df, results_path=None):
    """Залогировать агрегаты прогона: final_score + подметрики (сравнимые
    единичные значения), разбивку status/finish_reason и results.csv-артефакт.

    `df` — итоговый DataFrame из run_benchmark (по строке на сэмпл)."""
    if task is None:
        return

    logger_cml = task.get_logger()
    ok = df[df["status"] == "scored"] if "status" in df else df

    # Средние по успешно оценённым — это и есть числа прогона. report_single_value
    # кладёт их в сравнимую таблицу "single values" в UI.
    for col in _METRIC_COLS:
        if col in ok and len(ok) > 0:
            logger_cml.report_single_value(col, float(ok[col].mean()))

    logger_cml.report_single_value("n_total", int(len(df)))
    logger_cml.report_single_value("n_scored", int(len(ok)))

    # Гигиена прогона: сколько обрезано по токенам (H2) и как распределились статусы.
    if "finish_reason" in df:
        n_len = int((df["finish_reason"] == "length").sum())
        logger_cml.report_single_value("n_length_truncated", n_len)
        for reason, n in df["finish_reason"].value_counts().items():
            logger_cml.report_single_value(f"finish_reason/{reason}", int(n))
    if "status" in df:
        for status, n in df["status"].value_counts().items():
            # статус может быть 'metric_error: ...' — берём только префикс
            key = str(status).split(":")[0]
            logger_cml.report_single_value(f"status/{key}", int(n))

    if results_path is not None:
        try:
            task.upload_artifact("results", artifact_object=str(results_path))
        except Exception as e:  # noqa: BLE001
            logger.warning("не удалось залить results.csv в clearml: %s", e)


def log_benchmark_summary(task, summary, results_path=None):
    """То же, но из готовой сводки `summary` — для run_benchmark_batched.

    Батчевый прогон копит агрегаты на лету и не держит DataFrame по всем
    сэмплам (при 70k это память), поэтому логируем из summary.json: подметрики
    уже усреднены взвешенно, счётчики посчитаны."""
    if task is None:
        return

    logger_cml = task.get_logger()
    for col in _METRIC_COLS:
        v = summary.get(col)
        # nan прилетает, когда ни один сэмпл не оценён — в UI он бесполезен.
        if v is not None and not math.isnan(float(v)):
            logger_cml.report_single_value(col, float(v))

    for key in (
        "n_samples_processed",
        "n_scored",
        "n_excluded_total",
        "n_generation_errors",
        "n_metric_errors",
        "n_other_errors",
        "n_length_truncated",
    ):
        if key in summary:
            logger_cml.report_single_value(key, int(summary[key]))

    if results_path is not None:
        try:
            task.upload_artifact("results", artifact_object=str(results_path))
        except Exception as e:  # noqa: BLE001
            logger.warning("не удалось залить results.csv в clearml: %s", e)
