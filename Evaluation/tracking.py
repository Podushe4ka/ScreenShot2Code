"""ClearML-трекинг прогонов бенча. Опциональный и безопасный.

Главная цель плана — `final_score` на Design2Code выше базы. Именно прогон бенча,
а не обучение, даёт эту цифру, поэтому каждый прогон = отдельная ClearML-задача:
в UI их сравниваешь бок о бок (сетап × final_score), и это те самые «одна строка =
один сетап» из протокола метрик.

Безопасность: если clearml не установлен / сервер не настроен / CLEARML_DISABLE=1 —
все функции становятся no-op и возвращают None. Бенч не должен падать из-за трекера.
"""

import logging
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

    task.connect(
        {
            "model": model,
            "hf_dataset": args.hf_dataset,
            "n_samples": args.n_samples,
            "max_new_tokens": args.max_new_tokens,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "max_model_len": args.max_model_len,
            "enable_thinking": args.enable_thinking,
            "seed": args.seed,
        },
        name="bench_args",
    )
    return task


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
