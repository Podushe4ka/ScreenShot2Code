"""Обогащение ClearML-задачи обучения тегами/гиперпараметрами.

Саму задачу создаёт авто-callback `transformers` (когда `report_to=["clearml"]`,
см. `enable_clearml_if_configured` в train_sft.py) — здесь мы её НЕ создаём, а
только досыпаем то, чего авто-логика не знает: осмысленные теги (датасет,
LoRA/full, пиксель-бюджет, S-ID) и плоский срез гиперпараметров по осям свипа.

Делается через `TrainerCallback`, добавленный ПОСЛЕ ClearMLCallback: на
`on_train_begin` тот уже создал `Task`, и мы обогащаем `Task.current_task()`.
На не-главных рангах текущей задачи нет → тихий no-op. Если clearml не
установлен — модуль тоже становится no-op и обучение не падает.
"""

import json
import logging
import os
from pathlib import Path

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


def _pixels_tag(pixels) -> str | None:
    """`2097152 -> 'px2.10Mp'` — тег пиксель-бюджета для оси A из плана."""
    if not pixels:
        return None
    return f"px{pixels / 1e6:.2f}Mp"


def _env_tags() -> list[str]:
    """Доп. теги из CLEARML_TAGS (через запятую) — сюда кладём S-ID и приоритет,
    напр. `CLEARML_TAGS=S5,P0`."""
    raw = os.environ.get("CLEARML_TAGS", "")
    return [t.strip() for t in raw.split(",") if t.strip()]


def _tags(meta: dict) -> list[str]:
    """Теги для фильтрации/сравнения в UI. Ложатся на оси плана (датасет,
    LoRA/full, пиксель-бюджет)."""
    tags = ["SFT"]
    if meta.get("dataset"):
        # 'org/WebCode2M-hf' -> 'WebCode2M-hf'
        tags.append(str(meta["dataset"]).split("/")[-1])
    tags.append("lora" if meta.get("peft") else "full-ft")
    px = _pixels_tag(meta.get("image_pixels"))
    if px:
        tags.append(px)
    return tags + _env_tags()


def _hparams(meta: dict, training_args, model_args) -> dict:
    """Плоский срез гиперпараметров ровно по осям свипа A–H — по нему в UI
    сравниваются раны."""
    return {
        "model": model_args.model_name_or_path,
        "lora": bool(meta.get("peft")),
        "dataset": meta.get("dataset"),
        "max_pixels": meta.get("image_pixels"),
        "max_length": meta.get("max_length"),
        "lr": getattr(training_args, "learning_rate", None),
        "epochs": getattr(training_args, "num_train_epochs", None),
        "per_device_bs": getattr(training_args, "per_device_train_batch_size", None),
        "grad_accum": getattr(training_args, "gradient_accumulation_steps", None),
        "seed": getattr(training_args, "seed", None),
    }


class ClearMLEnrichCallback(TrainerCallback):
    """Досыпает теги/hparams/meta в задачу, созданную ClearMLCallback.

    Добавлять в трейнер ПОСЛЕ штатных колбэков (`trainer.add_callback(...)`),
    чтобы на on_train_begin текущая задача уже существовала."""

    def __init__(self, meta, training_args, model_args):
        self._meta = meta
        self._training_args = training_args
        self._model_args = model_args
        self._done = False

    def on_train_begin(self, args, state, control, **kwargs):
        if self._done:
            return
        try:
            from clearml import Task
        except ImportError:
            return
        task = Task.current_task()
        if task is None:  # не главный ранг или clearml выключен — тихо выходим
            return
        try:
            task.add_tags(_tags(self._meta))
            task.connect(self._meta, name="meta")
            task.connect(
                _hparams(self._meta, self._training_args, self._model_args),
                name="hparams",
            )
            logger.info("ClearML: задача обогащена тегами/hparams")
        except Exception as e:  # noqa: BLE001 — трекер не должен ронять обучение
            logger.warning("не удалось обогатить clearml Task: %s", e)
        save_task_link(self._training_args.output_dir, task, self._meta)
        self._done = True


def save_task_link(output_dir, task=None, meta=None) -> None:
    """Положить id задачи обучения рядом с чекпоинтом (`clearml_task.json`).

    Это мост train -> bench: бенч запускается отдельным контейнером и о ClearML
    обучения ничего не знает. `merge_lora` переносит этот файл к слитым весам, а
    `Evaluation/tracking.py` подхватывает его и проставляет ссылку на ран.
    Ошибки глушим — трекер не должен ронять обучение."""
    if task is None:
        try:
            from clearml import Task

            task = Task.current_task()
        except ImportError:
            return
    if task is None or not output_dir:
        return
    try:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        payload = {"train_task_id": task.id, "train_task_name": task.name}
        if meta:
            payload["model"] = meta.get("model")
            payload["dataset"] = meta.get("dataset")
            payload["peft"] = meta.get("peft")
        (path / "clearml_task.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("ClearML: ссылка на задачу сохранена в %s", path / "clearml_task.json")
    except Exception as e:  # noqa: BLE001
        logger.warning("не удалось сохранить clearml_task.json: %s", e)
