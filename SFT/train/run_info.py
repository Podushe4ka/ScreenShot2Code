"""Опознание рана: имя, каталог, снимок конфига и метаданных.

Задача — чтобы через две недели по кривой в трекере можно было восстановить,
каким конфигом она получена. Сохраняются ЭФФЕКТИВНЫЕ аргументы (после
переопределений из командной строки), а не исходный yaml, плюс то, чего в
аргументах нет: бюджет токенов, сколько сэмплов отбраковано, коммит.
"""

import json
import logging
import subprocess
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

RUN_INFO_FILE = "run_info.yaml"


def git_commit() -> str | None:
    """Текущий коммит, если запускаемся из рабочей копии."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def make_run_name(training_args, model_args) -> str:
    """Уникальное имя рана: база + seed + отметка времени.

    HF по умолчанию приравнивает `run_name` к `output_dir`, поэтому такое
    значение считаем «не задано». Отметка времени нужна, чтобы повторный запуск
    того же конфига не затирал предыдущий и был отличим в трекере.
    """
    base = training_args.run_name
    if not base or base == training_args.output_dir:
        base = Path(training_args.output_dir).name or "run"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{base}_s{training_args.seed}_{stamp}"


def _as_dict(obj) -> dict:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if is_dataclass(obj):
        return asdict(obj)
    return dict(vars(obj))


def _plain(value):
    """Привести к тому, что переживёт yaml.safe_dump."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_run_info(output_dir, *, script_args, training_args, model_args, meta) -> Path:
    """Записать снимок рана в `output_dir/run_info.yaml`."""
    payload = {
        "meta": _plain(meta),
        "script_args": _plain(_as_dict(script_args)),
        "model_args": _plain(_as_dict(model_args)),
        "training_args": _plain(_as_dict(training_args)),
    }
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    target = path / RUN_INFO_FILE
    target.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
    return target


def format_meta(meta: dict) -> str:
    """Однострочный дамп метаданных — чтобы он был виден в логе рана."""
    return json.dumps(meta, ensure_ascii=False, sort_keys=True)
