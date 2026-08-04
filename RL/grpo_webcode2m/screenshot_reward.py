from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from renderer import render_html
from reward import pixel_similarity


def extract_html(response: str) -> str:
    """Извлекает полный HTML-документ из ответа модели."""

    lower_response = response.lower()

    starts = [
        position
        for position in (
            lower_response.find("<!doctype html"),
            lower_response.find("<html"),
        )
        if position != -1
    ]

    if not starts:
        raise ValueError("HTML start was not found")

    start = min(starts)

    closing_tag = "</html>"
    end = lower_response.rfind(closing_tag)

    if end == -1:
        raise ValueError("Closing </html> tag was not found")

    return response[start : end + len(closing_tag)]


def resolve_target_path(ground_truth: Any) -> Path:
    """Получает путь к целевому скриншоту из ground_truth."""

    value = ground_truth

    if isinstance(value, dict):
        possible_keys = (
            "target_path",
            "image_path",
            "screenshot_path",
            "path",
        )

        for key in possible_keys:
            if key in value:
                value = value[key]
                break
        else:
            raise ValueError(
                "ground_truth dict does not contain a screenshot path"
            )

    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(
                "ground_truth list must contain exactly one path"
            )

        value = value[0]

    target_path = Path(str(value)).expanduser()

    if target_path.is_absolute():
        resolved_path = target_path.resolve()
    else:
        project_candidate = (PROJECT_DIR / target_path).resolve()
        current_directory_candidate = (
            Path.cwd() / target_path
        ).resolve()

        if project_candidate.exists():
            resolved_path = project_candidate
        else:
            resolved_path = current_directory_candidate

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Target screenshot was not found: {resolved_path}"
        )

    return resolved_path


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    device_scale_factor: float = 2.0,
    browser_endpoint: str | None = None,
) -> dict[str, float]:
    """
    Считает reward для Screenshot-to-Code.

    Ответ модели:
    HTML → Playwright → PNG → pixel similarity.
    """

    extra_info = extra_info or {}

    html_ok = 0.0
    render_ok = 0.0

    try:
        html = extract_html(str(solution_str))
        html_ok = 1.0

        target_path = resolve_target_path(ground_truth)

        with Image.open(target_path) as target_image:
            target_width, target_height = target_image.size

        scale_factor = float(
            extra_info.get(
                "device_scale_factor",
                device_scale_factor,
            )
        )

        if scale_factor <= 0:
            raise ValueError(
                "device_scale_factor must be greater than zero"
            )

        viewport_width = int(
            extra_info.get(
                "viewport_width",
                extra_info.get(
                    "width",
                    round(target_width / scale_factor),
                ),
            )
        )

        viewport_height = int(
            extra_info.get(
                "viewport_height",
                extra_info.get(
                    "height",
                    round(target_height / scale_factor),
                ),
            )
        )

        if viewport_width <= 0 or viewport_height <= 0:
            raise ValueError(
                "Viewport width and height must be positive"
            )

        # У каждого rollout будет отдельная временная папка,
        # чтобы параллельные оценки не перезаписывали файлы.
        with tempfile.TemporaryDirectory(
            prefix="screenshot_reward_"
        ) as temporary_directory:
            generated_path = (
                Path(temporary_directory) / "generated.png"
            )

            full_page = bool(extra_info.get("full_page", False))

            render_html(
                html=html,
                output_path=generated_path,
                width=viewport_width,
                height=viewport_height,
                device_scale_factor=scale_factor,
                full_page=full_page,
                browser_endpoint=browser_endpoint,
            )

            render_ok = 1.0

            score = pixel_similarity(
                target_path=target_path,
                generated_path=generated_path,
            )

        return {
            "score": float(score),
            "pixel_similarity": float(score),
            "html_ok": html_ok,
            "render_ok": render_ok,
        }

    except Exception as error:
        print(
            "[screenshot_reward] "
            f"data_source={data_source}; "
            f"{type(error).__name__}: {error}"
        )

        return {
            "score": 0.0,
            "pixel_similarity": 0.0,
            "html_ok": html_ok,
            "render_ok": render_ok,
        }
