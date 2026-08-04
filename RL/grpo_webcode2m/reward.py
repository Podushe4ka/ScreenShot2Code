import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def pixel_similarity(
    target_path: str | Path,
    generated_path: str | Path,
) -> float:
    """Считает простую попиксельную близость двух изображений."""

    with Image.open(target_path) as target_image:
        target = target_image.convert("RGB")

    with Image.open(generated_path) as generated_image:
        generated = generated_image.convert("RGB")

    if target.size != generated.size:
        canvas_size = (
            max(target.width, generated.width),
            max(target.height, generated.height),
        )

        target_canvas = Image.new("RGB", canvas_size, (255, 255, 255))
        generated_canvas = Image.new("RGB", canvas_size, (255, 255, 255))

        target_canvas.paste(target, (0, 0))
        generated_canvas.paste(generated, (0, 0))

        target = target_canvas
        generated = generated_canvas

    target_array = np.asarray(target, dtype=np.float32)
    generated_array = np.asarray(generated, dtype=np.float32)

    mean_absolute_error = np.mean(
        np.abs(target_array - generated_array)
    )

    similarity = 1.0 - mean_absolute_error / 255.0

    return float(np.clip(similarity, 0.0, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--generated", required=True)
    args = parser.parse_args()

    score = pixel_similarity(
        target_path=args.target,
        generated_path=args.generated,
    )

    print(f"Pixel similarity: {score:.4f}")


if __name__ == "__main__":
    main()