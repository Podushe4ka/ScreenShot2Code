import argparse
from pathlib import Path

from datasets import Dataset


INSTRUCTION = (
    "Recreate the screenshot as a complete self-contained HTML document "
    "with inline CSS. Use a compact implementation. "
    "Do not use JavaScript, external assets, analysis, or explanations. "
    "Start with <!DOCTYPE html> and end with </html>."
)


def build_example(
    image_path: Path,
    device_scale_factor: float,
) -> dict:
    """Создаёт одну строку датасета в формате verl."""

    image_path = image_path.resolve()

    if not image_path.exists():
        raise FileNotFoundError(
            f"Screenshot not found: {image_path}"
        )

    return {
        "data_source": "screenshot2code",

        # <image> показывает, в каком месте сообщения находится изображение.
        "prompt": [
            {
                "role": "user",
                "content": f"<image>\n{INSTRUCTION}",
            }
        ],

        # Изображение передаётся модели как сырые байты.
        "images": [
            {
                "bytes": image_path.read_bytes(),
                "path": None,
            }
        ],

        "ability": "screenshot2code",

        # Этот путь verl позже передаст в compute_score()
        # как аргумент ground_truth.
        "reward_model": {
            "style": "rule",
            "ground_truth": str(image_path),
        },

        "extra_info": {
            "image_path": str(image_path),
            "device_scale_factor": device_scale_factor,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the target screenshot.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(Path.home() / "mla_project/data/verl_smoke"),
    )

    parser.add_argument(
        "--device-scale-factor",
        type=float,
        default=2.0,
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    example = build_example(
        image_path=image_path,
        device_scale_factor=args.device_scale_factor,
    )

    # Пока используем один пример и для обучения, и для проверки.
    # Это только smoke-test полного GRPO-пайплайна.
    train_dataset = Dataset.from_list([example])
    test_dataset = Dataset.from_list([example])

    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"Train dataset: {train_path}")
    print(f"Test dataset:  {test_path}")
    print(f"Rows in train: {len(train_dataset)}")
    print(f"Rows in test:  {len(test_dataset)}")


if __name__ == "__main__":
    main()