import argparse
import os
import random
from pathlib import Path

STORAGE_ROOT = Path(os.environ.get(
    "STORAGE_ROOT",
    f"/mnt/storage-1/{os.environ.get('USER', 'user')}",
))

from datasets import Dataset, load_from_disk


INSTRUCTION = (
    "Recreate the screenshot as a complete self-contained HTML document "
    "with inline CSS. Use a compact implementation. "
    "Do not use JavaScript, external assets, analysis, or explanations. "
    "Start with <!DOCTYPE html> and end with </html>."
)


def convert_example(
    row: dict,
    sample_id: str,
    image_dir: Path,
) -> dict:
    """Преобразует одну страницу WebCode2M в формат для verl."""

    image = row["images"][0]

    image_path = image_dir / f"{sample_id}.png"
    image.save(image_path, format="PNG")

    image_bytes = image_path.read_bytes()

    return {
        "data_source": "webcode2m_screenshot2code",

        "prompt": [
            {
                "role": "user",
                "content": f"<image>\n{INSTRUCTION}",
            }
        ],

        "images": [
            {
                "bytes": image_bytes,
                "path": None,
            }
        ],

        "ability": "screenshot2code",

        "reward_model": {
            "style": "rule",
            "ground_truth": str(image_path),
        },

        # Правильный HTML сохраняем для анализа,
        # хотя pixel reward напрямую его не использует.
        "target_html": row["target_html"],

        "extra_info": {
            "sample_id": sample_id,
            "image_path": str(image_path),
            "viewport_width": 1280,
            "viewport_height": 1024,
            "device_scale_factor": 1.0,
            "full_page": True,
            "original_image_width": image.width,
            "original_image_height": image.height,
        },
    }


def convert_split(
    source_split,
    indices: list[int],
    split_name: str,
    output_dir: Path,
    limit: int | None,
) -> Dataset:
    if limit is not None:
        indices = indices[:limit]

    image_dir = output_dir / "images" / split_name
    image_dir.mkdir(parents=True, exist_ok=True)

    examples = []

    for original_index in indices:
        row = source_split[original_index]

        sample_id = (
            f"webcode2m_{split_name}_{original_index:05d}"
        )

        examples.append(
            convert_example(
                row=row,
                sample_id=sample_id,
                image_dir=image_dir,
            )
        )

    return Dataset.from_list(examples)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default="/mnt/storage-1/data/webcode2m_1000_split",
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            STORAGE_ROOT / "data/verl_webcode2m_991"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ограничить число примеров в каждом разделе.",
    )

    args = parser.parse_args()

    source = load_from_disk(args.source)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Один раз фиксированно выбираем 50 страниц для закрытого теста.
    train_indices = list(range(len(source["train"])))

    random_generator = random.Random(42)
    random_generator.shuffle(train_indices)

    test_indices = sorted(train_indices[:50])
    final_train_indices = sorted(train_indices[50:])

    validation_indices = list(
        range(len(source["validation"]))
    )

    train_dataset = convert_split(
        source_split=source["train"],
        indices=final_train_indices,
        split_name="train",
        output_dir=output_dir,
        limit=args.limit,
    )

    validation_dataset = convert_split(
        source_split=source["validation"],
        indices=validation_indices,
        split_name="validation",
        output_dir=output_dir,
        limit=args.limit,
    )

    test_dataset = convert_split(
        source_split=source["train"],
        indices=test_indices,
        split_name="test",
        output_dir=output_dir,
        limit=args.limit,
    )

    train_dataset.to_parquet(output_dir / "train.parquet")
    validation_dataset.to_parquet(
        output_dir / "validation.parquet"
    )
    test_dataset.to_parquet(output_dir / "test.parquet")

    print("Output directory:", output_dir)
    print("Train:", len(train_dataset))
    print("Validation:", len(validation_dataset))
    print("Closed test:", len(test_dataset))


if __name__ == "__main__":
    main()