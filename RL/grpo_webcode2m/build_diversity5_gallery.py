import html
import os
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

STORAGE_ROOT = Path(os.environ.get(
    "STORAGE_ROOT",
    f"/mnt/storage-1/{os.environ.get('USER', 'user')}",
))

from renderer import render_html


SRC = Path.home() / (
    "mla_project/outputs/grpo_4b_webcode_diversity5/"
    "validation/0.jsonl"
)

OUT = STORAGE_ROOT / "analysis/diversity5_gallery"

IMAGES = OUT / "images"
IMAGES.mkdir(parents=True, exist_ok=True)


def extract_html(text: str) -> str | None:
    fenced = re.search(
        r"```html\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        return fenced.group(1).strip()

    start = re.search(
        r"<!DOCTYPE\s+html|<html",
        text,
        flags=re.IGNORECASE,
    )
    end = text.lower().rfind("</html>")

    if start and end >= 0:
        return text[start.start():end + len("</html>")]

    return None


rows = [
    json.loads(line)
    for line in SRC.read_text().splitlines()
    if line.strip()
]

groups = defaultdict(list)

for index, row in enumerate(rows):
    row["_index"] = index
    groups[row["gts"]].append(row)

groups = sorted(
    groups.items(),
    key=lambda item: Path(item[0]).name,
)

cards = []
labels = ["A", "B", "C", "D"]

for page_number, (target_path, variants) in enumerate(groups, start=1):
    variants.sort(key=lambda row: row["_index"])

    target_src = Path(target_path)
    target_dst = IMAGES / f"{page_number:02d}_target.png"
    shutil.copy2(target_src, target_dst)

    columns = [
        f"""
        <div class="column">
            <h3>Target</h3>
            <img src="images/{target_dst.name}">
        </div>
        """
    ]

    rewards = []

    for variant_index, label in enumerate(labels):
        if variant_index >= len(variants):
            columns.append(
                f"""
                <div class="column">
                    <h3>{label}</h3>
                    <div class="invalid">Генерация отсутствует</div>
                </div>
                """
            )
            rewards.append((label, 0.0))
            continue

        row = variants[variant_index]
        generated_html = extract_html(row.get("output", ""))

        reward = float(
            row.get("reward", row.get("score", 0.0))
        )
        rewards.append((label, reward))

        valid = (
            generated_html is not None
            and float(row.get("html_ok", 0)) == 1
            and float(row.get("render_ok", 0)) == 1
        )

        if valid:
            generated_dst = (
                IMAGES / f"{page_number:02d}_{label}.png"
            )

            render_html(
                html=generated_html,
                output_path=generated_dst,
                width=1280,
                height=1024,
                device_scale_factor=1.0,
                browser_endpoint="ws://127.0.0.1:3001/",
                full_page=True,
            )

            visual = (
                f'<img src="images/{generated_dst.name}">'
            )
        else:
            visual = """
            <div class="invalid">
                HTML обрезан или невалиден
            </div>
            """

        columns.append(
            f"""
            <div class="column">
                <h3>Generation {label}</h3>
                {visual}
            </div>
            """
        )

    rewards_html = " &nbsp; ".join(
        f"{label}: <strong>{reward:.4f}</strong>"
        for label, reward in rewards
    )

    cards.append(
        f"""
        <section class="card">
            <h2>
                Страница {page_number}:
                {html.escape(target_src.name)}
            </h2>

            <div class="comparison">
                {''.join(columns)}
            </div>

            <p class="instruction">
                Сначала выбери глазами:
                <strong>A / B / C / D / tie</strong>
            </p>

            <details>
                <summary>Показать rewards</summary>
                <p>{rewards_html}</p>
            </details>
        </section>
        """
    )

index_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Diversity5 gallery</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 20px;
    background: #f2f2f2;
}}

.card {{
    margin-bottom: 36px;
    padding: 18px;
    background: white;
    border: 1px solid #bbb;
}}

.comparison {{
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 12px;
    align-items: start;
}}

.column img {{
    width: 100%;
    border: 1px solid #999;
}}

.invalid {{
    min-height: 220px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    background: #ddd;
    border: 1px solid #999;
    font-weight: bold;
}}

.instruction {{
    padding: 12px;
    background: #fff4bd;
}}

summary {{
    cursor: pointer;
    font-weight: bold;
}}
</style>
</head>
<body>
<h1>Diversity5: Target и четыре генерации</h1>
<p>
Сначала выбери лучший вариант глазами.
Только потом раскрывай rewards.
</p>

{''.join(cards)}
</body>
</html>
"""

(OUT / "index.html").write_text(index_html)

print("Готово:", OUT / "index.html")
print("Target-страниц:", len(groups))
print("Генераций:", len(rows))
