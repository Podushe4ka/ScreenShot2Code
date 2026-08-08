"""
labeler.py — GUI-разметка 400 сэмплов: кто ближе к ref.png, baseline или
checkpoint.

Структура входных данных (см. --data-root):
    batch_00000/
        sample_00000/
            pred_baseline.png
            pred_checkpoint.png
            ref.png
        sample_00001/
        ...
        sample_00399/

На экране: ref сверху по центру, снизу два кандидата слева/справа.
Anti-position-bias: для КАЖДОГО сэмпла независимо (seed зависит от имени
сэмпла, так что порядок стабилен между перезапусками одного и того же
сэмпла, но не коррелирует между сэмплами) подбрасывается монетка — baseline
показывается слева или справа. На экране кандидаты подписаны только "1" и
"2" (не "baseline"/"checkpoint") — Даниил как разметчик тоже не должен знать,
какая картинка чем сгенерирована, иначе это не blind-разметка и в неё
просочится тот же position/label bias, от которого мы защищаем LLM-судью.

Хоткеи: Left / A -> выбрать левый как победителя, Right / D -> выбрать
правый, Backspace -> отменить последний ответ и вернуться на шаг назад.
Прогресс сохраняется в labels.csv после КАЖДОГО ответа (не только в конце) —
разметку 400 сэмплов не обязательно делать за один присест, скрипт сам
находит первый неразмеченный сэмпл при перезапуске.

labels.csv формат:
    sample_id,winner,left_is,swapped,timestamp
где winner всегда "baseline" или "checkpoint" (уже развёрнуто обратно из
"left"/"right" через swapped) — при анализе результатов можно вообще не
думать о том, где что было показано на экране.
"""

import argparse
import csv
import time
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

from sample_order import swapped_for_sample

CSV_FIELDS = ["sample_id", "winner", "left_is", "swapped", "timestamp"]
MAX_PREVIEW_W = 520
MAX_PREVIEW_H = 380


def discover_samples(data_root: Path) -> list[str]:
    """Возвращает отсортированные имена папок sample_XXXXX, у которых
    реально есть все три нужных файла — пропускаем (с предупреждением)
    сэмплы с недостающими картинками, а не падаем посреди разметки."""
    samples = []
    for batch_dir in sorted(data_root.glob("batch_*")):
        if not batch_dir.is_dir():
            continue
        for sample_dir in sorted(batch_dir.glob("sample_*")):
            if not sample_dir.is_dir():
                continue
            needed = ["ref.png", "pred_baseline.png", "pred_checkpoint.png"]
            if all((sample_dir / f).exists() for f in needed):
                # sample_id хранит относительный путь batch/sample, чтобы
                # при нескольких батчах (не только batch_00000) id оставались
                # уникальными и человекочитаемыми в CSV.
                samples.append(str(sample_dir.relative_to(data_root)))
            else:
                missing = [f for f in needed if not (sample_dir / f).exists()]
                print(f"[labeler] Пропускаю {sample_dir}: нет файлов {missing}")
    return samples


def load_existing_labels(labels_path: Path) -> dict[str, dict]:
    if not labels_path.exists():
        return {}
    with open(labels_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["sample_id"]: row for row in reader}


def append_label(labels_path: Path, row: dict) -> None:
    file_exists = labels_path.exists()
    with open(labels_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def rewrite_labels(labels_path: Path, rows: list[dict]) -> None:
    """Полная перезапись CSV — используется только при Backspace (удаление
    последней строки), поэтому нечастая операция; append_label выше
    достаточно для обычного потока разметки вперёд."""
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def fit_image(path: Path, max_w: int, max_h: int) -> ImageTk.PhotoImage:
    img = Image.open(path)
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    return ImageTk.PhotoImage(img)


class LabelerApp:
    def __init__(self, root: tk.Tk, data_root: Path, labels_path: Path, samples: list[str]):
        self.root = root
        self.data_root = data_root
        self.labels_path = labels_path
        self.samples = samples
        self.existing = load_existing_labels(labels_path)
        # Порядок ответов важен для Backspace (нужно знать, какая строка
        # последняя) — держим отдельный список id-шников в порядке разметки,
        # а не полагаемся на порядок словаря/файла.
        self.answered_order = [s for s in samples if s in self.existing]
        self.idx = self._first_unanswered_index()

        self.root.title("Design2Code — разметка baseline vs checkpoint")
        self.root.geometry("1150x760")
        self.root.bind("<Left>", lambda e: self.choose("left"))
        self.root.bind("<a>", lambda e: self.choose("left"))
        self.root.bind("<Right>", lambda e: self.choose("right"))
        self.root.bind("<d>", lambda e: self.choose("right"))
        self.root.bind("<BackSpace>", lambda e: self.undo())

        self.progress_label = tk.Label(root, font=("Helvetica", 13))
        self.progress_label.pack(pady=(8, 0))

        self.sample_label = tk.Label(root, font=("Helvetica", 10), fg="#666")
        self.sample_label.pack()

        ref_frame = tk.Frame(root)
        ref_frame.pack(pady=6)
        tk.Label(ref_frame, text="ЭТАЛОН (ref)", font=("Helvetica", 11, "bold")).pack()
        self.ref_canvas = tk.Label(ref_frame)
        self.ref_canvas.pack()

        cand_frame = tk.Frame(root)
        cand_frame.pack(pady=6)

        left_frame = tk.Frame(cand_frame)
        left_frame.pack(side="left", padx=20)
        tk.Label(left_frame, text="Вариант 1  (← / A)", font=("Helvetica", 12, "bold")).pack()
        self.left_canvas = tk.Label(left_frame, cursor="hand2")
        self.left_canvas.pack()
        self.left_canvas.bind("<Button-1>", lambda e: self.choose("left"))

        right_frame = tk.Frame(cand_frame)
        right_frame.pack(side="left", padx=20)
        tk.Label(right_frame, text="Вариант 2  (→ / D)", font=("Helvetica", 12, "bold")).pack()
        self.right_canvas = tk.Label(right_frame, cursor="hand2")
        self.right_canvas.pack()
        self.right_canvas.bind("<Button-1>", lambda e: self.choose("right"))

        hint = tk.Label(
            root,
            text="← / A = вариант 1 лучше приближает эталон   |   → / D = вариант 2 лучше   |   Backspace = отменить последний ответ",
            font=("Helvetica", 10), fg="#888",
        )
        hint.pack(pady=(6, 4))

        self.status_label = tk.Label(root, font=("Helvetica", 10), fg="#0a0")
        self.status_label.pack()

        self._image_refs = []  # держим ссылки на PhotoImage, иначе GC их снесёт
        self.render_current()

    def _first_unanswered_index(self) -> int:
        for i, s in enumerate(self.samples):
            if s not in self.existing:
                return i
        return len(self.samples)  # всё уже размечено

    def render_current(self):
        if self.idx >= len(self.samples):
            self.finish_screen()
            return

        sample_id = self.samples[self.idx]
        sample_dir = self.data_root / sample_id
        swapped = swapped_for_sample(sample_id)
        left_is = "checkpoint" if swapped else "baseline"
        right_is = "baseline" if swapped else "checkpoint"

        self._image_refs.clear()
        ref_img = fit_image(sample_dir / "ref.png", MAX_PREVIEW_W, 260)
        left_img = fit_image(sample_dir / f"pred_{left_is}.png", MAX_PREVIEW_W, MAX_PREVIEW_H)
        right_img = fit_image(sample_dir / f"pred_{right_is}.png", MAX_PREVIEW_W, MAX_PREVIEW_H)
        self._image_refs.extend([ref_img, left_img, right_img])

        self.ref_canvas.configure(image=ref_img)
        self.left_canvas.configure(image=left_img)
        self.right_canvas.configure(image=right_img)

        self._current_left_is = left_is
        self._current_right_is = right_is
        self._current_swapped = swapped

        n_done = len(self.answered_order)
        self.progress_label.configure(
            text=f"Сэмпл {self.idx + 1} / {len(self.samples)}   (размечено: {n_done})"
        )
        self.sample_label.configure(text=sample_id)
        self.status_label.configure(text="")

    def choose(self, side: str):
        if self.idx >= len(self.samples):
            return
        sample_id = self.samples[self.idx]
        winner = self._current_left_is if side == "left" else self._current_right_is

        row = {
            "sample_id": sample_id,
            "winner": winner,
            "left_is": self._current_left_is,
            "swapped": str(self._current_swapped),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        append_label(self.labels_path, row)
        self.existing[sample_id] = row
        self.answered_order.append(sample_id)

        self.idx += 1
        self.render_current()

    def undo(self):
        if not self.answered_order:
            self.status_label.configure(text="Отменять нечего — ответов ещё не было.")
            return
        last_id = self.answered_order.pop()
        del self.existing[last_id]
        rewrite_labels(self.labels_path, list(self.existing.values()))
        self.idx = self.samples.index(last_id)
        self.render_current()
        self.status_label.configure(text=f"Отменён ответ для {last_id}, показываю его снова.")

    def finish_screen(self):
        for w in self.root.winfo_children():
            w.destroy()
        n = len(self.answered_order)
        tk.Label(
            self.root,
            text=f"Готово! Размечено {n} / {len(self.samples)} сэмплов.\nРезультат: {self.labels_path}",
            font=("Helvetica", 16), pady=40,
        ).pack()


def main():
    parser = argparse.ArgumentParser(description="Blind-разметка baseline vs checkpoint по 400 сэмплам")
    parser.add_argument("--data-root", type=Path, required=True,
                         help="Папка, содержащая batch_00000/ (и, при необходимости, другие batch_*)")
    parser.add_argument("--labels-out", type=Path, default=Path("labels.csv"),
                         help="Куда писать CSV с разметкой (по умолчанию ./labels.csv). "
                              "При повторном запуске с тем же путём продолжает с первого неразмеченного сэмпла.")
    args = parser.parse_args()

    if not args.data_root.exists():
        raise SystemExit(f"Папка не найдена: {args.data_root}")

    samples = discover_samples(args.data_root)
    if not samples:
        raise SystemExit(f"В {args.data_root} не нашлось ни одного sample_* с полным набором из 3 картинок.")
    print(f"[labeler] Найдено сэмплов: {len(samples)}")

    root = tk.Tk()
    app = LabelerApp(root, args.data_root, args.labels_out, samples)
    root.mainloop()


if __name__ == "__main__":
    main()
