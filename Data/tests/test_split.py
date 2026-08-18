"""Разрезы train/validation: главная тихая ошибка — утечка между сплитами.

Почти-копия в обоих сплитах занижает `eval_loss`, и заметить это по метрикам нельзя —
кривая выглядит просто «хорошей». Поэтому оба разреза проверяются приёмкой, а не глазами.
"""
import os
import subprocess
import sys

import pytest

from conftest import load

datasets = pytest.importorskip("datasets")

DATA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAKE_SPLIT = os.path.join(DATA, "converters", "make_split.py")


def sample(i, page=None, task="drafting"):
    return {"task_type": task, "images": [], "current_html": "",
            "target_html": f"<html>{i}</html>", "instruction": "",
            "page_id": page or f"p{i}"}


def test_make_split_no_overlap(tmp_path):
    ds = datasets.Dataset.from_list([sample(i) for i in range(200)])
    inp = tmp_path / "in"; out = tmp_path / "out"
    ds.save_to_disk(str(inp))
    r = subprocess.run([sys.executable, MAKE_SPLIT, str(inp), str(out), "--val-frac", "0.1"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    dd = datasets.load_from_disk(str(out))
    assert set(dd) == {"train", "validation"}
    assert len(dd["validation"]) == 20
    assert not set(dd["train"]["target_html"]) & set(dd["validation"]["target_html"])


def test_make_split_refuses_datasetdict(tmp_path):
    """Повторный разрез уже разрезанного набора — почти всегда ошибка оператора."""
    dd = datasets.DatasetDict({"train": datasets.Dataset.from_list([sample(i) for i in range(4)]),
                               "validation": datasets.Dataset.from_list([sample(9)])})
    inp = tmp_path / "in"; dd.save_to_disk(str(inp))
    r = subprocess.run([sys.executable, MAKE_SPLIT, str(inp), str(tmp_path / "out")],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "DatasetDict" in r.stderr


def test_make_split_refuses_tiny_input(tmp_path):
    inp = tmp_path / "in"
    datasets.Dataset.from_list([sample(0)]).save_to_disk(str(inp))
    r = subprocess.run([sys.executable, MAKE_SPLIT, str(inp), str(tmp_path / "out")],
                       capture_output=True, text=True)
    assert r.returncode != 0


def test_grouped_split_keeps_page_whole():
    """Из одной страницы выходит до семи сэмплов с одним target_html: разрезать их
    случайно — значит показать модели валидацию на обучении."""
    grouped = load("synth", "make_split_grouped")
    rows = []
    for p in range(60):
        rows.append(sample(p, page=f"page{p}", task="drafting"))
        for k in range(3):
            rows.append(sample(f"{p}_{k}", page=f"page{p}", task="editing"))
        rows.append(sample(f"{p}_d", page=f"page{p}", task="polishing"))
    dd = grouped.grouped_split(datasets.Dataset.from_list(rows))
    grouped.assert_no_leak(dd)
    assert not set(dd["train"]["page_id"]) & set(dd["validation"]["page_id"])


def test_grouped_split_keeps_every_task_in_validation():
    """Стратификация по task_type: редкая задача не должна исчезнуть из валидации."""
    grouped = load("synth", "make_split_grouped")
    rows = []
    for p in range(60):
        rows.append(sample(p, page=f"page{p}", task="drafting"))
        rows.append(sample(f"{p}_e", page=f"page{p}", task="editing"))
    for p in range(60, 66):                      # редкая задача — 6 страниц из 66
        rows.append(sample(f"{p}_r", page=f"page{p}", task="polishing"))
    dd = grouped.grouped_split(datasets.Dataset.from_list(rows))
    grouped.assert_no_leak(dd)
    assert set(dd["validation"]["task_type"]) == {"drafting", "editing", "polishing"}
