"""Барьеры отбора не должны молчать.

Два прежних поведения были тихими и меняли состав набора:
  * `--max-total-tokens` при отсутствии `tokens_total` подставлял `tokens_code`, то есть
    сравнивал код-без-картинки с бюджетом на код+картинку (разница до 2048 токенов);
  * `--density-min` (порог откалиброван на рендере) на статическом файле признаков
    не находил своего ключа и не отсекал ничего.
Оба теперь — явная ошибка с подсказкой. Тесты держат именно это.
"""
import json
import os
import subprocess
import sys

import pytest

SELECT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "converters", "complexity", "select_by_complexity.py")


def write(tmp_path, rows, name="feats.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return str(p)


def run(args):
    return subprocess.run([sys.executable, SELECT] + args, capture_output=True, text=True)


def rendered_row(i, **kw):
    row = {"id": f"r{i}", "complexity": 0.1 * i, "features_mode": "rendered",
           "render_ok": True, "nodes": 100 + i, "tokens_code": 1000,
           "density_visible": 0.05, "tokens_total": 3000, "w": 1280, "h": 900}
    row.update(kw)
    return row


def static_row(i, **kw):
    row = {"id": f"s{i}", "complexity": 0.1 * i, "features_mode": "static",
           "render_ok": False, "nodes": 100 + i, "tokens_code": 1000,
           "density_static": 0.1}
    row.update(kw)
    return row


def test_max_total_tokens_without_field_is_an_error(tmp_path):
    src = write(tmp_path, [static_row(i) for i in range(1, 9)])
    r = run([src, "--out", str(tmp_path / "o.jsonl"), "-n", "3", "--max-total-tokens", "16224"])
    assert r.returncode != 0
    assert "tokens_total" in r.stderr
    assert "--max-tokens" in r.stderr           # подсказка, чем это заменить


def test_density_min_on_static_features_is_an_error(tmp_path):
    src = write(tmp_path, [static_row(i) for i in range(1, 9)])
    r = run([src, "--out", str(tmp_path / "o.jsonl"), "-n", "3", "--density-min", "0.01"])
    assert r.returncode != 0
    assert "density_visible" in r.stderr
    assert "rendered" in r.stderr               # подсказка, как пересчитать


def test_density_min_static_works_on_static_features(tmp_path):
    """Для статических признаков есть свой флаг — со своей шкалой."""
    rows = [static_row(i) for i in range(1, 9)] + [static_row(99, density_static=0.0001)]
    src = write(tmp_path, rows)
    out = tmp_path / "o.jsonl"
    r = run([src, "--out", str(out), "-n", "5", "--density-min-static", "0.01"])
    assert r.returncode == 0, r.stderr
    picked = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert all(p["id"] != "s99" for p in picked)


def test_barriers_pass_on_rendered_features(tmp_path):
    src = write(tmp_path, [rendered_row(i) for i in range(1, 13)])
    out = tmp_path / "o.jsonl"
    r = run([src, "--out", str(out), "-n", "4",
             "--density-min", "0.01", "--max-total-tokens", "16224"])
    assert r.returncode == 0, r.stderr
    assert len(out.read_text(encoding="utf-8").splitlines()) == 4


def test_over_budget_pages_are_dropped(tmp_path):
    """Страница с суммой выше бюджета не должна попасть в набор."""
    rows = [rendered_row(i) for i in range(1, 11)]
    rows.append(rendered_row(99, tokens_total=99_000, complexity=0.99))
    src = write(tmp_path, rows)
    out = tmp_path / "o.jsonl"
    r = run([src, "--out", str(out), "-n", "5", "--max-total-tokens", "16224"])
    assert r.returncode == 0, r.stderr
    picked = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert all(p["id"] != "r99" for p in picked)
