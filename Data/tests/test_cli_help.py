"""`--help` каждого CLI обязан работать.

Не формальность: `%` в help-строке argparse форматирует через оператор `%`, одиночный
процент роняет `--help` (лечилось коммитом b8d7300), а любая опечатка в прологе импортов
проявляется здесь же — до того, как скрипт запустят на многочасовом прогоне.
"""
import os
import subprocess
import sys

import pytest

DATA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPTS = [
    "converters/make_split.py",
    "converters/complexity/score.py",
    "converters/complexity/select_by_complexity.py",
    "converters/mix/build_mix.py",
    "converters/mix/export_examples.py",
    "converters/synth/build.py",
    "converters/synth/pack.py",
    "converters/synth/mutate.py",
    "converters/synth/degrade.py",
    "converters/synth/contact_sheet.py",
    "converters/webcode2m/convert_parallel.py",
    "converters/webcode2m/convert_raw.py",
    "converters/webcode2m/convert_candidates.py",
    "converters/webcode2m/scan_complex.py",
    "converters/webcode2m/export_htmls.py",
    "converters/websight/convert_parallel.py",
    "converters/websight/filter_by_height.py",
    "converters/websight/view_arrow.py",
    "converters/webui/convert_parallel.py",
    "converters/webui/compare_raw.py",
    "converters/webui/fetch_columns.py",
    "eda/tools/pixel_budget.py",
    "eda/tools/plot_hist.py",
    "eda/tools/design2code_study.py",
    "eda/tools/webui_clean_eda.py",
]


@pytest.mark.parametrize("rel", SCRIPTS, ids=[s.split("/")[-1] for s in SCRIPTS])
def test_help_runs(rel):
    path = os.path.join(DATA, rel)
    assert os.path.exists(path), f"{rel} нет — список в тесте устарел"
    # cwd = каталог скрипта: так их запускают, и так работают импорты соседей.
    r = subprocess.run([sys.executable, os.path.basename(path), "--help"],
                       cwd=os.path.dirname(path), capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, f"{rel}:\n{r.stdout[-800:]}\n{r.stderr[-800:]}"
    assert "usage" in r.stdout.lower()


def test_script_list_is_complete():
    """Появился новый CLI — добавь его сюда. Тест ловит забытые скрипты."""
    found = set()
    for root in ("converters", "eda/tools"):
        for dirpath, _, files in os.walk(os.path.join(DATA, root)):
            if "__pycache__" in dirpath or os.path.basename(dirpath) == "common":
                continue
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                p = os.path.join(dirpath, fn)
                src = open(p, encoding="utf-8").read()
                if "argparse.ArgumentParser" in src and "__main__" in src:
                    found.add(os.path.relpath(p, DATA))
    missing = found - set(SCRIPTS)
    assert not missing, f"CLI без теста на --help: {sorted(missing)}"
