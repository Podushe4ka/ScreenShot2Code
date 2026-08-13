"""
sample_order.py — общая для labeler.py и run_judge_eval.py логика: по имени
сэмпла детерминированно решает, показывать ли baseline или checkpoint
первым ("Candidate A" / левая картинка). Вынесено в отдельный модуль без
тяжёлых зависимостей (только hashlib), чтобы run_judge_eval.py (крутится в
Docker-контейнере вместе с vLLM) не тянул tkinter из labeler.py, и чтобы
порядок показа гарантированно совпадал между тем, что видел человек при
разметке, и тем, что видит judge-модель при прогоне на train/test.
"""

import hashlib


def swapped_for_sample(sample_id: str) -> bool:
    h = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 2 == 1
