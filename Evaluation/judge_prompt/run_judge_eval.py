"""
run_judge_eval.py — прогоняет judge-модель (уже запущенную через
serve_judge.sh) по train или test выборке с заданным промптом и печатает
процент совпадения с человеческой разметкой (accuracy) + confusion matrix.

Типичный цикл подбора промпта:
    1. Поднять judge-сервер с нужной моделью (serve_judge.sh, см. README).
    2. python run_judge_eval.py --labels labels_train.csv --prompt-key v1_baseline
    3. Посмотреть accuracy/confusion matrix, поправить промпт в prompts.py
       (или добавить новый ключ), повторить п.2 сколько угодно раз — train
       можно использовать свободно.
    4. Когда промпт зафиксирован — ОДИН финальный прогон на test:
       python run_judge_eval.py --labels labels_test.csv --prompt-key <финальный>

Не считает ничего, кроме accuracy и confusion matrix — задача не в
ранжировании кандидатов по метрикам качества картинки (это отдельно есть в
основном eval-пайплайне через Design2Code-метрики), а в том, насколько
судья согласен с человеком.
"""

import argparse
import csv
import json
import time
from pathlib import Path

from judge_client import JudgeError, judge_one
from prompts import get_prompt
from sample_order import swapped_for_sample
from vllm_client import VLLMClient


def read_labels(path: Path) -> list[dict]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_eval(
    client: VLLMClient,
    data_root: Path,
    rows: list[dict],
    prompt_text: str,
    max_tokens: int,
) -> dict:
    n_correct = 0
    n_error = 0
    # confusion[human][judge] = count
    confusion = {"baseline": {"baseline": 0, "checkpoint": 0},
                 "checkpoint": {"baseline": 0, "checkpoint": 0}}
    per_sample_results = []

    total = len(rows)
    start = time.monotonic()
    for i, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        human_winner = row["winner"]
        sample_dir = data_root / sample_id
        swapped = swapped_for_sample(sample_id)

        try:
            result = judge_one(client, prompt_text, sample_dir, swapped, max_tokens=max_tokens)
            judge_winner = result["winner_model"]
            confusion[human_winner][judge_winner] += 1
            correct = judge_winner == human_winner
            n_correct += int(correct)
            per_sample_results.append({
                "sample_id": sample_id,
                "human_winner": human_winner,
                "judge_winner": judge_winner,
                "correct": correct,
            })
        except JudgeError as e:
            n_error += 1
            print(f"[run_judge_eval] {sample_id}: ошибка судьи — {e}")
            per_sample_results.append({
                "sample_id": sample_id,
                "human_winner": human_winner,
                "judge_winner": None,
                "correct": False,
                "error": str(e),
            })

        if i % 20 == 0 or i == total:
            elapsed = time.monotonic() - start
            print(f"[run_judge_eval] {i}/{total} обработано ({elapsed:.0f}с)")

    n_scored = total - n_error
    accuracy = n_correct / n_scored if n_scored > 0 else 0.0

    return {
        "n_total": total,
        "n_scored": n_scored,
        "n_error": n_error,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "confusion": confusion,
        "per_sample": per_sample_results,
    }


def print_report(summary: dict, prompt_key: str, model_name: str):
    print()
    print("=" * 60)
    print(f"Модель судьи : {model_name}")
    print(f"Промпт       : {prompt_key}")
    print(f"Всего сэмплов: {summary['n_total']}  (успешно обработано: {summary['n_scored']}, ошибок: {summary['n_error']})")
    print(f"Accuracy (совпадение с человеком): {summary['accuracy']:.1%}  ({summary['n_correct']}/{summary['n_scored']})")
    print()
    print("Confusion matrix (строки = что сказал человек, столбцы = что сказал судья):")
    c = summary["confusion"]
    print(f"{'':>18}{'judge=baseline':>18}{'judge=checkpoint':>18}")
    for human_cls in ("baseline", "checkpoint"):
        print(f"{'human=' + human_cls:>18}{c[human_cls]['baseline']:>18}{c[human_cls]['checkpoint']:>18}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Прогон judge-модели на размеченной выборке")
    parser.add_argument("--data-root", type=Path, required=True,
                         help="Та же папка, что передавалась в labeler.py (содержит batch_00000/...)")
    parser.add_argument("--labels", type=Path, required=True,
                         help="labels_train.csv или labels_test.csv (результат split_data.py)")
    parser.add_argument("--prompt-key", type=str, required=True,
                         help="Ключ промпта из prompts.py, например v1_baseline")
    parser.add_argument("--judge-url", type=str, default="http://127.0.0.1:8001",
                         help="Адрес vLLM-сервера судьи (см. serve_judge.sh)")
    parser.add_argument("--judge-model", type=str, required=True,
                         help="Имя модели, как оно передано в vllm serve (напр. Qwen/Qwen3.5-4B)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--out-json", type=Path, default=None,
                         help="Опционально: куда сохранить полный отчёт (включая per-sample результаты) в JSON")
    args = parser.parse_args()

    prompt_text = get_prompt(args.prompt_key)
    rows = read_labels(args.labels)
    print(f"[run_judge_eval] Загружено {len(rows)} размеченных сэмплов из {args.labels}")

    client = VLLMClient(args.judge_url, args.judge_model)
    print(f"[run_judge_eval] Жду готовности сервера {args.judge_url} ...")
    client.wait_until_ready()
    print("[run_judge_eval] Сервер готов, начинаю прогон.")

    summary = run_eval(client, args.data_root, rows, prompt_text, args.max_tokens)
    print_report(summary, args.prompt_key, args.judge_model)

    if args.out_json:
        report = {
            "judge_model": args.judge_model,
            "prompt_key": args.prompt_key,
            "prompt_text": prompt_text,
            "labels_file": str(args.labels),
            **summary,
        }
        args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[run_judge_eval] Полный отчёт сохранён в {args.out_json}")


if __name__ == "__main__":
    main()
