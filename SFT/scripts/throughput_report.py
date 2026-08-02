"""Сводка по логам throughput_matrix.sh.

Считает МАРЖИНАЛЬНУЮ скорость (разница между соседними шагами), а не
накопительную: `train_tokens_per_second` в логе усредняет всё вместе с первым
шагом, а тот втрое дороже из-за компиляции Triton-ядер.

    python -m scripts.throughput_report throughput-logs
"""

import argparse
import ast
import re
from pathlib import Path

A100_BF16_TFLOPS = 312.0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logdir", nargs="?", default="throughput-logs")
    ap.add_argument("--params-billions", type=float, default=4.0,
                    help="параметров в модели, млрд (см. scripts.model_facts)")
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--skip-warmup", type=int, default=2,
                    help="сколько первых шагов выбросить как прогрев")
    return ap.parse_args(argv)


_DICT_RE = re.compile(r"\{'loss'.*?\}")


def read_steps(path: Path) -> list[dict]:
    """Все строки-логи шагов из файла, по порядку."""
    steps = []
    for line in path.read_text(errors="replace").splitlines():
        m = _DICT_RE.search(line)
        if not m:
            continue
        try:
            d = ast.literal_eval(m.group(0))
        except (ValueError, SyntaxError):
            continue
        if "train_runtime" in d and "num_input_tokens_seen" in d:
            steps.append({
                "runtime": float(d["train_runtime"]),
                "tokens": int(d["num_input_tokens_seen"]),
                "loss": float(d["loss"]),
            })
    return steps


def failure_reason(path: Path) -> str | None:
    text = path.read_text(errors="replace")
    for needle, label in (
        ("OutOfMemoryError", "OOM"),
        ("out of memory", "OOM"),
        ("Error building extension", "сборка расширения"),
        ("Traceback (most recent call last)", "исключение"),
    ):
        if needle in text:
            return label
    return None


def summarize(path: Path, skip: int) -> dict:
    steps = read_steps(path)
    row = {"name": path.stem, "n_steps": len(steps)}
    if len(steps) < skip + 2:
        row["status"] = failure_reason(path) or f"мало шагов ({len(steps)})"
        return row

    tail = steps[skip:]
    d_runtime = tail[-1]["runtime"] - tail[0]["runtime"]
    d_tokens = tail[-1]["tokens"] - tail[0]["tokens"]
    n = len(tail) - 1
    if d_runtime <= 0 or n <= 0:
        row["status"] = "нулевой интервал"
        return row

    row["status"] = "ok"
    row["s_per_step"] = d_runtime / n
    row["tokens_per_s"] = d_tokens / d_runtime
    row["loss"] = tail[-1]["loss"]
    return row


def main(argv=None):
    args = parse_args(argv)
    logdir = Path(args.logdir)
    # с подчёркивания начинаются служебные прогоны (прогрев) — не показываем
    logs = sorted(p for p in logdir.glob("*.log") if not p.stem.startswith("_"))
    if not logs:
        raise SystemExit(f"в {logdir} нет .log файлов")

    rows = [summarize(p, args.skip_warmup) for p in logs]
    ok = [r for r in rows if r["status"] == "ok"]
    ok.sort(key=lambda r: -r["tokens_per_s"])

    flops_per_token = 6 * args.params_billions * 1e9
    peak = A100_BF16_TFLOPS * args.gpus

    print(f"{'эксперимент':<22} {'с/шаг':>9} {'ток/с':>9} {'TFLOPS':>8} {'MFU':>7} {'loss':>7}")
    print("-" * 66)
    for r in ok:
        tflops = flops_per_token * r["tokens_per_s"] / 1e12
        print(f"{r['name']:<22} {r['s_per_step']:>9.1f} {r['tokens_per_s']:>9.0f} "
              f"{tflops:>8.1f} {tflops / peak:>6.1%} {r['loss']:>7.4f}")

    bad = [r for r in rows if r["status"] != "ok"]
    if bad:
        print("\nне замерены:")
        for r in bad:
            print(f"  {r['name']:<22} {r['status']}")

    if ok:
        best, base = ok[0], next((r for r in ok if r["name"] == "base_accum8"), None)
        print(f"\nлучший: {best['name']} — {best['tokens_per_s']:.0f} ток/с")
        if base and base is not best:
            gain = best["tokens_per_s"] / base["tokens_per_s"] - 1
            print(f"против base_accum8: +{gain:.1%}")

    print(f"\nMFU считается от 6N флопс/токен при N={args.params_billions} млрд "
          f"и пике {peak:.0f} TFLOPS ({args.gpus}xA100 bf16).")
    print("Уточните N через: python -m scripts.model_facts --load-weights")


if __name__ == "__main__":
    main()
