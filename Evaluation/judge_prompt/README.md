# judge-prompt-lab

Отдельный, независимый от основного eval-пайплайна проект: разметка
baseline vs checkpoint человеком + подбор промпта/модели LLM-судьи так,
чтобы судья максимально совпадал с человеческой разметкой.

## Файлы

- `labeler.py` — GUI (Tkinter) для blind-разметки 400 сэмплов. Работает
  локально, без Docker/GPU.
- `split_data.py` — стратифицированный train(300)/test(100) split
  `labels.csv`.
- `sample_order.py` — общая логика "кто показывается первым" (baseline или
  checkpoint), используется и в `labeler.py`, и в `run_judge_eval.py`, чтобы
  порядок показа человеку и судье совпадал.
- `vllm_client.py` / `judge_client.py` — HTTP-клиент к vLLM и pairwise-judge
  логика со structured outputs.
- `prompts.py` — варианты промпта судьи (добавляй новые ключи по мере
  экспериментов).
- `run_judge_eval.py` — прогоняет judge-модель по train/test, печатает
  accuracy (% совпадения с человеком) и confusion matrix.
- `serve_judge.sh` / `Dockerfile` / `build.sh` / `run.sh` — Docker-обвязка
  для запуска judge-модели через vLLM.

## Шаг 1 — разметка (на своей машине, без Docker)

```bash
pip install pillow
python labeler.py --data-root /путь/к/папке/с/batch_00000 --labels-out labels.csv
```

Слева/справа показываются варианты "1" и "2" (без подписи baseline/checkpoint
— blind-разметка). `←`/`A` — вариант 1 лучше, `→`/`D` — вариант 2 лучше,
`Backspace` — отменить последний ответ. Прогресс сохраняется после каждого
ответа, можно останавливаться и продолжать позже тем же вызовом.

## Шаг 2 — train/test split

```bash
python split_data.py --labels labels.csv --train-out labels_train.csv --test-out labels_test.csv
```

300 в train, 100 в test, стратифицировано по классу-победителю.

## Шаг 3 — эксперименты с judge-моделью и промптом (train)

Собери образ:

```bash
./build.sh
```

Подбор промпта — гоняй сколько угодно раз на train, меняя `--prompt-key`
(добавляя новые варианты в `prompts.py`) и `MODEL_JUDGE`:

```bash
MODEL_JUDGE=Qwen/Qwen3.5-2B ./run.sh \
    --data-root-host /путь/к/папке/с/batch_00000 \
    --labels-host ./labels_train.csv \
    --prompt-key v1_baseline

MODEL_JUDGE=Qwen/Qwen3.5-4B GPU_JUDGE=1 ./run.sh \
    --data-root-host /путь/к/папке/с/batch_00000 \
    --labels-host ./labels_train.csv \
    --prompt-key v2_criteria_list \
    --out-json /app/output/v2_4b_train.json
```

MODEL_JUDGE=Qwen/Qwen3.5-4B GPU_JUDGE=1 GPU_MEMORY_UTILIZATION=0.85 MAX_MODEL_LEN=24000 ./run.sh \
    --data-root-host ./_work/ \
    --labels-host ./labels_train.csv \
    --prompt-key v1_baseline \
    --out-json /app/output/v1_4b_train.json

Для линейки 3.5 меняй `MODEL_JUDGE` между: `Qwen/Qwen3.5-2B`,
`Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.5-27B` (подставь точные
HF-имена, если они отличаются).

Скрипт печатает accuracy и confusion matrix — по ним сравниваешь модели и
формулировки промпта между собой.

## Шаг 4 — финальный замер на test

Один раз, когда промпт и модель зафиксированы:

```bash
MODEL_JUDGE=Qwen/Qwen3.5-9B ./run.sh \
    --data-root-host /путь/к/папке/с/batch_00000 \
    --labels-host ./labels_test.csv \
    --prompt-key v2_criteria_list \
    --out-json /app/output/final_test.json
```

## Без Docker (если vLLM уже установлен локально)

```bash
vllm serve Qwen/Qwen3.5-4B --port 8001 --trust-remote-code --limit-mm-per-prompt '{"image": 3}'

# в другом терминале:
python run_judge_eval.py \
    --data-root /путь/к/папке/с/batch_00000 \
    --labels labels_train.csv \
    --prompt-key v1_baseline \
    --judge-url http://127.0.0.1:8001 \
    --judge-model Qwen/Qwen3.5-4B
```
