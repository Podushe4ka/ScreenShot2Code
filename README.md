# ScreenShot2Code

Репозиторий проекта Летней Академии ML Яндекса

Руководитель: Данил Кашин

Задача: по скриншоту страницы получить HTML, который её воспроизводит.
Базовая модель — Qwen3.5-4B (VLM), метрика — Design2Code.

## С чего начать

| документ | о чём |
|---|---|
| [`docs/STRUCTURE.md`](docs/STRUCTURE.md) | карта репозитория и общего диска, грабли окружения — **читать первым** |
| [`docs/RESULTS.md`](docs/RESULTS.md) | все модели, гиперпараметры и скоры в одной таблице + описание бенчмарков |
| [`docs/experiments/`](docs/experiments) | что уже проверено и чем закончилось |
| [`docs/storage_inventory.md`](docs/storage_inventory.md) | что занимает место на `/mnt/storage-1` и что удалено |
| [`experiments/README.md`](experiments/README.md) | как запускать прогоны |
| [`SFT/DATA_FORMAT_CONTRACT.md`](SFT/DATA_FORMAT_CONTRACT.md) | контракт формата датасета |

## Раскладка

```
SFT/          обучение (torchrun + TRL + DeepSpeed), образ `sft`
Evaluation/   бенч Design2Code (vLLM + playwright + CLIP), образ `design2code-bench`
Data/         сборка датасетов и анализ
experiments/  оркестраторы прогонов (.sh), запускаются на хосте
docs/         документация
RL/           задел, пока пусто
```

Всё, что на серверах, гоняется **только через docker** — подробности и
подводные камни в [`docs/STRUCTURE.md`](docs/STRUCTURE.md).
