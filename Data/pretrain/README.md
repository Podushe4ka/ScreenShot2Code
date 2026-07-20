# Pretrain — подготовка данных

Единый **стриминговый** микс корпусов `image → HTML` для continual pretraining
(стадия pretrain из UI2Code^N): широкое шумное знание vision→code до SFT.

## Файл
- [`pretrain_stream.ipynb`](pretrain_stream.ipynb) — собирает несколько датасетов в один
  `IterableDataset` **без скачивания** и отдаёт тренеру.

## Идея (кратко)
```
каждый источник:  load_dataset(streaming=True)
                       │  .map(normalize)     → общая схема {image, html, source}
                       │  .filter(...)        → отсев пустых/битых
                       │  .cast_column(image) → единый тип
                       ▼
        interleave_datasets(probabilities)    → микс потоков по весам
                       │  .shuffle(buffer)     → потоковый шафл
                       ▼
                     mix  (IterableDataset → тренеру)
```
- **Без скачивания**: `streaming=True` тянет шарды по мере итерации.
- **Единая выдача**: все источники приводятся к `{image, html, source}`.
- **Микс**: `interleave_datasets` берёт сэмпл из источника по вероятности (веса в конфиге).

## Источники (по умолчанию)
- **WebSight v0.2** (синтетика, Tailwind) — вес 0.4.
- **WebCode2M purified** (реальные, де-блоб) — вес 0.4.
- **Web2Code** — выключен (проверить точные поля instruction-формата перед включением).

## Запуск
1. Открыть в `Data/`; при необходимости `HF_TOKEN` в окружении.
2. Настроить `SOURCES`/веса/`SHUFFLE_BUFFER` в ячейке «1. Конфиг».
3. Run All → ячейка «5. Проверка выдачи» покажет распределение источников и пример записи.
4. `mix` передать в тренер (стриминг: `max_steps`, не эпохи).

## Отличие от drafting-конвертера
Претрейн — **сырые** пары `image+html` (широкое знание), а не контрактная task-схема.
Форматирование и гигиена под конкретную задачу — на SFT-стадии (`websight_to_contract`).

## Проверить перед боевым прогоном
- Точные поля Web2Code; доступность/гейтинг датасетов.
- Что `interleave_datasets` не спотыкается на `features` (выровнено `remove_columns` + `cast_column`).
- Баланс весов и `SHUFFLE_BUFFER` (память ↔ перемешивание).
- При нужде — фильтр по бюджету токенов (переиспользовать `../analysis/token_len.py`), но в потоке дорого.
