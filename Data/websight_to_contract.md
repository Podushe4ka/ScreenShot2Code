# Инструкция: `websight_to_contract.ipynb`

Конвертер **WebSight → формат контракта** (`SFT/DATA_FORMAT_CONTRACT.md`), задача
`task_type="drafting"` (скриншот → HTML). На выходе — датасет HuggingFace `datasets`,
готовый к `load_from_disk` на SFT-стороне.

## Что делает
Стриминг WebSight → гигиена → `save_to_disk` → приёмка §6 → токенный отчёт → handoff-памятка.

## Предпосылки
- Python-пакеты: `datasets`, `pillow`, `beautifulsoup4`, `transformers` (для токен-отчёта/фильтра).
- Запускать **из папки `Data/`** (конвертер делает `sys.path.append("analysis")` для `token_len.py`).
- Для **полного** боевого набора дополнительно нужны (пока НЕ в этом ноутбуке, см. «Границы»):
  Tailwind CLI/Node (precompile) и детерминированный рендерер eval (re-render).

## Конфиг-ручки (ячейка «1. Импорты и конфиг»)

| Параметр | Смоук (по умолч.) | Боевой | Что делает |
|---|---|---|---|
| `DATASET` | `HuggingFaceM4/WebSight` | тот же (v0.2, Tailwind) | источник; `mrm8488/WebSight_70k` — лёгкий v0.1 для отладки |
| `TARGET_COUNT` | 500 | 1 000–десятки тыс. | сколько **принятых** сэмплов собрать (стрим идёт, пока не наберём; при OOM на save — уменьшить) |
| `MAX_SCAN` | 20 000 | ↑ при низком выходе | предел просмотренных сырых (защита от бесконечного стрима) |
| `TARGET_SIZE` | `None` (авто) | **согласовать с eval**, напр. `(1280,720)` | единый размер скриншота (контракт §4a) |
| `SIZE_MODE` | `"filter"` | `"filter"` | `filter` — оставлять только `TARGET_SIZE`; `pad` — crop/pad (рвёт full-page) |
| `DROP_IMG` | `True` | `True` до рендерера | выкидывать сэмплы с `<img>` (относительные `src` битые) |
| `APPLY_PLACEHOLDERS` | `False` | `True` **только с ре-рендером** | заменить `<img>` на серый плейсхолдер |
| `MAX_TOKENS` | `None` | cap по коду (WebSight v0.2 ≈ **896**) | отсев длинных `target_html` токенайзером Qwen |
| `IMAGE_TOKEN_BUDGET` | 0 | визуальные токены/картинку | прибавляется к рекомендуемому `max_length` |
| `DECONTAM_DOMAINS` | `set()` | блоклист доменов eval | не пускать eval-страницы в train |
| `NEAR_DUP_HAMMING` | `None` | напр. `4` | резать почти-дубли картинок (average-hash; O(n²)) |
| `PRECOMPILE_TAILWIND` | `False` | см. «Границы» | хук; `True` без тулинга → `NotImplementedError` |
| `TOKENIZER_ID` | `Qwen/Qwen3-VL-8B-Instruct` | финальная модель | токенайзер для бюджета/отчёта |

## Как запускать
1. Открыть в `Data/`, выставить конфиг (боевой профиль — таблица выше).
2. Run All. Смотреть:
   - ячейку сборки — счётчики пропусков (`размер!=`, `с <img>`, `длинных`, `дублей`, `decontam`, `near-dup`);
   - **приёмку §6** — assert'ы должны пройти; печатает предупреждение про CDN-Tailwind / self-contained;
   - **токенный отчёт** — `median/p99/max` + `MAX_LENGTH_CODE`;
   - **handoff** — готовая памятка передачи.
3. Артефакт: `OUT_DIR` (по умолчанию `Data/websight_drafting_pilot/`). **В git не коммитится** (gitignore).

## Передача SFT-треку
- Данные идут **через диск/том, не через git**. Смонтировать `OUT_DIR` (контракт §7):
  `-v <OUT_DIR>:/data` → SFT читает `load_from_disk("/data")`.
- Из handoff отдать SFT число **`max_length`** = `MAX_LENGTH_CODE` + `IMAGE_TOKEN_BUDGET`.

### Согласование с реальным SFT-кодом (ветка `sft`)
- **Модель зафиксирована:** `Qwen/Qwen3.5-9B`, в конфиге **`max_length=8192`** (наш WebSight-код
  ~896 влезает с запасом). `TOKENIZER_ID` в конвертере для точного совпадения имеет смысл
  выставить в `Qwen/Qwen3.5-9B` (нужна совместимая `transformers`; по умолчанию стоит прокси
  Qwen3-VL-8B — счётчики близки).
- **`SFT/data/loader.py`** делает `load_from_disk` → фильтр `task_type=='drafting'` — наша схема
  потребляется как есть.
- **`SFT/train/formatting.py` `DRAFTING_PROMPT`**: «single self-contained HTML with **precompiled
  Tailwind**, replace images with **gray placeholder blocks**». → чтобы данные соответствовали
  промпту, **Tailwind-precompile и серые плейсхолдеры обязательны** для боевого набора (см.
  «Границы»); текущий CDN-Tailwind без картинок — только смоук для проверки пайплайна.

## Приёмка (контракт §6)
Грузится `load_from_disk`; поля по §2; картинки `Image()`-байты; **единый размер**; при
`DROP_IMG` — **нет висячих `<img>`**; 5 сэмплов глазами. Всё в ячейке «7. Приёмка».

## Границы (что этот ноутбук НЕ делает — нужен внешний тулинг/другие треки)
- **Tailwind precompile** (CDN → статический `<style>`): нужен Tailwind CLI/Node. Хук
  `precompile_tailwind()`. Без него артефакт **НЕ self-contained** (v0.2 тянет Tailwind с CDN) —
  ок для обучения, но не для детерминированного рендера eval.
- **Плейсхолдеры + ре-рендер**: чтобы вернуть **~56%** v0.2-сэмплов с картинками (сейчас их режет
  `DROP_IMG`), нужен рендерер eval (Этап 2) — отрендерить placeholder-версию в скриншот, чтобы
  картинка соответствовала коду. Хук `rerender_from_html()`.

**Итого:** текущий выход — валидный drafting-датасет для **пилота LoRA** (пайплайн end-to-end).
Полноценный **self-contained набор с картинками** разблокируется после Tailwind-precompile и
рендерера eval.
