# Инструкция: `websight_to_contract.ipynb`

Конвертер **WebSight → формат контракта** (`SFT/DATA_FORMAT_CONTRACT.md`), задача
`task_type="drafting"` (скриншот → HTML). На выходе — датасет HuggingFace `datasets`,
готовый к `load_from_disk` на SFT-стороне.

## Что делает
Стриминг WebSight → гигиена → `save_to_disk` → приёмка §6 → токенный отчёт → handoff-памятка.

## Предпосылки
- Смоук-профиль: `datasets`, `pillow`, `beautifulsoup4`, `transformers` (для токен-отчёта/фильтра).
- Запускать **из папки `Data/`** (конвертер делает `sys.path.append("analysis")` для `token_len.py`).
- **Боевой профиль** (self-contained + плейсхолдеры) дополнительно:
  - `pip install pytailwindcss` — даёт standalone-бинарь `tailwindcss` (precompile, **без Node**);
  - `pip install playwright && playwright install chromium` — ре-рендер скриншотов.
  - ⚠ Реализации `precompile_tailwind`/`rerender_from_html` написаны, но **не прогнаны** — проверить
    на первом боевом прогоне (per-page subprocess/рендер медленные — гонять на разумном `TARGET_COUNT`).

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

## Профили: смоук vs боевой

**Смоук (по умолчанию):** `APPLY_PLACEHOLDERS=False`, `PRECOMPILE_TAILWIND=False`, `DROP_IMG=True`.
Быстро, без внешних тулов. Даёт валидный drafting-датасет для пилота LoRA (проверка пайплайна),
но CDN-Tailwind и без картинок.

**Боевой (self-contained + плейсхолдеры):** `APPLY_PLACEHOLDERS=True`, `PRECOMPILE_TAILWIND=True`
(+ `pip install pytailwindcss playwright` и `playwright install chromium`). Пайплайн на сэмпл:
1. `replace_images_with_placeholder` — `<img>` → серый блок (совпадает с eval-конвенцией и промптом SFT);
2. `precompile_tailwind` — вкомпилировать Tailwind в `<style>`, убрать CDN → **self-contained**;
3. `rerender_from_html` — перерисовать скриншот из финального HTML (картинка следует за кодом, офлайн).
Так возвращаются ~56% сэмплов с картинками (их больше не режет `DROP_IMG`) и данные соответствуют
промпту SFT («precompiled Tailwind, gray placeholders»).

⚠ Реализации `precompile_tailwind`/`rerender_from_html` **написаны, но не прогонялись** — проверить
на первом боевом запуске; они per-page (subprocess + рендер) и медленные, гонять на разумном
`TARGET_COUNT`. По завершении вызвать `close_renderer()` (закрыть Playwright).

**decontam / масштаб** — уже в пайплайне: `DECONTAM_DOMAINS` (фильтр доменов; для синтетики WebSight
пусто) и count-driven сбор (`TARGET_COUNT`).
