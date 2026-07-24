# Data-трек — план работ

Рабочий документ Data-трека проекта Screenshot2Code (дообучение Qwen-VL под
генерацию/редактирование UI по скриншотам). Методический ориентир — статья
**UI2Code^N** (`papers/UI2CodeN_2511.08195.pdf`): пайплайн pretrain → SFT → RL,
задачи drafting / polishing / editing.

## Навигация по папке

| Путь | Что внутри |
|---|---|
| [`PLAN.md`](PLAN.md) | Этот файл — план работ и дорожная карта |
| [`list_data.md`](list_data.md) | Каталог датасетов: train-кандидаты, бенчмарки, смежное |
| [`drafting/`](drafting/) | Drafting-конвертер (`convert_lib` + notebook + parallel + Docker + `view_arrow`) — ветка `data/drafting` |
| [`pretrain/`](pretrain/) | Стриминговый претрейн-микс датасетов (ветка `data/pretrain`) |
| [`analysis/`](analysis/) | Этап 0 — EDA корпусов (ноутбуки + заметки + методика метрик) |
| [`papers/`](papers/) | PDF статей ко всем датасетам и методу ([индекс](papers/README.md)) |

**Ветки Data-трека** (все = `data/analysis` + свои файлы, различаются только вперёд):
`data/analysis` (общий baseline: анализ + токенные метрики), `data/drafting`
(+ drafting-конвертер), `data/pretrain` (+ претрейн-микс).

Формат передачи данных в SFT-трек — контракт `../SFT/DATA_FORMAT_CONTRACT.md`
(ветка `sft`).

**Статус:** ✅ Этап 0 (разведка + токенные метрики, перепрогон на v0.2) · ▶ Этап 1
(drafting-конвертер `drafting/` — production работает: параллельный батч через Docker,
self-contained, ~5k за пару минут) · ⏳ Этапы 2–4 (рендерер, синтетика, масштаб).
Работа по drafting — на ветке `data/drafting`.

---

## 0. Где мы сейчас — Этап 0 (разведка) ЗАВЕРШЁН

- EDA трёх train-кандидатов по 7 метрикам — `analysis/` (`webcode2m.ipynb`,
  `websight.ipynb`, `webui.ipynb`, методика — `analysis/required_data.md`,
  результаты/особенности — `analysis/dataset_notes.md`).
- Каталог источников — `list_data.md` (train-кандидаты + бенчмарки + смежное).
- Статьи по всем датасетам и методу — `papers/` (индекс в `papers/README.md`).

**Вывод по объёму данных.** UI-специфичная часть претрейна UI2Code^N — это по
сути WebCode2M + WebSight (те же датасеты, что у нас). Наши ~3–6M реальных пар
(WebCode2M 2.5M + WebSight 2M + Web2Code 1.18M) — в рабочем диапазоне. Дефицит
**не в объёме**, а в (а) детерминированном рендер+reward-пайплайне и
(б) генерации SFT/edit/polish данных.

**Токенные результаты (метрика 3 — готово).** Длина кода считается токенайзером
Qwen (`analysis/token_len.py`). Рабочий `max_length` кода: **WebSight v0.2 ~896**
(p99=851; было 768 на v0.1 — v0.2/Tailwind многословнее), **WebCode2M ~9 920** (p99),
**WebUI 8 000 @ cap 8k** (теряем ~24% — тяжёлый хвост из инлайнового CSS дизайн-систем,
не base64). Числа посчитаны **без визуальных токенов** (`IMAGE_TOKEN_BUDGET=0`) —
реальный бюджет выше на ~1.1–1.2k токенов/картинку при 1280×720. Таблица —
`analysis/dataset_notes.md`. Отсюда правило гигиены для реальных данных: **де-блоб
(data-URI) + фильтр по токенам** (отсечка = `max_length`). WebUI для MVP не берём.

---

## 1. Что от нас ждёт SFT-трек (интерфейс = контракт)

- **Контракт формата:** `SFT/DATA_FORMAT_CONTRACT.md` (владелец — SFT-трек,
  сейчас лежит в ветке `sft`; приедет в `main` при мёрдже). Читать как истину
  по формату передачи.
- **MVP:** отдать SFT-треку **только `task_type=="drafting"`**: `save_to_disk`,
  картинки встроены через `Image()` (байты, не пути), единая схема сэмпла.
- **Состояние SFT-трека (обновлено по ветке `sft`):** модель зафиксирована —
  **`Qwen/Qwen3.5-9B`**, **`max_length=8192`**, LoRA r16, flash-attn2, DeepSpeed ZeRO-2.
  Тренировочный код написан: `data/loader.py` (`load_from_disk` → фильтр `task_type=='drafting'`),
  `train/formatting.py` (`to_message` + `collate_fn` с маскировкой лейблов), `train/train_sft.py`,
  скрипты `overfit20.py`/`smoke_test.py`. **SFT ждёт только данные** — наш drafting-датасет.
- **⚠ Промпт SFT диктует формат таргета:** `DRAFTING_PROMPT` = «single self-contained HTML with
  **precompiled Tailwind**, replace images with **gray placeholder blocks**». Значит для
  согласованности данных с промптом **Tailwind-precompile + серые плейсхолдеры обязательны**
  (наш смоук-пилот на CDN-Tailwind без картинок — только для проверки пайплайна).

---

## 1a. Что уже решил eval-трек (ветка `Eval`, `Evaluation/Experiments.ipynb`)

- **Плейсхолдеры — конвенция зафиксирована, использовать РОВНО её:** каждый `<img>`
  меняется на `<div>` с классом `bg-gray-300 w-full h-48 rounded` и инлайн-стилем
  `background-color:#d1d5db;width:100%;height:12rem;border-radius:0.5rem;display:block;`,
  одинаково на эталоне и предсказании, до рендера (через BeautifulSoup). Инлайн-стиль
  даёт рендер даже без скомпилированного Tailwind.
- **Рендерер + метрики уже есть:** официальный Design2Code (`screenshot_single.py`,
  Playwright+Chromium) + `visual_eval_v3_multi` (block / text / position / color / clip).
  Это готовый рендер+reward — переиспользовать для polishing/RL (Этап 2), не плодить свой.
- **Бенч eval:** `SALT-NLP/Design2Code-hf` → Design2Code **никогда в train** (декотаминация).
- **Всплывшие расхождения (вынесены в §4):** eval-baseline = `Qwen2.5-VL-3B-Instruct`;
  eval-промпт просит plain HTML с inline `<style>`, а НЕ Tailwind.
- ⚠ **Про пилот и Tailwind:** текущий пилот собран на WebSight **v0.1** (plain HTML +
  inline `<style>`) — он *случайно* совпадает с eval-промптом, конфликт «Tailwind vs
  inline» на нём не проявляется. После миграции конвертера на **v0.2** (Tailwind)
  таргеты станут Tailwind → конфликт снова актуален, а **Tailwind-precompile становится
  обязательным** (иначе utility-классы у eval без скомпилированного CSS не отрендерятся).

---

## 2. Ключевые проектные решения (из UI2Code^N — обоснование в статье)

1. **Tailwind фиксируется на SFT-таргетах, НЕ в претрейне.** Претрейн намеренно
   мешает разные формы CSS (реальный краул + `<style>` из WebCode2M + Tailwind
   из WebSight) — это «широкое шумное знание». Формат вывода задаёт SFT.
2. **SFT / editing / polishing данные генерятся заново (reverse construction),
   а НЕ выбираются из претрейн-корпуса.** Три стадии — три независимых источника.
   - drafting SFT: LLM пишет HTML (ответ) → рендер → скриншот (запрос).
   - editing «добавь X»: разворот пар удаления (полный → урезанный).
   - polishing: хороший код → испорченный рендер/ранние генерации →
     (target-скрин + плохой код + его рендер → хороший код).
3. **Объёмы-якоря (UI2Code^N-9B):** pretrain ~20M (вкл. общие VLM-задачи),
   SFT **80K** на все три задачи, RL **42K** промптов, polishing глубина
   `N ~ U[1,4]` (насыщается к N≈3).

---

## 3. Дорожная карта Data-трека

### Этап 1 — Drafting-датасет по контракту  ◀ ТЕКУЩИЙ ПРИОРИТЕТ (разблокирует SFT MVP)
Конвертер — папка `drafting/` (логика в `convert_lib.py`; `convert.ipynb` интерактив,
`convert_parallel.py` батч, `Dockerfile`, инструкция `drafting/README.md`).
Источник: **WebSight v0.2** (`HuggingFaceM4/WebSight`, ~1.92M) — Tailwind, чистая синтетика.

Готово:
- [x] конвертер WebSight v0.2 → схема контракта (`Dataset.from_list` + `Features`), `save_to_disk`;
- [x] единый размер скриншота (`TARGET_SIZE`/`SIZE_MODE`, §4a); отсев висячих `<img>` (`DROP_IMG`);
- [x] дедуп (SHA1) + **near-dup** (average-hash) + **декотаминация** по блоклисту доменов;
- [x] фильтр по бюджету токенов (`MAX_TOKENS`, `token_len.count_tokens`); переизмерено на v0.2 (p99=851 → `max_length` 896);
- [x] приёмка §6 (`load_from_disk`, поля §2, единый размер, нет `<img>`) + токенный отчёт + handoff-памятка.

Боевой профиль (`APPLY_PLACEHOLDERS=True` + `PRECOMPILE_TAILWIND=True`) — **реализован, но не прогнан**:
- [x] **Tailwind precompile** (`precompile_tailwind` через standalone `pytailwindcss`, без Node) — вкомпилировать в `<style>`, убрать CDN;
- [x] **плейсхолдеры + ре-рендер** (`rerender_from_html` через Playwright) — вернуть ~56% сэмплов с картинками; порядок: placeholder → precompile → rerender;
- [x] **масштаб** — count-driven (`TARGET_COUNT`), не завязан на срез;
- [ ] **прогнать боевой профиль** и проверить precompile/rerender вживую (нужны `pytailwindcss` + `playwright`; per-page медленно);
- [ ] согласовать `TARGET_SIZE` с вьюпортом eval и проставить `IMAGE_TOKEN_BUDGET` (см. §4 — размер).

### Этап 2 — Детерминированный рендерер (HTML → PNG)
Общий компонент: нужен и для `current_render` в polishing, и для RL-reward.
**У eval-трека это уже есть** (§1a): официальный Design2Code (`screenshot_single.py`,
Playwright+Chromium) + метрики `visual_eval_v3_multi`. Переиспользовать, не писать второй.
- [ ] взять рендерер/вьюпорт eval-трека как источник истины по размеру скриншота;
- [ ] детерминизм: без сетевых вызовов (отсюда запрет Tailwind-CDN), фикс шрифтов/DPI;
- [ ] отсев не-рендерящихся страниц для гигиены Этапа 1.

### Этап 3 — Reverse-construction синтетика (polishing + editing)
- [ ] polishing: эталонный HTML → «порча» / прогон через VLM → пары `(target, bad_code, render) → good_code`;
- [ ] editing: операции add/del/replace/adjust; **addition = разворот deletion-пар**;
      инструкции на естественном языке; фильтр эвристиками + ручная проверка.
- Ориентир объёма — доля от ~80K SFT (см. §2).

### Этап 4 — Масштабирование претрейн-корпуса
- [ ] свести WebCode2M + WebSight + Web2Code к единому формату пар (форма CSS не важна);
- [ ] переклассифицировать/разобрать Web2Code (train-usable, ~1.18M) — `analysis/web2code.ipynb`
      теми же 7 метриками (переиспользовать streaming из `webcode2m.ipynb`);
- [ ] микс с общими VLM-задачами для сохранения общих способностей (как в UI2Code^N).

---

## 4. Открытые вопросы (согласовать между треками)

Часть закрыта после разбора ветки `Eval` (§1a) — ниже актуальное.

| Вопрос | Статус / с кем | Почему важно |
|---|---|---|
| Серые плейсхолдеры | ✅ решено eval-треком — берём их конвенцию (§1a) | иначе трейн и eval разъедутся |
| ⚠ **Размер скриншота — фильтр теряет данные.** Контракт §4a требует один размер; конвертер по умолчанию `SIZE_MODE="filter"` оставляет только доминирующий нативный (у WebSight v0.2 это 2560×1440) → из 300 сырых остаётся ~89 (size-фильтр + `DROP_IMG`). При этом **Qwen умеет переменное разрешение** (`min/max_pixels` в SFT-коде) — жёсткий один размер модели не обязателен | eval + SFT | либо договориться об одном целевом размере и **ре-рендерить** в него (не терять данные, нужен рендерер), либо **разрешить переменный размер** (убрать фильтр, eval мерит на том же переменном разрешении) |
| ✅ **Модель зафиксирована SFT-треком: `Qwen/Qwen3.5-9B`, `max_length=8192`** | остаётся согласовать eval-baseline (был Qwen2.5-VL-3B) и `TOKENIZER_ID` в ноутбуках (сейчас Qwen3-VL-8B как прокси — счётчики близки; `train_sft.py` грузит модель через `AutoModelForImageTextToText` + `AutoProcessor` — это VL, прокси-токенайзер валиден) | трейн и eval на одной модели |
| ⚠ **Tailwind vs inline-style:** SFT/Data хотят Tailwind, eval-промпт просит plain inline `<style>` | **все — приоритет** | если таргеты на Tailwind, а eval рендерит без скомпилированного CSS — utility-классы (кроме плейсхолдера) не отрендерятся |
| ⚠ **Численный конфликт `max_length`:** SFT-конфиг = **8192** (оба — `lora_ft`/`full_ft`), но рекомендация по WebCode2M = **9 920** (p99, ещё +~1.1–1.2k визуальных токенов/картинку). WebSight (896) влезает, WebCode2M — нет | SFT-трек | без согласования реальные пары (Этап 4) не влезут в окно; фильтрация по бюджету токенов |

---

## 5. Ссылки
- Контракт: `../SFT/DATA_FORMAT_CONTRACT.md` (ветка `sft`)
- Eval-трек (рендерер, метрики, плейсхолдеры): ветка `Eval`, `Evaluation/Experiments.ipynb`
- Каталог датасетов: `list_data.md`
- EDA и метрики: `analysis/`
- Статьи: `papers/` (`papers/README.md`)
