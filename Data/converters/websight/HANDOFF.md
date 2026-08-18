# Handoff: drafting-датасет → SFT-трек

Передача drafting-датасета (WebSight v0.2 → скриншот→HTML) SFT-треку по
контракту [`../../SFT/DATA_FORMAT_CONTRACT.md`](../../../SFT/DATA_FORMAT_CONTRACT.md).
Адресат — SFT-трек (владелец контракта). Источник — data-трек, ветка `data/drafting`.

## 1. Что передаётся

- **Что:** `task_type="drafting"` — пары (скриншот целевой страницы → эталонный HTML).
- **Объём:** пилот 4998 сэмплов (`--target 5000`, 2 отсева по некомпилящемуся Tailwind).
- **Источник:** `HuggingFaceM4/WebSight` (v0.2, Tailwind, чистая синтетика).
- **Профиль:** production — self-contained (`APPLY_PLACEHOLDERS=True` + `PRECOMPILE_TAILWIND=True`).

## 2. Как получить (передача через диск/том, НЕ git)

Датасет в `.gitignore` (`*drafting_pilot*/`) — большие Arrow-байты, передаём диском.

```python
from datasets import load_from_disk
ds = load_from_disk("<путь>/websight_drafting_pilot")
```

Монтирование в контейнер (контракт §7):
```bash
-v <путь>/websight_drafting_pilot:/data   # SFT: load_from_disk("/data")
```

Перегенерация: `docker run --rm -v "$PWD":/work --shm-size=2g ws-conv --target 5000 --n-workers 64`
(из корня репо; результат — в `Data/websight_drafting_pilot/`). Логика — `convert_lib.py`.

## 3. Схема сэмпла (контракт §2, профиль drafting)

| Поле | Тип | Значение |
|---|---|---|
| `task_type` | string | `"drafting"` |
| `images` | list[Image] | `[target_screenshot]` — 1 картинка, встроенные байты |
| `current_html` | string | пусто |
| `target_html` | string | эталонный self-contained HTML |
| `instruction` | string | пусто |

## 4. Гарантии (приёмка контракта §6)

| Пункт | Статус |
|---|---|
| читается `load_from_disk` без ошибок | ✅ |
| `task_type` + поля по §2 | ✅ (`FEATURES` = таблица §2) |
| картинки встроены как `Image()` (не пути) | ✅ (`{bytes, path:None}`) |
| self-contained HTML | ✅ |
| Tailwind предкомпилирован в `<style>` (без Play CDN) | ✅ (`precompile_tailwind`, standalone v4, без Node) |
| серые плейсхолдеры вместо `<img>` и CSS-фонов | ✅ дословно как eval: `class="bg-gray-300 w-full h-48 rounded"` + inline `background-color:#d1d5db;width:100%;height:12rem;border-radius:0.5rem;display:block;` |
| дедуп (SHA1) + near-dup (avg-hash) + декотаминация eval-доменов | ✅ |
| фильтр по бюджету токенов | ✅ |
| **единый фиксированный размер (§4a)** | ⚠️ **см. §6 — открытый вопрос** |

## 5. Бюджет токенов → `max_length`

Боевой таргет проходит `precompile_tailwind` (компилированный Tailwind в `<style>`),
поэтому он в ~4× длиннее сырого `text` WebSight. Замерено на пилоте (Qwen-токенайзер):

| | median | p99 | max |
|---|---|---|---|
| код (`target_html`) | ~2 321 | **~3 815** | ~4 927 |
| визуальные токены — ⚠ снято до Tier A | ~960 | ~1 672 | ~1 672 |
| **всего** — ⚠ то же | ~3 409 | **~5 305** | ~6 485 |

⚠ Строка «код» действительна, строки с картинкой — нет: они посчитаны при потолке 1.31 Мп
и patch=28. С 18 августа `qwen_image_tokens` читает `SFT_MIN_PIXELS`/`SFT_MAX_PIXELS`
(те же, что `SFT/train/formatting.py`), потолок — **2048** токенов, пол — 256.

→ **`max_length` не выше ~5 952** (код p99 3 815 + потолок картинки 2048), рабочее окно
16384 берёт с многократным запасом. Точный пересчёт: `--token-report` (тянет
transformers/torch) — он печатает медиану/p99/max по факту, а не по потолку.

## 6. ⚠️ Открытый вопрос — размер скриншота (нужно решение eval + SFT)

Ширина фиксирована **1280 px** (совпадает с рекомендацией §4a). А вот **высота
переменная** (full-page по контенту: min=60, median=588, max=6144), тогда как
контракт §4a рекомендует **фиксированную высоту с обрезкой/паддингом**.

Обоснование варианта data-трека: Qwen3-VL штатно ест переменное разрешение
(`min_pixels`/`max_pixels` в процессоре), а обрезка/паддинг до одной высоты либо
режет контент, либо добавляет пустой шум. Full-page скриншот точно соответствует
всему `target_html`.

**Нужно согласовать одно из двух** (детали — [`../PLAN.md`](../../PLAN.md) §4):
1. **разрешить переменную высоту** — eval мерит на том же переменном разрешении; или
2. **фиксировать высоту** — data-трек ре-рендерит в единый размер вьюпорта eval.

До решения датасет пригоден для пилотного LoRA-запуска (§5 контракта: даже небольшой
сет разблокирует пилот), но перед масштабным трейном размер надо зафиксировать.

## 7. Проверка на стороне SFT (воспроизвести приёмку §6)

```python
from datasets import load_from_disk
d = load_from_disk("/data")
assert len(d) > 0 and set(d["task_type"]) == {"drafting"}
s = d[0]
assert s["images"] and s["current_html"] == "" and s["instruction"] == ""
assert "<img" not in s["target_html"].lower() and "cdn.tailwindcss" not in s["target_html"]
assert "bg-gray-300" in "".join(x["target_html"] for x in d.select(range(min(50, len(d)))))
print("OK:", len(d), "сэмплов, ширина", {img.size[0] for img in [s["images"][0]]})
```

Плюс ручной глазами-осмотр 5 случайных сэмплов (§6): `python view_arrow.py
<путь>/data-00000-of-00001.arrow --html report.html` — открыть `report.html`,
сверить скриншот ↔ код (self-contained, Tailwind, серые плейсхолдеры на месте).
