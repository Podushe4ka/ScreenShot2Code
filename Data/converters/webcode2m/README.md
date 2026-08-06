# WebCode2M-конвертер → формат контракта

Готовит датасет (`task_type="drafting"`, скриншот → HTML) из **реального** корпуса
WebCode2M (`xcodemind/webcode2m_purified`) в схеме `SFT/DATA_FORMAT_CONTRACT.md`.
Структура зеркалит `../drafting/`, но под реальные страницы.

## Файлы
| Файл | Что |
|---|---|
| `convert_lib.py` | логика: sanitize внешних ресурсов + де-блоб + плейсхолдеры + рендер. **Переиспользует ядро `../drafting/convert_lib.py`** (плейсхолдеры, `render_full`, токены, `FEATURES`) — один источник правды |
| `convert.ipynb` | интерактив (пошагово, рендер через `render_threaded`) |
| `convert_parallel.py` | батч на многих ядрах (ProcessPool, spawn) |

## Отличия от WebSight-конвертера (`../drafting/`)
- источник — реальные pruned-страницы, CSS уже в `<style>`/`style=` → **Tailwind-precompile НЕ применяется**;
- страницы могут тянуть внешние ресурсы → `sanitize_offline` вырезает `<script>` и внешние
  `<link rel=stylesheet>` (инлайновый `<style>` остаётся) для детерминированного оффлайн-рендера;
- токены: p99 кода ~9920 — **выше SFT-окна 8192**, для SFT нужен фильтр по длине (для pretrain мягче).

## Запуск
```bash
# из этой папки; нужны playwright + chromium (Tailwind тут не нужен)
python convert_parallel.py --target 5000 --n-workers 32     # батч -> ./webcode2m_pilot
# или convert.ipynb — пошагово
```
Реальные страницы → следи за долей ошибок рендера (печатается) и за токенным хвостом.
