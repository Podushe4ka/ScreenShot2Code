# WebCode2M-конвертер → формат контракта

Готовит датасет (`task_type="drafting"`, скриншот → HTML) из **реального** корпуса
WebCode2M (`xcodemind/webcode2m_purified`) в схеме `SFT/DATA_FORMAT_CONTRACT.md`.
Структура зеркалит `../websight/`, но под реальные страницы.

## Файлы
| Файл | Что |
|---|---|
| `convert_lib.py` | логика: sanitize внешних ресурсов + де-блоб + плейсхолдеры + рендер. **Переиспользует ядро `../websight/convert_lib.py`** (плейсхолдеры, `render_full`, токены, `FEATURES`) — один источник правды |
| `convert.ipynb` | интерактив (пошагово, рендер через `render_threaded`) |
| `convert_parallel.py` | батч на многих ядрах (ProcessPool, spawn) |

## Отличия от WebSight-конвертера (`../websight/`)
- источник — реальные pruned-страницы, CSS уже в `<style>`/`style=` → **Tailwind-precompile НЕ применяется**;
- страницы могут тянуть внешние ресурсы → `sanitize_offline` вырезает `<script>` и внешние
  `<link rel=stylesheet>` (инлайновый `<style>` остаётся) для детерминированного оффлайн-рендера;
- токены: p99 кода ~9 920, плюс картинка до 2048 → ~12k при рабочем окне **16384** — влезает.
  (До поднятия окна это было «выше 8192, нужен фильтр по длине»; фильтр по бюджету полезен
  и сейчас, но уже не обязателен именно для WebCode2M.)

## Запуск
```bash
# из этой папки; нужны playwright + chromium (Tailwind тут не нужен)
python convert_parallel.py --target 5000 --n-workers 32     # батч -> ./webcode2m_pilot
# или convert.ipynb — пошагово
```
Реальные страницы → следи за долей ошибок рендера (печатается) и за токенным хвостом.

## Тестовый набор без пересечения с обучающим
Обучающие наборы берутся с начала корпуса (15k собран из первых 60k записей, см.
`experiments/run_wc2m_15k.sh`), поэтому тест берём из хвоста:

- `--skip N` — выбросить первые N записей стрима **до** фильтров; `--max-scan` считается
  уже после пропуска;
- `--exclude-html <кэш.jsonl.gz>` — не брать страницы, чей HTML лежит в html-кэше фазы 1
  другого набора (страховка от повторов страниц внутри самого корпуса, сверх позиционного
  `--skip`). Можно повторять. Ждём именно html-кэш: в нём **сырой** HTML, а `target_html`
  готового набора уже прошёл sanitize и по хэшу не совпадёт;
- html-кэш теперь помечен участком корпуса (`<кэш>.meta.json` со `skip`): кэш обучающего
  набора не подхватится под тестовый прогон и наоборот. У кэшей, сделанных до этого,
  сайдкара нет — они считаются `skip=0`, то есть началом корпуса.

```bash
# 500 страниц с 200k-й записи; в контейнере, как в run_wc2m_15k.sh
docker run --rm -v "$PWD":/w -v /mnt/storage-1:/storage --shm-size=2g \
  -e HF_HOME=/storage/Screenshot2Code/hf_cache ${HF_TOKEN:+-e HF_TOKEN} \
  -w /w/Data/converters/webcode2m --entrypoint python3 design2code-bench \
  convert_parallel.py --target 500 --skip 200000 --max-scan 5000 --n-workers 32 \
    --exclude-html /storage/Screenshot2Code/data/webcode2m_15000_htmls.jsonl.gz \
    --html-cache /storage/Screenshot2Code/data/webcode2m_test500_skip200k_htmls.jsonl.gz \
    --out /storage/Screenshot2Code/data/webcode2m_test500
```
⚠ Пропуск 200k записей — это протяжка стрима: без `--near-dup` конвертер читает только
колонку `text` (скриншоты источника не качаются, у них своя тяжёлая колонка), но время
всё равно кратно фазе 1 обучающего набора (~40 минут на 15k). Прогресс пропуска печатается
строкой каждые 10k записей со скоростью и оценкой остатка (не прогресс-баром: под
`docker run -d` бар на `\r` в логе не виден), а собранные HTML кладутся в кэш — падение
рендера не заставит стримить заново.
