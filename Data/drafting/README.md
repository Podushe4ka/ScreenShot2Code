# Drafting-конвертер: WebSight → формат контракта

Готовит drafting-датасет (`task_type="drafting"`, скриншот → HTML) в схеме
`SFT/DATA_FORMAT_CONTRACT.md`, готовый к `load_from_disk` на SFT-стороне.

## Файлы
| Файл | Что |
|---|---|
| `convert_lib.py` | **вся логика** (гигиена, плейсхолдеры, precompile, рендер, токены) — единый источник правды |
| `convert.ipynb` | интерактив (отладка/просмотр), импортит `convert_lib` |
| `convert_parallel.py` | батч на многих ядрах (ProcessPool, spawn), импортит `convert_lib` |
| `Dockerfile` | окружение (браузеры + либы + зависимости) для батча |
| `view_arrow.py` | просмотр .arrow-датасета (сводка / `--extract-images` / `--html` отчёт) |

## Два профиля
- **Смоук:** `APPLY_PLACEHOLDERS=False`, `PRECOMPILE_TAILWIND=False`. Быстро, **без браузера** (берёт
  оригинальные скриншоты WebSight). CDN-Tailwind, без картинок — для проверки пайплайна.
- **Production:** `APPLY_PLACEHOLDERS=True` + `PRECOMPILE_TAILWIND=True`. Self-contained: `<img>` и
  CSS-фоны → серые плейсхолдеры, Tailwind вкомпилен в `<style>`, скриншот перерисован full-page
  (ширина 1280, высота по контенту). Нужны `playwright` + `pytailwindcss`.

## Массовая генерация (рекомендуется — через Docker)
```bash
# из корня репозитория
docker build -t ws-conv -f Data/drafting/Dockerfile .
docker run --rm -v "$PWD":/work --shm-size=2g ws-conv --target 5000 --n-workers 32
```
Датасет ляжет в `Data/drafting/websight_drafting_pilot/` на хосте. ~5000 за пару минут (48/с на 64 воркерах).
Ручки: `--target N` (сколько собрать), `--n-workers` (процессов; 32–48 оптимум), `--near-dup K`.

## Локально без Docker
```bash
pip install datasets pillow beautifulsoup4 transformers playwright pytailwindcss tqdm
playwright install chromium
python convert_parallel.py --target 500 --n-workers 16   # батч
# или convert.ipynb — пошагово (тот же convert_lib)
```

## Передача SFT
Данные — через диск/том (контракт §7), не через git (`*drafting_pilot*/` в .gitignore):
`-v <путь>/websight_drafting_pilot:/data` → SFT `load_from_disk("/data")`.
`max_length` берётся из токен-отчёта (код p99 + картинка p99; ~6144 с запасом под конфиг 8192).

## Границы
- Tailwind precompile — v4 через standalone `pytailwindcss` (без Node). Редкие кривые страницы
  (~0.15%) не компилятся и пропускаются.
- Размер: production рендерит full-page с фиксированной шириной (`RENDER_WIDTH=1280`), высота по
  контенту (Qwen переваривает переменное разрешение). Открытый вопрос согласования с eval — см. `../PLAN.md` §4.
