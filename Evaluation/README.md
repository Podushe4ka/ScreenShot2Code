# Evaluation — четыре независимых инструмента

Каталог разделён по назначению: у каждого подкаталога свой Dockerfile, свой
`run.sh` и свой образ. Они **не переиспользуют** код друг друга — копии `render.py`
живут отдельно в `metrics_only/` и `streamlit/`.

| подкаталог | что делает | точка входа |
|---|---|---|
| [`metrics_only/`](metrics_only/) | **основной бенч Design2Code**: генерация через vLLM, рендер, пять метрик, `final_score` | `metrics_only/run.sh` → [`RUNNING.md`](metrics_only/RUNNING.md) |
| [`judge_one_gpu/`](judge_one_gpu/) | те же метрики **плюс VLM-судья**: попарное сравнение чекпоинта с базой, `judge_winrate` | `judge_one_gpu/run.sh` |
| [`judge_prompt/`](judge_prompt/) | калибровка промпта судьи на размеченных парах | `judge_prompt/run.sh` |
| [`streamlit/`](streamlit/) | веб-просмотр генераций | `streamlit/run.sh` |

`Experiments.ipynb` — исторический ноутбук из Colab, с которого начинался
eval-трек. Не часть ни одного из образов; оставлен как запись происхождения
метрик. Его единственный числовой результат перенесён в
[`../docs/experiments/misc-smoke-and-compare.md`](../docs/experiments/misc-smoke-and-compare.md).

## Что чем меряли

Оркестраторы `experiments/*.sh` зовут **`metrics_only/run.sh`** — все числа в
[`../docs/RESULTS.md`](../docs/RESULTS.md) сняты им.

Прогон судьи (`judge_one_gpu/bench_results/`) — отдельный набор из 2000 сэмплов,
с таблицами `RESULTS.md` **напрямую не сравнивается**: там Design2Code-484.

## ⚠ Дублирование, которое стоит помнить

`render.py`, `metrics.py`, `clip_server.py` существуют в двух-трёх копиях
(`metrics_only/`, `judge_one_gpu/`, `streamlit/`). Фикс, внесённый в одну копию,
в остальные сам не приедет. Копии уже разошлись: четыре фикса харнесса есть
только в `metrics_only/`, поэтому числа с разных инструментов несравнимы.

Расхождения между копиями фиксируются в
[`../docs/experiments/DIVERGENCES.md`](../docs/experiments/DIVERGENCES.md).
