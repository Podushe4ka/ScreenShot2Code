# Формат ТЗ (`briefs.jsonl`)

Одна строка — одна страница. Пример:

```json
{
  "id": "p0000",
  "seed_source": "flame_evo_react",
  "seed_id": "000018243",
  "task_description": "I need a simple todo application where I can add new tasks, mark them as completed, and delete them…",
  "layout_description": "The page features a todo application with a clean and modern design, centered with a maximum width of 800 pixels…",
  "impl": "react_cdn",
  "tier": "S",
  "dom_nodes": [40, 80],
  "target_bytes": [3000, 8000],
  "lang": "ja",
  "visual": "clean_light_saas",
  "features": ["breadcrumbs", "css_gradients", "z_index_overlap"]
}
```

| поле | смысл |
|---|---|
| `id` | имя выходного файла: `Data/synth_pilot/raw/<id>.html` |
| `seed_source`, `seed_id` | откуда взято ТЗ. Справочно, на генерацию не влияет |
| `task_description` | что за продукт. **Может быть пустым** — тогда всё содержание в `layout_description` |
| `layout_description` | готовое описание раскладки. Если оно есть — следуй ему, а не выдумывай своё |
| `impl` | `react_cdn` или `static_inline` — технология. Жёсткое требование |
| `tier` | `S` / `M` / `L` — плотность |
| `dom_nodes` | целевой диапазон числа DOM-узлов. **Критерий приёмки** |
| `target_bytes` | ориентир по размеру файла. Мягкий: см. ниже |
| `lang` | язык ВСЕГО видимого текста, включая форматы дат и чисел |
| `visual` | визуальный стиль. От темы не зависит, выполнять как заказано |
| `features` | элементы, которые обязаны присутствовать и быть видимыми на первом рендере |

## Две тонкости, на которых спотыкаются

**`target_bytes` — ориентир, `dom_nodes` — закон.** Байт-на-узел зависит и от `impl`
(у react исходник компактнее отрисованного дерева, потому что данные разворачиваются
через `.map()`: на замере 31 Б/узел против 51 у static), и от языка (UTF-8 добавляет
8–13% на кириллице и CJK). Попал в диапазон узлов, а по байтам слегка вышел — ничего
не режь. Единственный жёсткий потолок — 34 КБ, выше страница не влезает в бюджет
обучения по токенам.

**Пустой `task_description` — не ошибка.** Примерно у каждого восьмого ТЗ содержание
целиком лежит в `layout_description`, потому что в исходном корпусе там стояла
техническая заглушка. Работай по описанию раскладки.
