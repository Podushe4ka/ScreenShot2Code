# Статьи к датасетам и методам (для контекста)

Скачано 2026-07-17. Все PDF — с arXiv (кроме отмеченных).

| Файл | Что это | arXiv | Роль |
|---|---|---|---|
| `UI2CodeN_2511.08195.pdf` | **UI2Code^N** — базовая статья нашего подхода (Zhipu/Tsinghua, GLM-4.1V-9B) | 2511.08195 | метод: pretrain→SFT→RL, editing+polishing |
| `WebSight_2403.09029.pdf` | WebSight — 2M синтетика, Tailwind CSS | 2403.09029 | train (pretrain) |
| `WebCode2M_Vision2UI_2404.06369.pdf` | WebCode2M / Vision2UI — 2.5M реальных страниц | 2404.06369 | train (pretrain) |
| `Web2Code_2406.20098.pdf` | Web2Code — ~1.18M instruction-пар + QA | 2406.20098 | train (SFT/drafting) + eval |
| `Design2Code_2403.03163.pdf` | Design2Code — бенчмарк Stanford | 2403.03163 | eval |
| `Flame_2503.01619.pdf` | Flame — React-генерация + Flame-React-Eval | 2503.01619 | eval |
| `WebGen-Bench_2505.03733.pdf` | WebGen-Bench — функциональные сайты | 2505.03733 | eval (расширение) |
| `DesignBench_2506.06251.pdf` | DesignBench — generation/editing/repair | 2506.06251 | eval (в т.ч. editing) |
| `Vision2Web_2603.26648.pdf` | Vision2Web — иерархический бенч (тот же Zhipu) | 2603.26648 | eval |
| `pix2code_1705.07962.pdf` | pix2code — исторический GUI→DSL | 1705.07962 | справка |

## Нет на arXiv (добавить вручную при необходимости)
- **WebUI** (ronantakizawa/webui) — CHI 2023, Wu et al. "WebUI: A Dataset for Enhancing
  Visual UI Understanding with Web Semantics". Ссылка на HF в `../list_data.md`.
- **ScreenBench**, **Interaction2Code (ASE 2025)**, **UI2Code-Real / UIPolish-bench**
  (собственные бенчи UI2Code^N) — см. репозиторий github.com/zai-org/UI2Code_N.
