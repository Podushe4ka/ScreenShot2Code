WebCode2M (он же Vision2UI)

https://huggingface.co/datasets/xcodemind/webcode2m
Очищенная версия: https://huggingface.co/datasets/xcodemind/webcode2m_purified
Статья: arXiv:2404.06369 — https://arxiv.org/abs/2404.06369
Проект: https://webcode2m.github.io

WebSight

https://huggingface.co/datasets/HuggingFaceM4/WebSight (v0.1 — 823k, v0.2 — 1.92M)
Уменьшенная версия: https://huggingface.co/datasets/mrm8488/WebSight_70k
Статья: arXiv:2403.09029 — https://arxiv.org/abs/2403.09029
Blog post: https://huggingface.co/blog/websight

WebUI (реальные UI, CodePen/дизайн-системы, мультивьюпорт)

https://huggingface.co/datasets/ronantakizawa/webui

Новые train-кандидаты (найдено 2026-07-17)

Web2Code (MBZUAI) — крупный instruction-датасет, а НЕ только бенчмарк:
  ~1.18M пар «изображение страницы + инструкция → HTML» + QA о странице.
  Сгенерирован GPT-3.5/4, ложится на основной drafting-сценарий и «генерацию по инструкции».
  https://huggingface.co/datasets/MBZUAI/Web2Code
  Статья: arXiv:2406.20098 — https://arxiv.org/abs/2406.20098

Pix2Code (Beltramelli, 2017) — маленький исторический GUI→DSL (web/iOS/android),
  для web-части полезен как sanity-набор, не для основного train.
  https://github.com/tonybeltramelli/pix2code

Sketch2Code (Microsoft) — рукописные наброски интерфейса → HTML.
  Другая модальность входа; для устойчивости к «черновым» входам.

Бенчмарки (только для eval, НЕ для train — во избежание contamination)
Design2Code

Статья: arXiv:2403.03163 — https://arxiv.org/abs/2403.03163
(репозиторий/датасет искать по названию на GitHub/HF от Stanford — Si et al., 2024)

Flame-React-Eval

Статья (Flame): arXiv:2503.01619 — https://arxiv.org/abs/2503.01619

Web2Code (eval-split; train-часть — см. раздел «Новые train-кандидаты» выше)

Статья: arXiv:2406.20098 — https://arxiv.org/abs/2406.20098

WebGen-Bench (расширение до функциональных сайтов, упомянуто в Related Work UI2CodeN)

Статья: arXiv:2505.03733 — https://arxiv.org/abs/2505.03733

Vision2Web (zai-org / Zhipu, 2026) — иерархический бенч: L1 статичная страница (100),
  L2 интерактивный фронт (66), L3 фуллстек (27). Лицензия CC-BY-NC-SA-4.0 (eval-only).
  https://huggingface.co/datasets/zai-org/Vision2Web · https://github.com/zai-org/Vision2Web

DesignBench (2025) — несколько фреймворков + задачи generation/editing/repair;
  напрямую покрывает наш сценарий редактирования.
  Статья: arXiv:2506.06251 — https://arxiv.org/html/2506.06251v3

Interaction2Code (ASE 2025) — интерактивные/многостраничные сайты, за рамками MVP.

ScreenBench — 1000 свежих реальных скриншотов + HTML, разнообразные темы.

Смежные модальности (grounding/pretraining, не HTML напрямую)

Rico / RicoSCA / Enrico — ~66k экранов Android UI; вторично для HTML/CSS-таргета.

Датасеты из UI2CodeN, собственные (не опубликованы отдельно, для справки)

UI2Code-Real — 115 реальных страниц, собственный бенчмарк авторов UI2CodeN (не найден публичный HF-репозиторий, вероятно в их github: https://github.com/zai-org/UI2Code_N)
UIPolish-bench — синтетический+реальный, там же