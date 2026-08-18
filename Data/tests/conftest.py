"""Общая настройка для тестов Data-трека.

Конвертеры запускаются как скрипты (так стоит в Dockerfile), а не как пакет: каталог
`converters/` попадает в `sys.path`, дальше `from common import ...`. Здесь повторяется
ровно этот пролог.

⚠ Модули конвертеров грузим ЯВНО, а не через общий `sys.path`. Файлов с именем
`convert_lib.py` три (websight, webcode2m, webui), и держать все их каталоги в пути —
значит зависеть от порядка вставки: `import convert_lib` молча взял бы чужой. `load()`
даёт каждому уникальное имя модуля.
"""
import importlib.util
import os
import sys

import pytest

DATA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(DATA)
CONVERTERS = os.path.join(DATA, "converters")

if CONVERTERS not in sys.path:      # для `from common import ...` — как в самих скриптах
    sys.path.insert(0, CONVERTERS)


def load(pkg, module):
    """Загрузить `converters/<pkg>/<module>.py` под уникальным именем."""
    d = os.path.join(CONVERTERS, pkg)
    if d not in sys.path:           # соседние импорты внутри пакета (например cssprune)
        sys.path.insert(0, d)
    key = f"_data_{pkg}_{module}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(d, module + ".py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[key] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="session")
def data_dir():
    return DATA


@pytest.fixture(scope="session")
def repo_dir():
    return REPO


PAGE = """<!doctype html>
<html><head><style>
  .card { color: #c00; padding: 4px; }
  @media (max-width: 600px) { .card { padding: 2px } }
</style></head>
<body>
  <header><h1>Заголовок</h1></header>
  <main>
    <div class="card"><p style="margin:0;padding:1px">текст</p></div>
    <table><tr><td>1</td><td>2</td></tr></table>
    <form><label>имя</label><input name="a"><button>ок</button></form>
    <svg viewBox="0 0 4 4"></svg>
    <img src="pic.png">
  </main>
</body></html>"""


@pytest.fixture
def page_html():
    """Небольшая, но не вырожденная страница: таблица, форма, svg, img, стили."""
    return PAGE
