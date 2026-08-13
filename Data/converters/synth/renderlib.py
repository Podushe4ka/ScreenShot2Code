"""Общий рендер синтетических страниц. Используется и build.py, и degrade.py.

Вынесено в отдельный модуль сознательно: здесь живёт асимметрия react_cdn — скриншот
снимается с МАТЕРИАЛИЗОВАННОГО DOM (после React+Babel), а в датасет едет СЫРОЙ исходник.
Если продублировать эту логику в двух местах, она однажды разъедется, и половина набора
будет собрана по одному правилу, половина по другому — молча.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "Evaluation" / "metrics_only"))
sys.path.insert(0, str(REPO / "Data" / "converters" / "websight"))

import numpy as np  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

import render as ev_render  # noqa: E402
from convert_lib import render_full  # noqa: E402


def count_nodes(html_text: str) -> int:
    soup = ev_render._make_soup(html_text)
    body = soup.body or soup
    return sum(1 for _ in body.find_all(True))


def screenshot_stats(img: PILImage.Image) -> dict:
    a = np.asarray(img.convert("RGB"))
    return {"w": img.width, "h": img.height,
            "std": round(float(a.std()), 2),
            "colors": int(len(np.unique(a.reshape(-1, 3), axis=0)))}


def render_page(raw: str, impl: str, work: Path, name: str) -> tuple:
    """(картинка, отрисованный_html, инфо). Для react_cdn отрисованный != сырому.

    `отрисованный_html` нужен только для подсчёта узлов и диагностики; в датасет
    всегда кладётся `raw`.
    """
    info = {}
    if impl == "react_cdn":
        tmp = work / f"{name}.html"
        tmp.write_text(raw, encoding="utf-8")
        mi = ev_render.materialize_dom(str(tmp))
        info["materialized"] = bool(mi.get("materialized"))
        info["error"] = mi.get("error")
        if info["materialized"]:
            rendered = tmp.read_text(encoding="utf-8", errors="replace")
            # Прирост длины — дешёвый признак того, что React действительно отрисовал:
            # при пустом #root DOM почти не растёт.
            info["grew"] = mi["len_after"] >= mi["len_before"] * 1.2
        else:
            rendered, info["grew"] = raw, False
    else:
        rendered, info["materialized"], info["grew"] = raw, None, True

    img = render_full(rendered)
    return img, rendered, info


def visual_diff(a: PILImage.Image, b: PILImage.Image, width: int = 640) -> float:
    """Доля заметно различающихся пикселей, 0..1.

    Сравниваем с сохранением пропорций и на достаточно крупном холсте. Первая версия
    ужимала обе картинки в квадрат 256x256, и это давало ложные нули: перекраска 43
    бейджей на плотном дашборде 1280x1853 давала diff 0.0066, а смена цвета текста в
    85 местах — 0.0003, потому что мелкие элементы при таком ужатии просто усредняются
    с фоном. Высоты могут отличаться (правка меняет длину страницы), поэтому сравниваем
    общую верхнюю часть, а расхождение высот учитываем отдельным слагаемым.
    """
    def norm(img):
        h = max(1, round(img.height * width / img.width))
        return np.asarray(img.convert("RGB").resize((width, h)), dtype=np.int16)

    xa, xb = norm(a), norm(b)
    h = min(xa.shape[0], xb.shape[0])
    changed = (np.abs(xa[:h] - xb[:h]).max(axis=2) > 16).mean()
    # Если одна страница длиннее, «лишняя» часть — тоже различие, и немалое.
    tall = max(xa.shape[0], xb.shape[0])
    return float(changed * h / tall + (tall - h) / tall)
