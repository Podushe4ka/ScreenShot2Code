"""Серые плейсхолдеры вместо картинок — одна конвенция на трек.

⚠ Строки классов и стиля обязаны ПОСИМВОЛЬНО совпадать с `Evaluation/metrics_only/render.py`:
бенч подменяет `<img>` и в предсказании, и в эталоне, поэтому расхождение конвенции сделало
бы обучающие таргеты непохожими на то, что видит метрика.

Раньше реализаций было две — по тексту (websight) и по soup (webui), — и они могли
разъехаться независимо друг от друга, а вместе с ними разъехались бы и корпуса. Теперь
одна функция принимает и то и другое: строку (вернёт строку) или уже разобранный soup
(правит его на месте).
"""
import re

from bs4 import BeautifulSoup

PLACEHOLDER_CLASSES = ["bg-gray-300", "w-full", "h-48", "rounded"]
PLACEHOLDER_STYLE = ("background-color:#d1d5db;width:100%;height:12rem;"
                     "border-radius:0.5rem;display:block;")

# CSS background-image: url(...) — hero-фото тоже картинка, и его тоже надо гасить.
_BG_RE = re.compile(r'background(-image)?\s*:\s*[^;{}"\']*url\([^)]*\)[^;{}"\']*', re.I)


def strip_background_images(html_text):
    """`background-image: url(...)` -> серый фон."""
    return _BG_RE.sub("background-color:#d1d5db", html_text)


def _replace_in_soup(soup):
    """`<img>` -> серый `<div>` прямо в дереве. Возвращает число замен."""
    n = 0
    for img in soup.find_all("img"):
        div = soup.new_tag("div")
        div["class"] = list(PLACEHOLDER_CLASSES)
        div["style"] = PLACEHOLDER_STYLE
        img.replace_with(div)
        n += 1
    return n


def replace_images_with_placeholder(html_or_soup, strip_backgrounds=True):
    """`<img>` -> серый `<div>`; на вход строка ИЛИ soup.

    * строка  -> возвращает `(html, n_img)`; фоновые картинки гасятся тем же вызовом,
      потому что по тексту это единственный момент, когда они видны;
    * soup    -> правит дерево на месте и возвращает просто `n_img`. Фоновые картинки
      здесь НЕ трогаются: у soup-пути (WebUI) стили едут отдельной колонкой и гасятся
      на своём шаге, а `str(soup)` ради регулярки означал бы лишний круг сериализации.
    """
    if isinstance(html_or_soup, str):
        soup = BeautifulSoup(html_or_soup, "html.parser")
        n = _replace_in_soup(soup)
        out = str(soup)
        return (strip_background_images(out) if strip_backgrounds else out), n
    return _replace_in_soup(html_or_soup)
