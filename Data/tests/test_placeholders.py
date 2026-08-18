"""Серые плейсхолдеры: одна конвенция на трек.

Строки классов и стиля обязаны совпадать с Evaluation/metrics_only/render.py посимвольно —
бенч подменяет `<img>` и в предсказании, и в эталоне. Реализаций раньше было две (по
тексту и по soup), и они могли разъехаться независимо; тесты проверяют, что теперь это
одна функция и результат у обоих входов одинаковый.
"""
from bs4 import BeautifulSoup

from common.placeholders import (PLACEHOLDER_CLASSES, PLACEHOLDER_STYLE,
                                 replace_images_with_placeholder, strip_background_images)


def test_img_becomes_gray_div():
    html, n = replace_images_with_placeholder('<div><img src="a.png"><img src="b.png"></div>')
    assert n == 2
    assert "<img" not in html.lower()
    assert html.count("bg-gray-300") == 2
    assert PLACEHOLDER_STYLE in html


def test_placeholder_convention_is_the_bench_one():
    """Если эти строки поедут — обучающие таргеты перестанут походить на то, что видит метрика."""
    assert PLACEHOLDER_CLASSES == ["bg-gray-300", "w-full", "h-48", "rounded"]
    assert PLACEHOLDER_STYLE == ("background-color:#d1d5db;width:100%;height:12rem;"
                                 "border-radius:0.5rem;display:block;")


def test_background_images_are_killed():
    """Hero-фото — тоже картинка, и оно живёт в CSS, а не в <img>."""
    out = strip_background_images('<div style="background-image:url(hero.jpg);color:red">x</div>')
    assert "url(" not in out
    assert "#d1d5db" in out
    assert "color:red" in out


def test_text_path_also_kills_backgrounds():
    html, _ = replace_images_with_placeholder('<div style="background:url(x.png) no-repeat"></div>')
    assert "url(" not in html


def test_soup_and_text_paths_agree():
    """Оба входа должны давать одинаковую разметку — иначе корпуса разъедутся."""
    src = '<section><img src="a.png" alt="а"><p>текст</p></section>'
    from_text, n_text = replace_images_with_placeholder(src, strip_backgrounds=False)
    soup = BeautifulSoup(src, "html.parser")
    n_soup = replace_images_with_placeholder(soup)
    assert n_text == n_soup == 1
    assert str(soup) == from_text


def test_idempotent():
    """Повторный прогон не должен плодить вложенные плейсхолдеры."""
    once, n1 = replace_images_with_placeholder('<div><img src="a.png"></div>')
    twice, n2 = replace_images_with_placeholder(once)
    assert n2 == 0
    assert twice == once
