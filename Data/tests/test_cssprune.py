"""Счётчик CSS-объявлений: он входит в сводную сложность, значит двигает отбор.

Разбор свой (с оффсетами — их требует tree-shaking через CDP), поэтому и краевые случаи
свои: вложенные `@media`, комментарии, точка с запятой внутри строкового значения.
"""
import pytest

from conftest import load


@pytest.fixture(scope="module")
def cssprune():
    return load("webui", "cssprune")


def test_flat_rule(cssprune):
    assert cssprune.count_declarations("a { color: red; padding: 1px }") == 2


def test_nested_at_media_counted(cssprune):
    """Правила внутри @media — тоже объявления; считаются листья, а не блоки."""
    css = "a{color:red} @media (max-width:600px){ a{color:blue; margin:0} }"
    assert cssprune.count_declarations(css) == 3


def test_comments_do_not_add_declarations(cssprune):
    css = "a{ /* color: red; padding: 2px */ margin: 0 }"
    assert cssprune.count_declarations(css) == 1


def test_semicolon_inside_string_is_not_a_separator(cssprune):
    css = 'a { content: "a;b"; color: red }'
    assert cssprune.count_declarations(css) == 2


def test_empty_and_garbage(cssprune):
    assert cssprune.count_declarations("") == 0
    assert cssprune.count_declarations("а тут вообще не css") == 0


# ── guard_style_close: буквальный `</style` внутри CSS не должен закрывать <style> ────
#
# WebUI берёт CSS с живых сайтов (документация, style-guide'ы), а не генерирует его сам —
# в отличие от websight/webcode2m. У таких страниц `content:` в ::before/::after нередко
# показывает пример разметки, и там встречается буквальный `</style>`. HTML-токенайзер
# в состоянии RAWTEXT (внутри <style>) реагирует на эту подстроку независимо от того,
# что по CSS-синтаксису она лежит внутри строки или комментария — тег закрывается
# раньше времени, хвост CSS вытекает в тело как видимый текст, а всё, что шло дальше в
# исходном HTML, парсится заново (в том числе вернувшийся `<script>`, который
# sanitize_html уже вырезал ДО этого момента).

def test_guard_leaves_ordinary_css_untouched(cssprune):
    css = '.card > .title { color: red; } a[href^="http"]::after{content:"→"}'
    assert cssprune.guard_style_close(css) == css


def test_guard_breaks_literal_close_tag(cssprune):
    css = '.x::before{content:"see </style> here"}'
    out = cssprune.guard_style_close(css)
    assert "</style" not in out.lower()
    assert "​" in out


def test_guard_is_case_insensitive_and_handles_all_boundaries(cssprune):
    # Каждый хвост уже несёт свою границу (`>`, `/`, пробел) — только ПОСЛЕ такой
    # границы HTML-токенайзер вообще признаёт "</style" настоящим закрывающим тегом.
    for tail in ("</style>", "</STYLE>", "</Style >", "</style/"):
        out = cssprune.guard_style_close(f'content:"x{tail}y"')
        assert "</style" not in out.lower(), tail
    # Голое "</style" без ничего после — граница здесь конец строки (EOF), и её
    # нельзя сымитировать, приписав `y`: тогда "style" оказалось бы продолжено
    # буквой, что уже НЕ читается браузером как настоящий закрывающий тег вовсе.
    out_eof = cssprune.guard_style_close('content:"x</style')
    assert "</style" not in out_eof.lower()


def test_guard_does_not_touch_unrelated_close_tags(cssprune):
    css = 'content:"</div><//style><script>"'
    out = cssprune.guard_style_close(css)
    assert out == css


@pytest.mark.slow
def test_guard_end_to_end_survives_real_render(cssprune):
    """Полный конвейер: css с </style> внутри доходит до реального Chromium неповреждённым."""
    playwright = pytest.importorskip("playwright.sync_api")
    conv = load("webui", "convert_lib")
    soup, css_all, _ = conv.assemble_page(
        '<html><head></head><body><div class="x">demo</div><p>after</p></body></html>',
        '.x::before { content: "see </style><script>window.__pwned=1</script>"; } '
        '.x { color: rgb(255, 0, 0); } p { color: rgb(0, 0, 255); }')
    html = conv.build_html(soup, css_all)
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        try:
            page.set_content(html, wait_until="load")
            assert page.eval_on_selector(".x", "el => getComputedStyle(el).color") == "rgb(255, 0, 0)"
            assert page.eval_on_selector("p", "el => getComputedStyle(el).color") == "rgb(0, 0, 255)"
            assert page.eval_on_selector_all("style", "els => els.length") == 1
            assert page.eval_on_selector_all("script", "els => els.length") == 0
            assert "window.__pwned" not in page.eval_on_selector("body", "el => el.innerHTML")
        finally:
            browser.close()
