"""Гигиена реальных страниц: что вырезается перед оффлайн-рендером.

Оба конвертера реальных корпусов рендерят БЕЗ сети. Если внешний ресурс переживёт чистку,
рендер станет недетерминированным (или повиснет), а таргет — непохожим на скриншот.
"""
import pytest

from common.placeholders import replace_images_with_placeholder
from conftest import load


@pytest.fixture(scope="module")
def wc2m():
    return load("webcode2m", "convert_lib")


@pytest.fixture(scope="module")
def webui():
    return load("webui", "convert_lib")


def test_scripts_are_dropped(wc2m):
    out = wc2m.sanitize_offline('<html><body><script>alert(1)</script><p>да</p></body></html>')
    assert "<script" not in out.lower()
    assert "<p>да</p>" in out


def test_external_stylesheet_dropped_inline_style_kept(wc2m):
    src = ('<html><head><link rel="stylesheet" href="https://cdn/x.css">'
           '<style>b{color:#c00}</style></head><body>t</body></html>')
    out = wc2m.sanitize_offline(src)
    assert "cdn/x.css" not in out
    assert "color:#c00" in out


def test_link_rel_as_list_is_handled(wc2m):
    """bs4 отдаёт rel списком — ветка, на которой легко ошибиться."""
    out = wc2m.sanitize_offline('<link rel="preload stylesheet" href="http://a/x.css">')
    assert "x.css" not in out


def test_data_uris_are_stripped(wc2m):
    out = wc2m.strip_data_uris('<img src="data:image/png;base64,AAAABBBBCCCC">')
    assert "base64,AAAABBBB" not in out
    assert "data:," in out


def test_full_pipeline_leaves_no_external_refs(wc2m):
    src = ('<html><head><script src="http://a/x.js"></script>'
           '<link rel="stylesheet" href="http://a/x.css"></head>'
           '<body style="background-image:url(http://a/hero.jpg)">'
           '<img src="data:image/png;base64,QQQQ"></body></html>')
    out = wc2m.sanitize_offline(src)
    out = wc2m.strip_data_uris(out)
    out, _ = replace_images_with_placeholder(out)
    assert "http://a/" not in out
    assert "<img" not in out.lower()
    assert "base64" not in out


def test_webui_deblob_cuts_long_payloads(webui):
    """У WebUI блобы длинные и их много: де-блоб режет полезную нагрузку, а не тег."""
    long_uri = "data:image/png;base64," + "A" * 5000
    out = webui.deblob_data_uris(f'<img src="{long_uri}">')
    assert "A" * 5000 not in out
    assert "data:," in out
    assert len(out) < 200


def test_webui_deblob_keeps_small_icons(webui):
    """Мелкий inline-SVG — это реальная картинка почти бесплатно, его резать не надо."""
    small = "data:image/svg+xml;base64," + "A" * 40
    out = webui.deblob_data_uris(f'<img src="{small}">')
    assert small in out
