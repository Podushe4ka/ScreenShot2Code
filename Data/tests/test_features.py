"""Признаки сложности и сводный скор.

Отдельный акцент — на двух плотностях: до 18 августа обе писались под ключом `density`,
и порог `--density-min`, откалиброванный на рендере, на статическом файле не отсекал
ничего. Тесты фиксируют, что ключи теперь разные и что режим подписан.
"""
import pytest

from conftest import load


@pytest.fixture(scope="module")
def features():
    return load("complexity", "features")


def test_static_features_basic(features, page_html):
    f = features.static_features(page_html)
    assert f["nodes"] > 10
    assert f["depth"] >= 3
    assert f["distinct_tags"] >= 10
    assert f["css_decls"] >= 4          # .card + @media + инлайновый style=
    assert f["tables"] == 1 and f["svgs"] == 1
    assert f["form_fields"] >= 3        # label + input + button
    assert f["richness"] == f["tables"] * 3 + f["form_fields"] + f["svgs"]
    assert f["render_ok"] is False
    assert f["features_mode"] == "static"


def test_static_mode_writes_static_density_only(features, page_html):
    """Ключи плотности РАЗНЫЕ. Один общий `density` — это и была ловушка."""
    f = features.static_features(page_html, tokens_code=100)
    assert f["density_static"] == pytest.approx(f["nodes"] / 100)
    assert "density_visible" not in f


def test_density_needs_tokens(features, page_html):
    f = features.static_features(page_html)
    assert "density_static" not in f and "tokens_code" not in f


def test_inline_styles_counted(features):
    a = features.static_features('<div><p>x</p></div>')["css_decls"]
    b = features.static_features('<div><p style="margin:0;padding:1px">x</p></div>')["css_decls"]
    assert b == a + 2


def test_composite_scores_are_ranks(features):
    """Сводная сложность — взвешенное среднее ПЕРЦЕНТИЛЬНЫХ РАНГОВ, значит [0, 1]."""
    feats = [features.static_features(f"<html><body>{'<div><p>x</p></div>' * k}</body></html>")
             for k in (1, 3, 10, 30)]
    scores = features.composite_scores(feats)
    assert len(scores) == len(feats)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_composite_is_monotonic_in_size(features):
    """Больше узлов при прочих равных — не меньший скор. Иначе отбор «по сложности»
    перестал бы отбирать сложное."""
    feats = [features.static_features(f"<html><body>{'<div><p>x</p></div>' * k}</body></html>")
             for k in (1, 3, 10, 30)]
    scores = features.composite_scores(feats)
    assert scores == sorted(scores)


def test_composite_survives_missing_keys(features):
    """Записи без рендер-признаков не должны ронять скоринг — только терять по рангу."""
    feats = [{"nodes": 10}, {"nodes": 200, "columns": 3}, {}]
    scores = features.composite_scores(feats)
    assert len(scores) == 3 and all(0.0 <= s <= 1.0 for s in scores)
