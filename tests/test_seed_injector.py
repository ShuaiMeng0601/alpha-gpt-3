"""Tests for alpha_gpt.expr.seed_injector (LLM-string -> DEAP tree)."""

from alpha_gpt.expr.primitives import create_primitive_set
from alpha_gpt.expr.seed_injector import normalize_expression, parse_expression


def test_normalize_curries_windowed_ops():
    assert normalize_expression("cs_rank(ts_delta(close, 5))") == "cs_rank(ts_delta_5(close))"
    assert normalize_expression("ts_mean(volume, 20)") == "ts_mean_20(volume)"


def test_normalize_aliases():
    assert normalize_expression("rank(returns)") == "cs_rank(returns)"
    assert normalize_expression("zscore(close)") == "cs_zscore(close)"


def test_normalize_default_window_when_omitted():
    # ts_delta with no window -> default window (5)
    assert normalize_expression("ts_delta(close)") == "ts_delta_5(close)"


def test_normalize_does_not_double_prefix_curried():
    # already-curried names must not be mangled
    assert normalize_expression("ts_delta_5(close)") == "ts_delta_5(close)"
    assert normalize_expression("cs_rank(close)") == "cs_rank(close)"


def test_parse_valid_expression_returns_tree():
    pset = create_primitive_set(["close", "volume", "returns"])
    assert parse_expression("cs_rank(ts_delta(close, 5))", pset) is not None
    assert parse_expression("add(close, neg(volume))", pset) is not None


def test_parse_invalid_returns_none():
    pset = create_primitive_set(["close", "volume"])
    assert parse_expression("not_an_op(close)", pset) is None
    assert parse_expression("cs_rank(unknown_terminal)", pset) is None
    assert parse_expression("((((", pset) is None
