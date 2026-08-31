"""Tests for alpha_gpt.portfolio.construct (signal building + combination methods)."""

import numpy as np
import pandas as pd
import pytest

from alpha_gpt.portfolio.construct import (
    build_signals,
    combine,
)


def _signals(n_sig=3, n_days=12, n_stocks=10, seed=0):
    rs = np.random.RandomState(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    cols = list(range(n_stocks))
    return {i: pd.DataFrame(rs.randn(n_days, n_stocks), index=idx, columns=cols) for i in range(n_sig)}


def test_build_signals_keeps_native_sign_ignoring_is_ic(pset, panels):
    # We never sign-flip: same expression -> IDENTICAL signal regardless of is_ic sign.
    survivors = [
        {"id": 1, "expression": "cs_rank(close)", "is_ic": 0.2},
        {"id": 2, "expression": "cs_rank(close)", "is_ic": -0.2},  # opposite is_ic, NOT flipped
    ]
    out = build_signals(survivors, pset, panels)
    assert set(out) == {1, 2}
    diff = (out[1].fillna(0) - out[2].fillna(0)).abs().to_numpy()
    assert np.nanmax(diff) < 1e-9


def test_build_signals_skips_unparseable(pset, panels):
    survivors = [{"id": 9, "expression": "not_an_op(close)", "is_ic": 0.1}]
    assert build_signals(survivors, pset, panels) == {}


def test_combine_all_methods(pset):
    sigs = _signals()
    w = {i: 0.1 * (i + 1) for i in sigs}
    for method in ("equal", "orthogonalized"):
        comp, sel = combine(sigs, method=method, weights=w)
        assert comp is not None
        assert comp.shape == (12, 10)
        assert set(sel) <= set(sigs)


def test_combine_empty_returns_none():
    comp, sel = combine({}, method="equal")
    assert comp is None and sel == []


def test_combine_unknown_method_raises():
    with pytest.raises(ValueError):
        combine(_signals(), method="bogus")
