"""Tests for alpha_gpt.expr.terminal_selection (terminal screen for the GP)."""

import numpy as np
import pandas as pd

from alpha_gpt.evaluate.metrics import compute_ic
from alpha_gpt.expr.terminal_selection import fast_ic, select_terminals


def _setup(n_days=12, n_stocks=20, seed=0):
    rs = np.random.RandomState(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    cols = list(range(n_stocks))
    fwd = pd.DataFrame(rs.randn(n_days, n_stocks), index=idx, columns=cols)
    close = pd.DataFrame(rs.randn(n_days, n_stocks), index=idx, columns=cols)  # ~0 IC core
    good = fwd.copy()                                  # dense, high |IC|
    sparse = pd.DataFrame(np.nan, index=idx, columns=cols)
    sparse.iloc[:, :3] = fwd.iloc[:, :3]              # high |IC| but breadth 3
    return {"close": close, "good": good, "sparse": sparse}, fwd


def test_fast_ic_matches_compute_ic():
    panels, fwd = _setup()
    assert abs(fast_ic(panels["good"], fwd) - float(compute_ic(panels["good"], fwd).mean())) < 1e-12
    assert isinstance(fast_ic(panels["good"], fwd), float)


def test_select_terminals_keeps_core_and_top_ic():
    panels, fwd = _setup()
    sel = select_terminals(panels, fwd, candidates=["close", "good", "sparse"],
                           core=["close"], max_terminals=10, min_breadth=0)
    assert "close" in sel        # core always kept (even with ~0 IC)
    assert "good" in sel         # high |IC| extra selected


def test_select_terminals_breadth_screen_excludes_sparse_regression():
    """REGRESSION: a thin field can't win a GP-terminal seat on a spurious high |IC|."""
    panels, fwd = _setup()
    sel = select_terminals(panels, fwd, candidates=["close", "good", "sparse"],
                           core=["close"], max_terminals=10, min_breadth=10)
    assert "good" in sel
    assert "sparse" not in sel   # breadth 3 < 10 -> screened out


def test_select_terminals_respects_cap():
    panels, fwd = _setup()
    sel = select_terminals(panels, fwd, candidates=["close", "good", "sparse"],
                           core=["close"], max_terminals=1, min_breadth=0)
    assert sel == ["close"]      # cap leaves room only for the core
