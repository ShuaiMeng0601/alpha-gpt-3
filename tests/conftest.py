"""Shared pytest fixtures.

Everything here is small, deterministic, in-memory synthetic data — no real data
files, no network, no LLM, no GP evolution — so the whole suite is fast and stable.
"""

import numpy as np
import pandas as pd
import pytest

from alpha_gpt.data.loader import DataSplit
from alpha_gpt.factory import db
from alpha_gpt.expr.primitives import create_primitive_set

N_DAYS = 80
N_STOCKS = 30
TERMINALS = ["close", "open", "high", "low", "volume", "returns"]


@pytest.fixture
def dates():
    return pd.bdate_range("2020-01-01", periods=N_DAYS)


@pytest.fixture
def stocks():
    return list(range(1, N_STOCKS + 1))


@pytest.fixture
def panels(dates, stocks):
    """Synthetic date x PERMNO panels with a forward_returns target (returns.shift(-1))."""
    rs = np.random.RandomState(42)
    n, k = len(dates), len(stocks)
    close = pd.DataFrame(rs.randn(n, k).cumsum(0), index=dates, columns=stocks) + 100.0
    rets = close.pct_change().fillna(0.0)
    vol = pd.DataFrame(rs.rand(n, k) * 1e6 + 1e3, index=dates, columns=stocks)
    p = {
        "close": close,
        "open": close.shift(1).bfill(),
        "high": close * 1.01,
        "low": close * 0.99,
        "volume": vol,
        "returns": rets,
    }
    p["forward_returns"] = rets.shift(-1)
    return p


@pytest.fixture
def split(panels):
    """A DataSplit whose .panels excludes forward_returns (matches loader.split_data)."""
    terms = {k: v for k, v in panels.items() if k != "forward_returns"}
    return DataSplit(panels=terms, forward_returns=panels["forward_returns"])


@pytest.fixture
def pset():
    return create_primitive_set(TERMINALS)


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    yield c
    c.close()


# --- helpers (plain functions used inline by tests via `from conftest import ...`
#     is avoided; instead these are exposed as fixtures returning callables) ---

@pytest.fixture
def make_panel():
    """Factory: 2D array -> DataFrame on a fixed daily grid."""
    def _make(arr, start="2021-01-01"):
        arr = np.asarray(arr, dtype=float)
        idx = pd.bdate_range(start, periods=arr.shape[0])
        cols = list(range(arr.shape[1]))
        return pd.DataFrame(arr, index=idx, columns=cols)
    return _make
