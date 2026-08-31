"""Tests for alpha_gpt.evaluate.metrics (IC, ICIR, turnover)."""

import numpy as np
import pandas as pd

from alpha_gpt.evaluate.metrics import compute_ic, compute_icir, compute_turnover


def _ramp(n_days=10, n_stocks=30):
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    cols = list(range(n_stocks))
    return pd.DataFrame(np.tile(np.arange(n_stocks, dtype=float), (n_days, 1)), index=idx, columns=cols)


def test_compute_ic_perfect_positive():
    a = _ramp()
    ic = compute_ic(a, a.copy())  # identical ranking each day -> Spearman 1
    assert len(ic) == 10
    assert (ic > 0.999).all()


def test_compute_ic_perfect_negative():
    a = _ramp()
    ic = compute_ic(a, -a)
    assert (ic < -0.999).all()


def test_compute_ic_skips_days_with_few_names():
    # < 20 common names on a day -> that day is skipped entirely.
    idx = pd.bdate_range("2020-01-01", periods=4)
    cols = list(range(10))
    a = pd.DataFrame(np.random.RandomState(0).randn(4, 10), index=idx, columns=cols)
    ic = compute_ic(a, a.shift(-1))
    assert ic.empty


def test_compute_ic_ignores_nan_pairs():
    # NaNs are dropped pairwise; remaining common support still correlates.
    a = _ramp()
    f = a.copy()
    f.iloc[:, :5] = np.nan  # 25 names remain (>= 20)
    ic = compute_ic(a, f)
    assert (ic > 0.999).all()


def test_compute_ic_uses_common_support_only_regression():
    """REGRESSION: only cells valid in BOTH series enter the correlation.

    A one-sided NaN must not leak into just the numerator or just the denominator, which
    would bias IC toward 0 (worst for sparse fundamentals paired with dense returns). Here
    the common support is perfectly rank-correlated, with extra one-sided NaNs on each
    side; IC must stay ~1.
    """
    idx = pd.bdate_range("2020-01-01", periods=5)
    cols = list(range(30))
    a = pd.DataFrame(np.nan, index=idx, columns=cols)
    f = pd.DataFrame(np.nan, index=idx, columns=cols)
    for d in idx:
        for j in range(25):      # 25 common (>= 20 floor), perfectly correlated
            a.loc[d, j] = j
            f.loc[d, j] = j
        a.loc[d, 25] = 999       # alpha-only (f is NaN)
        f.loc[d, 26] = -999      # fwd-only (a is NaN)
    ic = compute_ic(a, f)
    assert len(ic) == 5
    assert (ic > 0.99).all()


def test_compute_icir_basic():
    s = pd.Series([0.1, 0.2, 0.3])
    assert compute_icir(s) == float(s.mean() / s.std())


def test_compute_icir_zero_std_is_zero():
    assert compute_icir(pd.Series([0.1, 0.1, 0.1])) == 0.0
    assert compute_icir(pd.Series([], dtype=float)) == 0.0


def test_compute_turnover_constant_signal_is_zero():
    idx = pd.bdate_range("2020-01-01", periods=10)
    cols = list(range(20))
    const = pd.DataFrame(np.tile(np.arange(20.0), (10, 1)), index=idx, columns=cols)
    assert compute_turnover(const) < 1e-9


def test_compute_turnover_reversing_signal_is_high():
    idx = pd.bdate_range("2020-01-01", periods=10)
    cols = list(range(20))
    rows = [np.arange(20.0) if i % 2 == 0 else np.arange(20.0)[::-1] for i in range(10)]
    flip = pd.DataFrame(rows, index=idx, columns=cols)
    # full rank reversal every day -> average |rank change| near the maximum (~0.5+)
    assert compute_turnover(flip) > 0.4
