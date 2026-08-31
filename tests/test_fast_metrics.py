"""Tests for alpha_gpt.factory.fast_metrics — the fast factor-return Sharpe proxy.

The IC path is the canonical compute_ic (tested in test_metrics.py). Includes a
regression for a fixed bug: summarize annualization produced NaN on a sub--100%
factor-return day.
"""

import numpy as np
import pandas as pd

from alpha_gpt.evaluate.backtester import RET_CLIP
from alpha_gpt.factory.fast_metrics import factor_returns, summarize, summarize_full


def test_summarize_full_tstat_is_icir_times_sqrt_days(panels):
    """t-stat of the mean IC = ICIR x sqrt(#IC days); ic_days_scale multiplies the day count
    to recover the full-sample t-stat when IC was measured on a date-subsample."""
    panel, fwd = panels["close"], panels["forward_returns"]
    ic, icir, tstat, *_ = summarize_full(panel, fwd)
    # scaling the (effective) day count by 4 scales the t-stat by sqrt(4) = 2, ICIR unchanged
    ic4, icir4, tstat4, *_ = summarize_full(panel, fwd, ic_days_scale=4)
    assert icir4 == icir and ic4 == ic
    assert abs(tstat4 - 2.0 * tstat) < 1e-9
    assert (tstat > 0) == (icir > 0)  # t-stat carries the sign of the ICIR


def test_summarize_matches_summarize_full(panels):
    """The 5-tuple summarize is the t-stat-free view of summarize_full (no drift)."""
    panel, fwd = panels["close"], panels["forward_returns"]
    ic, icir, sharpe, ann, dd = summarize(panel, fwd)
    fic, ficir, _tstat, fsharpe, fann, fdd = summarize_full(panel, fwd)
    assert (ic, icir, sharpe, ann, dd) == (fic, ficir, fsharpe, fann, fdd)


def test_factor_returns_clips_extremes():
    idx = pd.bdate_range("2020-01-01", periods=3)
    cols = list(range(10))
    panel = pd.DataFrame(np.tile(np.arange(10.0), (3, 1)), index=idx, columns=cols)  # col 9 = top rank
    fwd = pd.DataFrame(0.0, index=idx, columns=cols)
    fwd.iloc[:, 9] = 10.0  # +1000% on the most heavily long-weighted name
    clipped = factor_returns(panel, fwd)                 # default RET_CLIP
    unclipped = factor_returns(panel, fwd, ret_clip=None)
    assert (clipped < unclipped).all()
    assert RET_CLIP == (-0.95, 1.0)


def test_summarize_no_nan_on_crash_day_regression():
    """REGRESSION: a dispersed crash day must not yield NaN annual_return.

    The clip keeps factor returns > -100% so the cumulative wealth stays positive
    (negative base ** fractional power = NaN otherwise), and summarize also guards it.
    """
    idx = pd.bdate_range("2020-01-01", periods=40)
    cols = list(range(20))
    rs = np.random.RandomState(0)
    panel = pd.DataFrame(rs.randn(40, 20), index=idx, columns=cols)
    fwd = pd.DataFrame(rs.randn(40, 20) * 0.01, index=idx, columns=cols)
    # crash day: top-ranked names tank, bottom-ranked names spike -> very negative factor return
    order = panel.iloc[10].rank()
    fwd.iloc[10] = np.where(order > 10, -9.0, 9.0)
    ic, icir, sharpe, ann, dd = summarize(panel, fwd)
    assert np.isfinite(ann)
    assert np.isfinite(sharpe)
    assert np.isfinite(dd)


def test_summarize_returns_five_floats(panels):
    out = summarize(panels["close"], panels["forward_returns"])
    assert len(out) == 5
    assert all(isinstance(x, float) for x in out)
