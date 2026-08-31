"""Tests for alpha_gpt.evaluate.backtester (long-short quintile backtest)."""

import numpy as np
import pandas as pd

from alpha_gpt.evaluate.backtester import (
    RET_CLIP,
    _annualized_return,
    _max_drawdown,
    backtest_alpha,
)


def _predictive(n_days=60, n_stocks=25, seed=0):
    """alpha that perfectly ranks next-day returns -> long-short should make money."""
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    cols = list(range(n_stocks))
    rs = np.random.RandomState(seed)
    fwd = pd.DataFrame(rs.randn(n_days, n_stocks) * 0.02, index=idx, columns=cols)
    alpha = fwd.copy()  # alpha == forward return -> perfect ranking
    return alpha, fwd


def test_backtest_predictive_alpha_is_profitable():
    alpha, fwd = _predictive()
    bt = backtest_alpha(alpha, fwd)
    assert bt.annual_return > 0
    assert bt.sharpe > 0
    # quintile means should be monotonically increasing (Q1 < Q5)
    q = bt.quantile_returns["mean_daily_return"]
    assert q.loc["Q5"] > q.loc["Q1"]


def test_backtest_anti_predictive_loses():
    alpha, fwd = _predictive()
    bt = backtest_alpha(-alpha, fwd)  # inverted signal
    assert bt.annual_return < 0


def test_backtest_too_few_names_returns_zero_pnl():
    idx = pd.bdate_range("2020-01-01", periods=5)
    cols = list(range(6))  # < n_quantiles*2 = 10
    a = pd.DataFrame(np.random.RandomState(0).randn(5, 6), index=idx, columns=cols)
    bt = backtest_alpha(a, a.shift(-1))
    assert (bt.portfolio_returns == 0).all()


def test_ret_clip_tames_blowup_regression():
    """REGRESSION: ret_clip bounds a single explosive print so it can't dominate P&L,
    and the clip default is shared with fast_metrics via RET_CLIP."""
    idx = pd.bdate_range("2020-01-01", periods=30)
    cols = list(range(20))
    rs = np.random.RandomState(1)
    fwd = pd.DataFrame(rs.randn(30, 20) * 0.01, index=idx, columns=cols)
    alpha = pd.DataFrame(np.tile(np.arange(20.0), (30, 1)), index=idx, columns=cols)  # col 19 always long
    fwd.iloc[:, 19] = 9.0  # +900% every day on a long-quintile name (data-error style)
    clipped = backtest_alpha(alpha, fwd)                  # default clip
    unclipped = backtest_alpha(alpha, fwd, ret_clip=None)
    assert clipped.annual_return < unclipped.annual_return
    assert RET_CLIP == (-0.95, 1.0)


def test_max_drawdown_known_series():
    cum = pd.Series([1.0, 1.5, 0.75, 1.2])  # peak 1.5 -> trough 0.75 = -50%
    assert abs(_max_drawdown(cum) - (-0.5)) < 1e-9


def test_annualized_return_known():
    # +0.1% per day for exactly one year (252 days) -> ~ (1.001**252 - 1)
    r = pd.Series([0.001] * 252)
    expected = (1.001 ** 252) ** (252 / 252) - 1
    assert abs(_annualized_return(r) - expected) < 1e-9


def test_backtest_runs_on_fixture(split):
    a = split.panels["close"]
    bt = backtest_alpha(a, split.forward_returns)
    assert isinstance(bt.sharpe, float)
    assert not np.isnan(bt.max_drawdown)
