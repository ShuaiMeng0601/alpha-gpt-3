"""Exhaustive correctness tests for the long-short quintile backtester.

The backtester is the foundation of every reported number, so this file pins it down
hard: hand-computed known answers, an INDEPENDENT reference implementation cross-check,
algebraic invariants (sign-flip, monotone-transform invariance), metric-formula checks,
NaN/alignment/clip edge cases, and regression guards for two fixed bugs
(tied-signal quantile assignment; NaN annualized return on a wipeout).
"""

import numpy as np
import pandas as pd
import pytest

from alpha_gpt.evaluate.backtester import (
    RET_CLIP,
    BacktestResult,
    _annualized_return,
    _annualized_sharpe,
    _max_drawdown,
    backtest_alpha,
)

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _panel(rows, start="2020-01-01"):
    """rows: 2D array-like (days × stocks) -> DataFrame on a business-day grid."""
    arr = np.asarray(rows, dtype=float)
    idx = pd.bdate_range(start, periods=arr.shape[0])
    return pd.DataFrame(arr, index=idx, columns=list(range(arr.shape[1])))


def _random_panels(n_days, n_stocks, seed):
    rs = np.random.RandomState(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    cols = list(range(n_stocks))
    # randn -> effectively-unique values (no ties) so quintiles are well defined
    alpha = pd.DataFrame(rs.randn(n_days, n_stocks), index=idx, columns=cols)
    fwd = pd.DataFrame(rs.randn(n_days, n_stocks) * 0.02, index=idx, columns=cols)
    return alpha, fwd


def _reference_daily_ls(alpha_row, ret_row, n_quantiles=5, clip=RET_CLIP):
    """Independent top-bin-minus-bottom-bin computed by SORTING (no qcut).

    Valid as a reference only when, after dropping NaNs, the surviving stock count is
    divisible by n_quantiles and alpha values are unique — then qcut's equal-frequency
    bins are exactly the sorted top/bottom k, with no tie ambiguity.
    """
    a = alpha_row.dropna()
    r = ret_row.dropna()
    common = a.index.intersection(r.index)
    a, r = a[common], r[common]
    if len(common) < n_quantiles * 2:
        return 0.0
    if clip is not None:
        r = r.clip(*clip)
    k = len(common) // n_quantiles
    order = a.sort_values(kind="stable").index  # ascending alpha
    top, bottom = order[-k:], order[:k]
    return float(r[top].mean() - r[bottom].mean())


def _reference_series(alpha, fwd, n_quantiles=5, clip=RET_CLIP):
    dates = alpha.index.intersection(fwd.index)
    return pd.Series(
        [_reference_daily_ls(alpha.loc[d], fwd.loc[d], n_quantiles, clip) for d in dates],
        index=dates,
    )


# ===========================================================================
# 1. Known-answer (hand-computed)
# ===========================================================================

def test_single_day_long_short_handcomputed():
    # 10 stocks, 5 quintiles -> 2 per bin. alpha rank == column order.
    # Q1={0,1}, Q5={8,9}. ls = mean(ret 8,9) - mean(ret 0,1).
    alpha = _panel([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    fwd = _panel([[0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]])
    bt = backtest_alpha(alpha, fwd, ret_clip=None)
    expected = (0.08 + 0.09) / 2 - (0.00 + 0.01) / 2  # 0.085 - 0.005 = 0.08
    assert bt.portfolio_returns.iloc[0] == pytest.approx(0.08)
    assert expected == pytest.approx(0.08)


def test_single_day_quantile_means_handcomputed():
    alpha = _panel([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    fwd = _panel([[0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]])
    q = backtest_alpha(alpha, fwd, ret_clip=None).quantile_returns["mean_daily_return"]
    assert q.loc["Q1"] == pytest.approx(0.005)
    assert q.loc["Q2"] == pytest.approx(0.025)
    assert q.loc["Q3"] == pytest.approx(0.045)
    assert q.loc["Q4"] == pytest.approx(0.065)
    assert q.loc["Q5"] == pytest.approx(0.085)


def test_two_day_cumulative_handcomputed():
    alpha = _panel([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])      # ranking reversed day 2
    fwd = _panel([[0, 0, 0, 0, 0, 0, 0, 0, 0.10, 0.10],   # day1: top two (8,9) earn .10
                  [0.10, 0.10, 0, 0, 0, 0, 0, 0, 0, 0]])   # day2: stocks 0,1 are now top
    bt = backtest_alpha(alpha, fwd, ret_clip=None)
    assert bt.portfolio_returns.iloc[0] == pytest.approx(0.10)   # long .10, short 0
    assert bt.portfolio_returns.iloc[1] == pytest.approx(0.10)   # long .10, short 0
    assert bt.cumulative_returns.iloc[0] == pytest.approx(1.10)
    assert bt.cumulative_returns.iloc[1] == pytest.approx(1.10 * 1.10)


def test_long_and_short_both_positive_ls_is_difference():
    # long bucket returns 0.05, short bucket returns 0.02 -> ls = 0.03 (not 0.05)
    alpha = _panel([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    fwd = _panel([[0.02, 0.02, 0, 0, 0, 0, 0, 0, 0.05, 0.05]])
    bt = backtest_alpha(alpha, fwd, ret_clip=None)
    assert bt.portfolio_returns.iloc[0] == pytest.approx(0.05 - 0.02)


@pytest.mark.parametrize("n_quantiles,n_stocks", [(2, 10), (3, 9), (4, 12), (5, 10), (10, 20)])
def test_n_quantiles_param_handcomputed_topbottom(n_quantiles, n_stocks):
    # alpha == column index; ret == column index scaled. Top k and bottom k known.
    k = n_stocks // n_quantiles
    alpha = _panel([list(range(n_stocks))])
    fwd = _panel([[0.001 * i for i in range(n_stocks)]])
    bt = backtest_alpha(alpha, fwd, n_quantiles=n_quantiles, ret_clip=None)
    top = np.mean([0.001 * i for i in range(n_stocks - k, n_stocks)])
    bot = np.mean([0.001 * i for i in range(k)])
    assert bt.portfolio_returns.iloc[0] == pytest.approx(top - bot)


# ===========================================================================
# 2. Metric-formula correctness (helpers in isolation)
# ===========================================================================

def test_annualized_sharpe_known_value():
    r = pd.Series([0.01, 0.02, 0.03])  # mean .02, sample std .01
    assert _annualized_sharpe(r) == pytest.approx(np.sqrt(TRADING_DAYS) * 0.02 / 0.01)


def test_annualized_sharpe_uses_sample_std_ddof1():
    r = pd.Series([0.0, 0.0, 0.0, 0.04])
    expected = np.sqrt(TRADING_DAYS) * r.mean() / r.std(ddof=1)
    assert _annualized_sharpe(r) == pytest.approx(expected)


def test_annualized_sharpe_zero_when_constant():
    assert _annualized_sharpe(pd.Series([0.01, 0.01, 0.01])) == 0.0


def test_annualized_sharpe_zero_when_single_obs():
    assert _annualized_sharpe(pd.Series([0.05])) == 0.0


def test_max_drawdown_known_series():
    cum = pd.Series([1.0, 1.5, 0.75, 1.2])  # peak 1.5 -> trough 0.75 = -50%
    assert _max_drawdown(cum) == pytest.approx(-0.5)


def test_max_drawdown_monotone_increasing_is_zero():
    assert _max_drawdown(pd.Series([1.0, 1.1, 1.2, 1.3])) == pytest.approx(0.0)


def test_max_drawdown_trough_at_end():
    cum = pd.Series([1.0, 2.0, 1.0])  # -50% from peak 2.0
    assert _max_drawdown(cum) == pytest.approx(-0.5)


def test_max_drawdown_empty_is_zero():
    assert _max_drawdown(pd.Series([], dtype=float)) == 0.0


def test_annualized_return_one_year_constant():
    r = pd.Series([0.001] * TRADING_DAYS)
    assert _annualized_return(r) == pytest.approx(1.001 ** TRADING_DAYS - 1)


def test_annualized_return_zero_returns():
    assert _annualized_return(pd.Series([0.0] * 100)) == pytest.approx(0.0)


def test_annualized_return_empty():
    assert _annualized_return(pd.Series([], dtype=float)) == 0.0


def test_annualized_return_half_year_compounds():
    # 126 days (~half year) of 0.001 -> annualized = total ** (1/(126/252)) - 1
    r = pd.Series([0.001] * 126)
    total = 1.001 ** 126
    assert _annualized_return(r) == pytest.approx(total ** (TRADING_DAYS / 126) - 1)


def test_annualized_return_wipeout_returns_minus_one_REGRESSION():
    """REGRESSION: a daily LS spread <= -100% drives total wealth <= 0. A fractional
    power of a negative base is NaN; the backtester must report -1.0 (full loss)."""
    r = pd.Series([-1.5] + [0.001] * 299)   # fractional annualization exponent
    out = _annualized_return(r)
    assert np.isfinite(out)
    assert out == pytest.approx(-1.0)


# ===========================================================================
# 3. Directional correctness / sign
# ===========================================================================

def test_perfect_predictor_is_profitable():
    alpha, fwd = _random_panels(60, 25, seed=0)
    alpha = fwd.copy()  # alpha == next-day return: perfect ranking
    bt = backtest_alpha(alpha, fwd)
    assert bt.annual_return > 0
    assert bt.sharpe > 0
    assert bt.portfolio_returns.mean() > 0


def test_anti_predictor_loses_money():
    alpha, fwd = _random_panels(60, 25, seed=0)
    bt = backtest_alpha(-fwd, fwd)
    assert bt.annual_return < 0
    assert bt.portfolio_returns.mean() < 0


def test_quantiles_monotonic_for_perfect_signal():
    alpha, fwd = _random_panels(60, 25, seed=3)
    q = backtest_alpha(fwd.copy(), fwd).quantile_returns["mean_daily_return"]
    vals = [q.loc[f"Q{i}"] for i in range(1, 6)]
    assert all(b > a for a, b in zip(vals, vals[1:])), vals  # strictly increasing


# ===========================================================================
# 4. Algebraic invariants
# ===========================================================================

def test_negating_alpha_flips_daily_ls_sign():
    """Negating the signal swaps top<->bottom quintiles exactly (n_stocks % q == 0,
    unique values), so every daily LS return flips sign."""
    alpha, fwd = _random_panels(40, 20, seed=1)
    pos = backtest_alpha(alpha, fwd).portfolio_returns
    neg = backtest_alpha(-alpha, fwd).portfolio_returns
    assert np.allclose(neg.values, -pos.values)


@pytest.mark.parametrize("transform", [
    lambda a: a + 100.0,        # additive shift
    lambda a: a * 2.5,          # positive scale
    lambda a: a * 1e-6,         # tiny positive scale
    lambda a: np.arctan(a),     # strictly increasing nonlinearity
])
def test_monotone_increasing_transform_is_invariant(transform):
    """The backtest depends only on the cross-sectional RANKING, so any strictly
    increasing transform of the signal must give identical P&L."""
    alpha, fwd = _random_panels(40, 20, seed=2)
    base = backtest_alpha(alpha, fwd).portfolio_returns
    transformed = backtest_alpha(transform(alpha), fwd).portfolio_returns
    assert np.allclose(base.values, transformed.values)


def test_constant_alpha_gives_zero_pnl_no_crash():
    alpha = _panel(np.full((10, 15), 7.0))            # identical everywhere
    _, fwd = _random_panels(10, 15, seed=5)
    fwd.index = alpha.index
    bt = backtest_alpha(alpha, fwd)
    assert (bt.portfolio_returns == 0).all()
    assert bt.sharpe == 0.0
    assert np.isfinite(bt.annual_return)


# ===========================================================================
# 5. Independent reference cross-check (the strongest test)
# ===========================================================================

@pytest.mark.parametrize("seed", range(8))
def test_matches_independent_reference_quintiles(seed):
    alpha, fwd = _random_panels(50, 20, seed=seed)      # 20 % 5 == 0, unique values
    bt = backtest_alpha(alpha, fwd)                      # default clip
    ref = _reference_series(alpha, fwd, n_quantiles=5, clip=RET_CLIP)
    assert np.allclose(bt.portfolio_returns.values, ref.values)


@pytest.mark.parametrize("n_quantiles,n_stocks", [(2, 20), (4, 20), (5, 20), (10, 20)])
def test_matches_reference_various_n_quantiles(n_quantiles, n_stocks):
    alpha, fwd = _random_panels(40, n_stocks, seed=11)
    bt = backtest_alpha(alpha, fwd, n_quantiles=n_quantiles)
    ref = _reference_series(alpha, fwd, n_quantiles=n_quantiles, clip=RET_CLIP)
    assert np.allclose(bt.portfolio_returns.values, ref.values)


def test_matches_reference_no_clip():
    alpha, fwd = _random_panels(40, 20, seed=7)
    bt = backtest_alpha(alpha, fwd, ret_clip=None)
    ref = _reference_series(alpha, fwd, n_quantiles=5, clip=None)
    assert np.allclose(bt.portfolio_returns.values, ref.values)


# ===========================================================================
# 6. Tie handling (regression for the fixed quantile-assignment bug)
# ===========================================================================

def test_tied_low_block_keeps_highs_in_long_REGRESSION():
    """REGRESSION: a cross-section with many tied-low names + a few genuinely high
    ones must put the highs in the LONG leg. Value-qcut(duplicates='drop') used to
    collapse everything into bin 1, shorting the highs (P&L sign inverted)."""
    alpha = _panel([[0.0] * 10 + [1.0, 1.0]])           # 10 tied low, 2 high
    fwd = _panel([[0.0] * 10 + [0.05, 0.05]])           # the highs earn +5%
    bt = backtest_alpha(alpha, fwd, ret_clip=None)
    assert bt.portfolio_returns.iloc[0] > 0             # long the winners, not short them
    # the two high names must land in the top quantile
    assert bt.quantile_returns.loc["Q5", "mean_daily_return"] > 0


def test_tied_high_block_keeps_lows_in_short():
    alpha = _panel([[0.0, 0.0] + [1.0] * 10])           # 2 low, 10 tied high
    fwd = _panel([[-0.05, -0.05] + [0.0] * 10])         # the lows drop -5%
    bt = backtest_alpha(alpha, fwd, ret_clip=None)
    assert bt.portfolio_returns.iloc[0] > 0             # short the losers -> positive LS


def test_heavy_ties_do_not_crash_and_bins_balanced():
    rs = np.random.RandomState(0)
    # signal takes only 3 distinct values across 30 stocks (lots of ties)
    alpha = _panel(rs.randint(0, 3, size=(8, 30)).astype(float))
    fwd = _panel(rs.randn(8, 30) * 0.02)
    fwd.index = alpha.index
    bt = backtest_alpha(alpha, fwd)
    assert len(bt.portfolio_returns) == 8
    assert np.isfinite(bt.portfolio_returns).all()
    # every quintile actually gets populated (ranking prevents bin collapse)
    assert (bt.quantile_returns["count"] > 0).all()


def test_no_tie_results_identical_to_value_qcut_reference():
    """Sanity: for tie-free signals the rank-based assignment equals the old
    value-based qcut, so the fix changed nothing on the common path."""
    alpha, fwd = _random_panels(30, 20, seed=9)
    bt = backtest_alpha(alpha, fwd)
    # reference uses pure sorting (value order) -> must match the rank-based engine
    ref = _reference_series(alpha, fwd)
    assert np.allclose(bt.portfolio_returns.values, ref.values)


# ===========================================================================
# 7. NaN / missing-data handling
# ===========================================================================

def test_nan_alpha_entries_dropped_pairwise():
    # 12 stocks, 2 with NaN alpha -> effectively a 10-stock cross-section.
    full = _panel([[float(i) for i in range(12)]])
    full.iloc[0, 0] = np.nan
    full.iloc[0, 5] = np.nan
    fwd = _panel([[0.001 * i for i in range(12)]])
    bt = backtest_alpha(full, fwd, ret_clip=None)

    # Independent: drop the two NaN columns and backtest the remaining 10.
    keep = [c for c in full.columns if c not in (0, 5)]
    bt_drop = backtest_alpha(full[keep], fwd[keep], ret_clip=None)
    assert bt.portfolio_returns.iloc[0] == pytest.approx(bt_drop.portfolio_returns.iloc[0])


def test_nan_return_entries_dropped():
    alpha = _panel([[float(i) for i in range(12)]])
    fwd = _panel([[0.001 * i for i in range(12)]])
    fwd.iloc[0, 11] = np.nan      # drop the top-alpha name's return
    bt = backtest_alpha(alpha, fwd, ret_clip=None)
    keep = [c for c in alpha.columns if c != 11]
    bt_drop = backtest_alpha(alpha[keep], fwd[keep], ret_clip=None)
    assert bt.portfolio_returns.iloc[0] == pytest.approx(bt_drop.portfolio_returns.iloc[0])


def test_all_nan_day_is_zero():
    alpha = _panel([[np.nan] * 15, [float(i) for i in range(15)]])
    fwd = _panel([[0.01] * 15, [0.001 * i for i in range(15)]])
    bt = backtest_alpha(alpha, fwd)
    assert bt.portfolio_returns.iloc[0] == 0.0          # all-NaN day -> flat
    assert bt.portfolio_returns.iloc[1] != 0.0


def test_all_nan_panel_finite_metrics():
    alpha = _panel(np.full((10, 15), np.nan))
    _, fwd = _random_panels(10, 15, seed=4)
    fwd.index = alpha.index
    bt = backtest_alpha(alpha, fwd)
    assert (bt.portfolio_returns == 0).all()
    assert bt.sharpe == 0.0
    assert np.isfinite(bt.max_drawdown)
    assert np.isfinite(bt.annual_return)


# ===========================================================================
# 8. Alignment / look-ahead contract / shape
# ===========================================================================

def test_only_common_dates_used():
    alpha, fwd = _random_panels(40, 20, seed=6)
    fwd2 = fwd.iloc[5:]                                   # drop first 5 dates of fwd
    bt = backtest_alpha(alpha, fwd2)
    expected = alpha.index.intersection(fwd2.index)
    assert list(bt.portfolio_returns.index) == list(expected)


def test_alpha_on_day_T_pairs_with_forward_return_on_day_T():
    """Contract: the backtester does NOT shift returns; alpha[T] is paired with
    forward_returns[T] (the caller supplies already-forward returns)."""
    # Day 0 alpha ranks stock 9 top; only day-0 forward return is nonzero.
    alpha = _panel([[float(i) for i in range(10)], [float(i) for i in range(10)]])
    fwd = _panel([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0.10],     # day 0: top name earns .10
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00]])     # day 1: nothing
    bt = backtest_alpha(alpha, fwd, ret_clip=None)
    assert bt.portfolio_returns.iloc[0] == pytest.approx(0.05)  # long {8,9}: (0+.10)/2
    assert bt.portfolio_returns.iloc[1] == pytest.approx(0.0)


def test_extra_stocks_in_one_panel_ignored():
    alpha = _panel([[float(i) for i in range(12)]])
    fwd = _panel([[0.001 * i for i in range(10)]])        # only 10 stocks
    bt = backtest_alpha(alpha, fwd, ret_clip=None)
    # intersection is 10 stocks -> matches backtesting the aligned 10
    bt2 = backtest_alpha(alpha[list(range(10))], fwd, ret_clip=None)
    assert bt.portfolio_returns.iloc[0] == pytest.approx(bt2.portfolio_returns.iloc[0])


def test_result_is_dataclass_with_expected_shapes():
    alpha, fwd = _random_panels(20, 15, seed=8)
    bt = backtest_alpha(alpha, fwd)
    assert isinstance(bt, BacktestResult)
    assert isinstance(bt.portfolio_returns, pd.Series)
    assert isinstance(bt.cumulative_returns, pd.Series)
    assert list(bt.quantile_returns.index) == [f"Q{i}" for i in range(1, 6)]
    assert set(bt.quantile_returns.columns) == {"mean_daily_return", "count"}
    for v in (bt.sharpe, bt.max_drawdown, bt.annual_return):
        assert isinstance(v, float)


def test_cumulative_returns_equals_cumprod():
    alpha, fwd = _random_panels(30, 20, seed=10)
    bt = backtest_alpha(alpha, fwd)
    expected = (1 + bt.portfolio_returns).cumprod()
    assert np.allclose(bt.cumulative_returns.values, expected.values)


# ===========================================================================
# 9. ret_clip behavior
# ===========================================================================

def test_ret_clip_caps_extreme_long_return():
    # Stock 9 is always top-alpha; its +900% prints should clip to +100%.
    alpha = _panel([[float(i) for i in range(10)]])
    fwd = _panel([[0.0] * 9 + [9.0]])                    # +900% on the long name
    clipped = backtest_alpha(alpha, fwd)                 # default clip (-0.95, 1.0)
    unclipped = backtest_alpha(alpha, fwd, ret_clip=None)
    # long = {8,9}: clipped (0 + 1.0)/2 = 0.5 ; unclipped (0 + 9.0)/2 = 4.5
    assert clipped.portfolio_returns.iloc[0] == pytest.approx(0.5)
    assert unclipped.portfolio_returns.iloc[0] == pytest.approx(4.5)


def test_ret_clip_floors_extreme_negative_return():
    alpha = _panel([[float(i) for i in range(10)]])
    fwd = _panel([[-3.0] + [0.0] * 9])                   # -300% on the bottom name
    clipped = backtest_alpha(alpha, fwd)                 # floor -0.95
    # short = {0,1}: clipped (-0.95 + 0)/2 = -0.475 ; ls = long(0) - short(-0.475)
    assert clipped.portfolio_returns.iloc[0] == pytest.approx(0.475)


def test_ret_clip_none_uses_raw_returns():
    alpha, fwd = _random_panels(30, 20, seed=12)
    fwd.iloc[0, :] = 5.0                                  # extreme day
    raw = backtest_alpha(alpha, fwd, ret_clip=None)
    ref = _reference_series(alpha, fwd, clip=None)
    assert np.allclose(raw.portfolio_returns.values, ref.values)


def test_ret_clip_default_matches_module_constant():
    assert RET_CLIP == (-0.95, 1.0)


# ===========================================================================
# 10. Thin days / minimum names
# ===========================================================================

def test_too_few_names_returns_zero_pnl():
    alpha, fwd = _random_panels(5, 6, seed=0)            # 6 < n_quantiles*2 = 10
    bt = backtest_alpha(alpha, fwd)
    assert (bt.portfolio_returns == 0).all()


def test_exactly_minimum_names_trades():
    alpha = _panel([[float(i) for i in range(10)]])      # exactly 10 = 5*2
    fwd = _panel([[0.001 * i for i in range(10)]])
    bt = backtest_alpha(alpha, fwd, ret_clip=None)
    assert bt.portfolio_returns.iloc[0] != 0.0
    assert (bt.quantile_returns["count"] == 1).all()     # 2 stocks/bin, 1 day each


def test_mixed_thin_and_full_days():
    # day 0 thin (6 stocks via NaN), day 1 full (12) -> day0=0, day1 trades
    alpha = _panel([[float(i) for i in range(12)], [float(i) for i in range(12)]])
    alpha.iloc[0, 6:] = np.nan
    fwd = _panel([[0.001 * i for i in range(12)], [0.001 * i for i in range(12)]])
    bt = backtest_alpha(alpha, fwd, ret_clip=None)
    assert bt.portfolio_returns.iloc[0] == 0.0
    assert bt.portfolio_returns.iloc[1] != 0.0


# ===========================================================================
# 11. Empty / degenerate inputs
# ===========================================================================

def test_empty_panels_do_not_crash():
    empty = pd.DataFrame()
    bt = backtest_alpha(empty, empty)
    assert bt.portfolio_returns.empty
    assert bt.sharpe == 0.0
    assert bt.max_drawdown == 0.0
    assert bt.annual_return == 0.0


def test_disjoint_dates_give_empty_result():
    alpha = _panel([[float(i) for i in range(10)]], start="2020-01-01")
    fwd = _panel([[0.001 * i for i in range(10)]], start="2021-01-01")
    bt = backtest_alpha(alpha, fwd)
    assert bt.portfolio_returns.empty


# ===========================================================================
# 12. End-to-end "golden" tests (full backtest -> externally hand-computed metrics)
# ===========================================================================

def test_end_to_end_golden_metrics():
    """A 3-day panel engineered so the daily LS returns are exactly [0.10, -0.05, 0.03],
    then check the WHOLE BacktestResult against numbers computed outside the codebase.
    This closes the loop between the quintile P&L and the metric helpers."""
    alpha = _panel([[float(i) for i in range(10)]] * 3)               # constant ranking
    fwd = _panel([[0, 0, 0, 0, 0, 0, 0, 0, 0.10, 0.10],              # long earns +.10
                  [0, 0, 0, 0, 0, 0, 0, 0, -0.05, -0.05],            # long earns -.05
                  [0, 0, 0, 0, 0, 0, 0, 0, 0.03, 0.03]])             # long earns +.03
    bt = backtest_alpha(alpha, fwd, ret_clip=None)

    # daily LS = long mean - short mean (short bucket is all zeros)
    assert np.allclose(bt.portfolio_returns.values, [0.10, -0.05, 0.03])

    # Sharpe = sqrt(252) * mean / std(ddof=1), hand-computed:
    #   mean = 0.0266667, std = 0.0750555  ->  sqrt(252)*mean/std = 5.6405
    assert bt.sharpe == pytest.approx(5.6405, abs=1e-3)

    # equity curve: 1.10, 1.0450, 1.076350 ; peak 1.10 -> trough 1.0450 = -5%
    assert bt.cumulative_returns.iloc[-1] == pytest.approx(1.10 * 0.95 * 1.03)
    assert bt.max_drawdown == pytest.approx(-0.05)

    # annualized (geometric): total ** (252/3) - 1, positive since net gain
    total = 1.10 * 0.95 * 1.03
    assert bt.annual_return == pytest.approx(total ** (TRADING_DAYS / 3) - 1)
    assert bt.annual_return > 0


def test_end_to_end_sharpe_matches_isolated_helper():
    """The Sharpe the backtester reports is exactly _annualized_sharpe of the P&L series."""
    alpha, fwd = _random_panels(50, 20, seed=21)
    bt = backtest_alpha(alpha, fwd)
    assert bt.sharpe == pytest.approx(_annualized_sharpe(bt.portfolio_returns))
    assert bt.annual_return == pytest.approx(_annualized_return(bt.portfolio_returns))
    assert bt.max_drawdown == pytest.approx(_max_drawdown(bt.cumulative_returns))
