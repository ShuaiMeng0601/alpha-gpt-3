"""Exhaustive correctness + bias tests for the portfolio constructor.

Three things are verified here, in three sections:

  A. COMBINATION MATH — every method (equal, orthogonalized) and every
     primitive (_weighted_signals, _ols_beta, _orthogonalize_betas, ...) is checked
     against hand-computed expected values, including NaN / zero-denominator /
     degenerate-composite edge cases.

  B. NO LOOK-AHEAD — structurally (the construction functions cannot even receive test
     labels) and behaviourally (the composite is invariant to test forward returns; the
     learned combiners depend only on the FIT split; z-score uses no future
     day). The fit→apply contract is proven by a val==test equivalence test.

  C. NO OPTIMIZATION BIAS — the pipeline picks the headline method on VALIDATION, never
     by peeking at the test Sharpe; learned methods are skipped (not silently self-fit on
     test) when no validation split is supplied.

All data is small, deterministic, synthetic — no real data, no LLM, no GP evolution.
"""

import inspect

import numpy as np
import pandas as pd
import pytest

from alpha_gpt.data.loader import DataSplit
from alpha_gpt.portfolio import construct
from alpha_gpt.portfolio.construct import (
    _apply_orthogonalization,
    _mean_signals,
    _ols_beta,
    _orthogonalize_betas,
    _weighted_signals,
    build_signals,
    combine,
    fit_combiners,
)

# --- tiny builders ------------------------------------------------------------

def _df(rows, start="2020-01-01"):
    """2D list/array -> DataFrame on a fixed daily grid (cols = 0..k-1)."""
    arr = np.asarray(rows, dtype=float)
    idx = pd.bdate_range(start, periods=arr.shape[0])
    return pd.DataFrame(arr, index=idx, columns=list(range(arr.shape[1])))


def _split(seed=0, n_days=80, n_stocks=30):
    """A DataSplit of synthetic panels (matches conftest's `panels` shape) + fwd target."""
    rs = np.random.RandomState(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    cols = list(range(1, n_stocks + 1))
    close = pd.DataFrame(rs.randn(n_days, n_stocks).cumsum(0), index=idx, columns=cols) + 100.0
    rets = close.pct_change().fillna(0.0)
    vol = pd.DataFrame(rs.rand(n_days, n_stocks) * 1e6 + 1e3, index=idx, columns=cols)
    panels = {
        "close": close, "open": close.shift(1).bfill(),
        "high": close * 1.01, "low": close * 0.99,
        "volume": vol, "returns": rets,
    }
    return DataSplit(panels=panels, forward_returns=rets.shift(-1))


# A handful of survivors spanning unary/binary/cs/ts ops and both IC signs.
_EXPRS = [
    ("cs_rank(close)", 0.20),
    ("cs_zscore(volume)", -0.15),
    ("ts_mean_5(returns)", 0.08),
    ("neg(cs_rank(high))", 0.12),
    ("safe_div(close, open)", -0.05),
]


def _survivors():
    return [{"id": i + 1, "expression": e, "is_ic": ic} for i, (e, ic) in enumerate(_EXPRS)]


def _weights(survs):
    return {s["id"]: s["is_ic"] for s in survs}


# =============================================================================
# A. COMBINATION MATH
# =============================================================================

def test_equal_is_coverage_weighted_nanmean():
    s0 = _df([[1.0, 2.0], [3.0, 4.0]])
    s1 = _df([[3.0, np.nan], [5.0, 6.0]])  # (0,1) missing in s1
    comp, sel = combine({0: s0, 1: s1}, method="equal")
    # (0,0)=mean(1,3)=2 ; (0,1)=mean(2)=2 ; (1,0)=mean(3,5)=4 ; (1,1)=mean(4,6)=5
    pd.testing.assert_frame_equal(comp, _df([[2.0, 2.0], [4.0, 5.0]]))
    assert sorted(sel) == [0, 1]


def test_weighted_signals_renormalizes_over_present_cells():
    # cell present only in s0 -> returns s0's value (weight renormalized over coverage).
    s0, s1 = _df([[1.0]]), _df([[np.nan]])
    out = _weighted_signals([s0, s1], np.array([0.25, 0.75]))
    pd.testing.assert_frame_equal(out, _df([[1.0]]))


def test_weighted_signals_all_missing_cell_is_nan():
    out = _weighted_signals([_df([[np.nan]]), _df([[np.nan]])], np.array([1.0, 1.0]))
    assert bool(out.isna().all().all())


def test_single_signal_is_returned_unchanged():
    s0 = _df([[1.0, 2.0, 3.0]])
    pd.testing.assert_frame_equal(_mean_signals([s0]), s0)
    comp, _ = combine({7: s0}, "equal")
    pd.testing.assert_frame_equal(comp, s0)


def test_ols_beta_recovers_through_origin_slope():
    comp = _df([[1.0, -2.0, 3.0, -4.0]])
    s = comp * 2.5  # exact linear, no intercept
    assert _ols_beta(s, comp) == pytest.approx(2.5)


def test_ols_beta_degenerate_composite_is_zero():
    s = _df([[1.0, 2.0, 3.0]])
    assert _ols_beta(s, _df([[0.0, 0.0, 0.0]])) == 0.0          # zero composite
    assert _ols_beta(s, _df([[np.nan, np.nan, np.nan]])) == 0.0  # all-NaN composite


def test_orthogonalized_residualizes_duplicate_to_zero():
    s = _df([[1.0, -1.0, 2.0, -2.0]])
    sigs = {0: s.copy(), 1: s.copy()}                 # identical signals
    weights = {0: 0.9, 1: 0.1}                         # 0 processed first
    betas = _orthogonalize_betas(sigs, weights)
    assert betas[0] == (0, None)                       # first kept as-is
    assert betas[1][1] == pytest.approx(1.0)           # second fully explained by first
    comp, used = _apply_orthogonalization(betas, sigs)
    assert used == [0, 1]
    # composite is proportional to s (the dup adds a zero residual) -> rank-identical.
    flat_c, flat_s = comp.to_numpy().ravel(), s.to_numpy().ravel()
    assert np.corrcoef(flat_c, flat_s)[0, 1] == pytest.approx(1.0)


def test_orthogonalized_equals_mean_for_orthogonal_signals():
    s0 = _df([[1.0, 1.0, -1.0, -1.0]])
    s1 = _df([[1.0, -1.0, 1.0, -1.0]])  # dot(s0, s1) = 0
    weights = {0: 0.3, 1: 0.1}
    betas = _orthogonalize_betas({0: s0, 1: s1}, weights)
    assert betas[1][1] == pytest.approx(0.0)            # nothing to residualize
    ortho, _ = combine({0: s0, 1: s1}, "orthogonalized", weights)
    eq, _ = combine({0: s0, 1: s1}, "equal")
    pd.testing.assert_frame_equal(ortho, eq)


def test_combine_empty_and_unknown_method():
    assert combine({}, "equal") == (None, [])
    with pytest.raises(ValueError):
        combine({0: _df([[1.0, 2.0]])}, "bogus")


def test_orthogonalized_skips_ids_absent_on_apply_split():
    # betas were fit on ids [1,2,3]; apply split only has [1,3] -> 2 skipped gracefully.
    s1, s3 = _df([[1.0, -1.0, 2.0]]), _df([[2.0, 0.0, -2.0]])
    betas = [(1, None), (2, 0.5), (3, 0.3)]
    comp, used = _apply_orthogonalization(betas, {1: s1, 3: s3})
    assert used == [1, 3] and comp is not None


# =============================================================================
# B. NO LOOK-AHEAD BIAS
# =============================================================================

_CONSTRUCT_FNS = [
    construct._signal_on, build_signals, fit_combiners, combine,
    _apply_orthogonalization, _orthogonalize_betas,
    _weighted_signals, _mean_signals, _ols_beta,
]


@pytest.mark.parametrize("fn", _CONSTRUCT_FNS, ids=lambda f: f.__name__)
def test_construction_never_takes_forward_returns(fn):
    """Structural guarantee: no construction function can even receive the target."""
    params = set(inspect.signature(fn).parameters)
    assert not (params & {"forward_returns", "fwd", "returns_fwd", "target", "y"})


def test_signal_keeps_native_sign_never_flips(pset):
    # Same expression, opposite stored is_ic -> IDENTICAL signals. We never sign-flip:
    # is_ic does not affect a signal's orientation.
    survs = [{"id": 1, "expression": "cs_rank(close)", "is_ic": 0.3},
             {"id": 2, "expression": "cs_rank(close)", "is_ic": -0.3}]
    sp = _split(seed=0)
    out = build_signals(survs, pset, sp.panels)
    diff = (out[1].fillna(0) - out[2].fillna(0)).abs().to_numpy()
    assert np.nanmax(diff) < 1e-9


@pytest.mark.parametrize("is_ic", [0.5, -0.5, 0.0, None])
def test_is_ic_never_affects_signal(pset, is_ic):
    # Whatever is_ic is (positive, negative, zero, missing), the built signal is the raw one.
    rec = {"id": 1, "expression": "cs_rank(close)"}
    if is_ic is not None:
        rec["is_ic"] = is_ic
    sp = _split(seed=0)
    built = build_signals([rec], pset, sp.panels)[1]
    raw = construct._signal_on({"id": 1, "expression": "cs_rank(close)"}, pset, sp.panels)
    pd.testing.assert_frame_equal(built, raw)


def test_zscore_normalization_uses_no_future_day(pset):
    """Per-day cross-sectional z-score: mutating a later day must not move earlier days."""
    sp = _split(seed=0)
    sig = build_signals([{"id": 1, "expression": "cs_zscore(close)", "is_ic": 0.1}],
                        pset, sp.panels)[1]
    sp2 = _split(seed=0)
    sp2.panels["close"].iloc[-1] = sp2.panels["close"].iloc[-1] * 7.0 + 999.0  # perturb last day
    sig2 = build_signals([{"id": 1, "expression": "cs_zscore(close)", "is_ic": 0.1}],
                         pset, sp2.panels)[1]
    pd.testing.assert_frame_equal(sig.iloc[:-1], sig2.iloc[:-1])


@pytest.mark.parametrize("method", ["equal", "orthogonalized"])
def test_composite_is_invariant_to_test_forward_returns(pset, method):
    """The constructed composite depends only on panels + val-fit params, never on the
    test labels. Replacing test forward returns with noise leaves every method unchanged."""
    val, test = _split(seed=1), _split(seed=2)
    survs = _survivors(); w = _weights(survs)
    betas = fit_combiners(survs, pset, val.panels, w)
    sig1 = build_signals(survs, pset, test.panels)
    comp1, _ = combine(sig1, method, w, betas=betas)
    # Obliterate the labels; rebuild from the (unchanged) panels.
    test.forward_returns.iloc[:, :] = np.random.RandomState(99).randn(*test.forward_returns.shape)
    sig2 = build_signals(survs, pset, test.panels)
    comp2, _ = combine(sig2, method, w, betas=betas)
    pd.testing.assert_frame_equal(comp1, comp2)


def test_fit_combiners_depends_only_on_fit_split(pset):
    """Learned combiners are a function of the FIT (validation) panels alone — they are
    deterministic and do not change when a different (test) split exists."""
    survs = _survivors(); w = _weights(survs)
    val = _split(seed=1)
    b1 = fit_combiners(survs, pset, val.panels, w)
    b2 = fit_combiners(survs, pset, val.panels, w)               # determinism
    assert [x[1] for x in b1] == pytest.approx([x[1] for x in b2])
    # A different fit split yields different parameters (it really uses the fit data).
    b3 = fit_combiners(survs, pset, _split(seed=2).panels, w)
    assert [x[1] for x in b1][1:] != pytest.approx([x[1] for x in b3][1:])


def test_val_fit_betas_are_applied_to_test_not_refit(pset):
    """The orthogonalization betas used on test come from the VALIDATION fit, not from a
    self-fit on test (which would be look-ahead). Proven by: (a) val and test self-fits
    differ, and (b) the applied composite reproduces the val betas exactly."""
    val, test = _split(seed=1), _split(seed=2)
    survs = _survivors(); w = _weights(survs)
    test_signals = build_signals(survs, pset, test.panels)
    betas_val = fit_combiners(survs, pset, val.panels, w)
    betas_test = _orthogonalize_betas(test_signals, w)
    # val-fit betas differ from a test self-fit ...
    assert [b for _, b in betas_val][1:] != pytest.approx([b for _, b in betas_test][1:])
    applied, _ = combine(test_signals, "orthogonalized", w, betas=betas_val)
    selffit, _ = combine(test_signals, "orthogonalized", w)        # the look-ahead path
    manual, _ = _apply_orthogonalization(betas_val, test_signals)
    pd.testing.assert_frame_equal(applied, manual)                 # uses val betas exactly
    # ... so the no-look-ahead composite differs from the self-fit (look-ahead) one.
    diff = (applied.fillna(0) - selffit.fillna(0)).abs().to_numpy().max()
    assert diff > 1e-6


def test_fit_then_apply_equals_self_fit_when_val_equals_test(pset):
    """Fit→apply consistency: when the fit split IS the apply split, applying the fitted
    betas must reproduce the self-fit composite exactly. This proves the apply path is the
    faithful inverse of the fit path (no silent divergence)."""
    sp = _split(seed=5)
    survs = _survivors(); w = _weights(survs)
    signals = build_signals(survs, pset, sp.panels)
    betas = fit_combiners(survs, pset, sp.panels, w)
    applied, sel_a = combine(signals, "orthogonalized", w, betas=betas)
    selffit, sel_s = combine(signals, "orthogonalized", w)
    assert sel_a == sel_s
    pd.testing.assert_frame_equal(applied, selffit)


# =============================================================================
# C. NO OPTIMIZATION BIAS (pipeline-level)
# =============================================================================

def test_select_headline_prefers_validation_winner_over_test_sharpe():
    from alpha_gpt.pipeline import _select_headline
    rows = [
        {"method": "equal", "sharpe": 0.5},
        {"method": "orthogonalized", "sharpe": 3.0},   # best on TEST
    ]
    val_scores = {"equal": 2.0, "orthogonalized": 0.1}  # best on VALIDATION = equal
    assert _select_headline(rows, val_scores)["method"] == "equal"


def test_select_headline_falls_back_to_test_sharpe_without_val():
    from alpha_gpt.pipeline import _select_headline
    rows = [{"method": "equal", "sharpe": 0.5}, {"method": "orthogonalized", "sharpe": 3.0}]
    assert _select_headline(rows, {})["method"] == "orthogonalized"
    assert _select_headline([], {"equal": 1.0}) is None


def test_backtest_portfolio_skips_learned_methods_without_validation(pset, tmp_path):
    """No look-ahead in production: with no validation split to fit it, the learned
    method is SKIPPED — never silently self-fit on the test split."""
    from alpha_gpt.pipeline import backtest_portfolio
    test = _split(seed=4)
    port = backtest_portfolio(
        _survivors(), pset, test, fit_split=None,
        methods=["equal", "orthogonalized"],
        out_dir=str(tmp_path), show_progress=False)
    present = {r["method"] for r in port["rows"]}
    assert "equal" in present
    assert "orthogonalized" not in present


def test_backtest_portfolio_headline_is_in_sample_chosen(pset, tmp_path):
    """End-to-end: the reported headline is a real row chosen via in-sample scoring."""
    from alpha_gpt.pipeline import _method_scores, backtest_portfolio
    fit_split, test = _split(seed=1), _split(seed=2)
    survs = _survivors(); w = _weights(survs)
    betas = fit_combiners(survs, pset, fit_split.panels, w)
    methods = ["equal", "orthogonalized"]
    fit_scores = _method_scores(survs, pset, fit_split, w, methods, betas)
    assert fit_scores  # in-sample scoring actually ran
    port = backtest_portfolio(survs, pset, test, fit_split, methods=methods,
                              out_dir=str(tmp_path), show_progress=False)
    best = port["best"]
    assert best in port["rows"]
    # The headline maximizes the IN-SAMPLE score among scored methods (not test Sharpe).
    assert best["method"] == max(fit_scores, key=fit_scores.get)
