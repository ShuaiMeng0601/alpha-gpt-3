"""Contract tests for the operator registry.

Every operator is called on thousands of expressions per run, so a silently wrong or
silently all-NaN operator poisons a whole arm of the study. These tests pin the two things
that matter: every registered operator survives realistic (ragged, gappy) panel input and
returns something with signal, and the operators whose semantics are easy to get subtly
wrong compute what their name claims.
"""

import numpy as np
import pandas as pd
import pytest

from alpha_gpt.expr.alpha_ops import ALL_OPS, BINARY_OPS, UNARY_OPS


def _panel(seed=0, n=120, m=40):
    """A panel shaped like the real thing: drifting levels, ragged listing history, gaps.

    Deliberately straddles zero. Operators are applied to intermediate results (deltas,
    z-scores, spreads) far more often than to raw prices, so an all-positive fixture would
    make `sign` look constant and hide anything that only misbehaves on negatives.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    values = rng.normal(size=(n, m)).cumsum(0) + rng.normal(0.0, 10.0, size=(1, m))
    df = pd.DataFrame(values, index=idx, columns=[f"s{i}" for i in range(m)])
    df.iloc[:5, :6] = np.nan        # names that list late
    df.iloc[40:50, 10:16] = np.nan  # mid-panel holes
    return df


@pytest.mark.parametrize("name", sorted(ALL_OPS))
def test_operator_returns_a_usable_panel(name):
    op = ALL_OPS[name]
    x, y = _panel(0), _panel(1)
    out = op(x, y) if op in BINARY_OPS else op(x)
    assert isinstance(out, pd.DataFrame), f"{name} did not return a DataFrame"
    assert out.shape == x.shape, f"{name} changed panel shape: {out.shape} != {x.shape}"
    vals = out.to_numpy(dtype=float)
    finite = np.isfinite(vals)
    assert finite.any(), f"{name} produced an entirely NaN/inf panel"
    assert vals[finite].std() > 0, f"{name} produced a constant panel (carries no signal)"


def test_registry_names_are_unique():
    """ALL_OPS is keyed by __name__; a duplicate would silently shadow an operator."""
    assert len(ALL_OPS) == len(UNARY_OPS) + len(BINARY_OPS)


def _col(series_values, op_name, *extra):
    frames = [pd.DataFrame({"a": v}) for v in (series_values,) + extra]
    return ALL_OPS[op_name](*frames)["a"].tolist()


def test_shift_and_sum():
    assert _col([1., 2., 5., 3., 4.], "ts_shift_1")[1:] == [1., 2., 5., 3.]
    assert _col([1., 2., 5., 3., 4.], "ts_sum_5") == [1., 3., 8., 11., 15.]


def test_elementwise_shapes_of_the_new_primitives():
    assert _col([-2., 0., 3.], "relu") == [0., 0., 3.]
    assert _col([-4., 9.], "signed_sqrt") == [-2., 3.]
    assert _col([-3., 4.], "square") == [9., 16.]
    assert _col([1., 5.], "cwise_max", [3., 3.]) == [3., 5.]
    assert _col([1., 5.], "cwise_min", [3., 3.]) == [1., 3.]


def test_comparisons_are_a_zero_one_mask():
    """greater/less are the only way to express a conditional, so the encoding matters."""
    assert _col([1., 5.], "greater", [3., 3.]) == [0., 1.]
    assert _col([1., 5.], "less", [3., 3.]) == [1., 0.]


def test_comparison_preserves_nan_rather_than_calling_it_false():
    """A missing value must stay missing — treating it as False would silently place bets
    on names with no data."""
    out = ALL_OPS["greater"](pd.DataFrame({"a": [np.nan, 5.0]}), pd.DataFrame({"a": [3.0, 3.0]}))
    assert np.isnan(out["a"].iloc[0]) and out["a"].iloc[1] == 1.0


def test_linear_reg_recovers_a_known_slope():
    ramp = pd.DataFrame({"a": np.arange(30.0)})
    assert ALL_OPS["ts_linear_reg_20"](ramp)["a"].iloc[-1] == pytest.approx(1.0)
    assert ALL_OPS["ts_linear_reg_20"](-ramp)["a"].iloc[-1] == pytest.approx(-1.0)


def test_argmax_reports_position_within_the_window():
    """0 = oldest observation in the window. A flipped convention would invert every
    'time since the high' signal built on it."""
    # exactly one 20-row window; padding sits strictly between the min and the max so the
    # extremes stay at the two positions being asserted
    s = pd.DataFrame({"a": [5., 1., 2., 9., 3.] + [4.] * 15})
    assert ALL_OPS["ts_argmax_20"](s)["a"].iloc[-1] == 3.0
    assert ALL_OPS["ts_argmin_20"](s)["a"].iloc[-1] == 1.0


def test_argmax_is_undefined_before_a_full_window_exists():
    """A 'position within the last 20 days' has no meaning on day 5, so it stays NaN
    rather than reporting a position inside a partial window."""
    short = pd.DataFrame({"a": [1., 2., 9., 3., 4.]})
    assert ALL_OPS["ts_argmax_20"](short)["a"].isna().all()


def test_maxmin_scale_is_bounded():
    out = ALL_OPS["ts_maxmin_scale_20"](_panel()).to_numpy(dtype=float)
    finite = out[np.isfinite(out)]
    assert finite.min() >= 0.0 and finite.max() <= 1.0


def test_winsorize_clips_the_cross_section_without_reordering_it():
    x = _panel()
    out = ALL_OPS["cs_winsorize"](x)
    assert out.max(axis=1).le(x.max(axis=1) + 1e-9).all()
    # clipping must not change relative order — it is a tail treatment, not a transform
    row = x.iloc[60].rank()
    assert (out.iloc[60].rank().values == row.values).all()


def test_argmax_matches_a_naive_reference_on_a_gappy_panel():
    """The chunked sliding-window implementation is the one place a subtle indexing or
    NaN-masking bug could hide, so check it against the obvious slow version.

    ``min_periods=1`` matches the convention every other rolling operator here uses: a
    window reports as soon as it holds one real value, and is NaN only if it holds none.
    """
    x = _panel(seed=3, n=60, m=8)
    got = ALL_OPS["ts_argmax_20"](x)
    ref = x.rolling(20, min_periods=1).apply(
        lambda w: np.nan if np.isnan(w).all() else np.nanargmax(w), raw=True)
    pd.testing.assert_frame_equal(got.iloc[19:], ref.iloc[19:], check_dtype=False)
