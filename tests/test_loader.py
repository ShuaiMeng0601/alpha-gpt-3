"""Tests for alpha_gpt.data.loader: parse_yyyymmdd, ffill_to_membership, split_data.

Includes regressions for:
  * ratio panels dropping off-calendar observations before ffill (ffill_to_membership)
  * train/val labels leaking one day across split boundaries (split_data embargo)
"""

import numpy as np
import pandas as pd

from alpha_gpt.data.loader import DataSplit, ffill_to_membership, parse_yyyymmdd, split_data


# --- parse_yyyymmdd ---

def test_parse_yyyymmdd_both_formats():
    s = pd.Series(["20200103", "2020-01-03", "2020-01-03T00:00:00"])
    out = parse_yyyymmdd(s)
    assert (out == pd.Timestamp("2020-01-03")).all()


def test_parse_yyyymmdd_junk_to_nat():
    out = parse_yyyymmdd(pd.Series(["garbage", "", "2021-06-15"]))
    assert out.isna().tolist() == [True, True, False]


# --- ffill_to_membership ---

def test_ffill_carries_offcalendar_dates_regression():
    """REGRESSION: a source observation dated on a non-trading day (e.g. a month-end
    Saturday public_date) must be carried forward, not dropped by reindexing to the
    trading calendar before the ffill."""
    mem_idx = pd.to_datetime(["2018-06-29", "2018-07-02", "2018-07-03"])  # Fri, Mon, Tue
    membership = pd.DataFrame(True, index=mem_idx, columns=[1, 2])
    panel = pd.DataFrame({1: [0.5], 2: [0.7]}, index=pd.to_datetime(["2018-06-30"]))  # Saturday
    out = ffill_to_membership(panel, membership)
    assert out.loc["2018-07-02", 1] == 0.5
    assert out.loc["2018-07-03", 2] == 0.7


def test_ffill_does_not_backfill_before_first_obs():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    membership = pd.DataFrame(True, index=idx, columns=[1])
    panel = pd.DataFrame({1: [9.0]}, index=pd.to_datetime(["2020-01-02"]))
    out = ffill_to_membership(panel, membership)
    assert pd.isna(out.loc["2020-01-01", 1])  # nothing to fill from yet
    assert out.loc["2020-01-02", 1] == 9.0
    assert out.loc["2020-01-03", 1] == 9.0


def test_ffill_masks_non_member_cells():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02"])
    membership = pd.DataFrame({1: [True, False], 2: [True, True]}, index=idx)
    panel = pd.DataFrame({1: [1.0, 1.0], 2: [2.0, 2.0]}, index=idx)
    out = ffill_to_membership(panel, membership)
    assert pd.isna(out.loc["2020-01-02", 1])   # masked out (not a member that day)
    assert out.loc["2020-01-02", 2] == 2.0


def test_ffill_restricts_to_membership_grid():
    membership = pd.DataFrame(True, index=pd.to_datetime(["2020-01-02"]), columns=[1])
    panel = pd.DataFrame({1: [1.0, 2.0]}, index=pd.to_datetime(["2020-01-01", "2020-01-03"]))
    out = ffill_to_membership(panel, membership)
    assert list(out.index) == [pd.Timestamp("2020-01-02")]  # only the membership day survives
    assert list(out.columns) == [1]


# --- split_data ---

def _daily_panels(start="2016-01-01", end="2021-12-31"):
    idx = pd.bdate_range(start, end)
    cols = list(range(5))
    rs = np.random.RandomState(0)
    rets = pd.DataFrame(rs.randn(len(idx), 5) * 0.01, index=idx, columns=cols)
    return {"close": 100 + rets.cumsum(), "returns": rets, "forward_returns": rets.shift(-1)}


def test_split_data_boundaries():
    p = _daily_panels()
    in_sample, test = split_data(p, is_end="2020-12-31")
    assert in_sample.panels["close"].index.max() <= pd.Timestamp("2020-12-31")
    assert test.panels["close"].index.min() > pd.Timestamp("2020-12-31")
    # forward_returns is held separately, not in the terminal panels dict
    assert "forward_returns" not in in_sample.panels


def test_split_data_embargoes_boundary_label_regression():
    """REGRESSION: the in-sample block's last forward_return is test's first-day return;
    it must be blanked so the in-sample labels don't peek across the boundary."""
    p = _daily_panels()
    in_sample, test = split_data(p, is_end="2020-12-31")
    assert in_sample.forward_returns.iloc[-1].isna().all()


def test_split_data_returns_two_datasplits():
    p = _daily_panels()
    out = split_data(p, is_end="2020-12-31")
    assert len(out) == 2
    assert all(isinstance(s, DataSplit) for s in out)
