"""Tests for alpha_gpt.data.universe — point-in-time, survivorship-free universe.

This is the module whose sibling bug (the CRSP SecurityHdrFlg drop) collapsed recent
coverage, so it gets thorough coverage: eligibility, trailing-year selection, the
year mask, and delisting/survivorship in finalize_membership.
"""

import numpy as np
import pandas as pd

from alpha_gpt.data.universe import (
    accumulate_year_aggregates,
    build_year_mask,
    chunk_year_aggregates,
    finalize_membership,
    row_eligible,
    select_universe,
)

ELIG = dict(ShareType="NS", SecurityType="EQTY", IssuerType="CORP", PrimaryExch="N", USIncFlg="Y")


# --- row_eligible ---

def test_row_eligible_accepts_common_stock():
    assert bool(row_eligible(pd.DataFrame([ELIG]), include_reits=True).iloc[0])


def test_row_eligible_reit_toggle():
    reit = pd.DataFrame([{**ELIG, "IssuerType": "REIT"}])
    assert not bool(row_eligible(reit, include_reits=False).iloc[0])
    assert bool(row_eligible(reit, include_reits=True).iloc[0])


def test_row_eligible_rejects_each_bad_field():
    for field, bad in [("ShareType", "AD"), ("SecurityType", "FUND"),
                       ("IssuerType", "ACOR"), ("PrimaryExch", "X"), ("USIncFlg", "N")]:
        df = pd.DataFrame([{**ELIG, field: bad}])
        assert not bool(row_eligible(df, include_reits=True).iloc[0]), field


# --- chunk / accumulate aggregates ---

def test_chunk_year_aggregates():
    df = pd.DataFrame({
        "PERMNO": [10, 10, 20],
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01"]),
        "DlyCap": [100.0, 200.0, 50.0],
        "ShareType": ["NS", "NS", "AD"],  # permno 20 ineligible
        "SecurityType": ["EQTY"] * 3, "IssuerType": ["CORP"] * 3,
        "PrimaryExch": ["N"] * 3, "USIncFlg": ["Y"] * 3,
    })
    agg = chunk_year_aggregates(df, include_reits=True)
    assert agg.loc[(10, 2020), "cap_sum"] == 300.0
    assert agg.loc[(10, 2020), "cap_cnt"] == 2
    assert bool(agg.loc[(10, 2020), "elig_any"]) is True
    assert bool(agg.loc[(20, 2020), "elig_any"]) is False


def test_accumulate_year_aggregates_combines_chunks():
    df = pd.DataFrame({
        "PERMNO": [10], "date": pd.to_datetime(["2020-01-01"]), "DlyCap": [100.0],
        **{k: [v] for k, v in ELIG.items()},
    })
    a = chunk_year_aggregates(df, True)
    b = chunk_year_aggregates(df.assign(DlyCap=[50.0]), True)
    combined = accumulate_year_aggregates([a, b])
    assert combined.loc[(10, 2020), "cap_sum"] == 150.0
    assert combined.loc[(10, 2020), "cap_cnt"] == 2


# --- select_universe ---

def _year_agg(rows):
    idx = pd.MultiIndex.from_tuples(rows.keys(), names=["PERMNO", "year"])
    return pd.DataFrame(
        [dict(cap_sum=v[0], cap_cnt=v[1], elig_any=v[2]) for v in rows.values()], index=idx)


def test_select_universe_topN_and_eligibility():
    ya = _year_agg({
        (10, 2020): (100, 1, True), (20, 2020): (200, 1, True),
        (30, 2020): (50, 1, True), (40, 2020): (300, 1, False),  # biggest but ineligible
    })
    m = select_universe(ya, n=2)
    assert m[2020] == [20, 10]  # top-2 eligible by mean cap; 40 excluded


def test_select_universe_uses_trailing_year():
    """Selection for year Y uses Y-1's cap AND eligibility (no look-ahead)."""
    ya = _year_agg({
        (10, 2020): (100, 1, True), (20, 2020): (200, 1, True), (40, 2020): (300, 1, False),
        (10, 2021): (100, 1, True), (20, 2021): (200, 1, True), (40, 2021): (300, 1, True),
    })
    m = select_universe(ya, n=2)
    # 2021 ranks on 2020 data, where 40 is ineligible -> still [20, 10]
    assert m[2021] == [20, 10]


# --- build_year_mask ---

def test_build_year_mask_membership_by_calendar_year():
    idx = pd.bdate_range("2020-12-28", "2021-01-06")
    mask = build_year_mask({2020: [10, 20], 2021: [20, 30]}, idx)
    d20 = idx[idx.year == 2020][0]
    d21 = idx[idx.year == 2021][0]
    assert mask.loc[d20, 10] and mask.loc[d20, 20] and not mask.loc[d20, 30]
    assert mask.loc[d21, 20] and mask.loc[d21, 30] and not mask.loc[d21, 10]
    assert sorted(mask.columns) == [10, 20, 30]


# --- finalize_membership ---

def test_finalize_membership_drops_at_delisting():
    idx = pd.bdate_range("2021-01-01", periods=4)
    year_mask = pd.DataFrame(True, index=idx, columns=[10, 20])
    close = pd.DataFrame({10: [1.0, 2.0, np.nan, np.nan], 20: [1.0, 2.0, 3.0, 4.0]}, index=idx)
    mem = finalize_membership(year_mask, close)
    assert mem[10].tolist() == [True, True, False, False]  # gone once it stops trading
    assert mem[20].all()


def test_finalize_membership_excludes_non_member_years():
    idx = pd.bdate_range("2021-01-01", periods=3)
    year_mask = pd.DataFrame({10: [True, True, True], 20: [False, False, False]}, index=idx)
    close = pd.DataFrame({10: [1.0, 2.0, 3.0], 20: [9.0, 9.0, 9.0]}, index=idx)
    mem = finalize_membership(year_mask, close)
    assert mem[10].all()
    assert not mem[20].any()  # trades, but not selected this year
