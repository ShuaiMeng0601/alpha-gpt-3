"""Tests for alpha_gpt.data.fundamentals (Compustat -> cs_* characteristics)."""

import pandas as pd

from alpha_gpt.data.fundamentals import compute_characteristics, load_and_map

_BASE = dict(atq=100.0, ltq=40.0, ceqq=50.0, saleq=80.0, revtq=80.0, cogsq=40.0,
             niq=5.0, oiadpq=5.0, dlttq=20.0, actq=30.0, lctq=15.0, cheq=10.0,
             cshoq=10.0, prccq=20.0, xrdq=1.0, gsector=40)


def _five_quarters(ibq):
    n = len(ibq)
    return pd.DataFrame({
        "permno": [1] * n,
        "datadate": pd.date_range("2015-03-31", periods=n, freq="QE"),
        "eff_date": pd.date_range("2015-05-15", periods=n, freq="QE"),
        "ibq": ibq, "niq": ibq,
        **{k: [v] * n for k, v in _BASE.items() if k not in ("niq",)},
    })


def test_earnings_growth_sign_regression():
    """REGRESSION: loss->profit must read as POSITIVE growth.

    Old code did ibq / |ibq_lag4| - 1 (one-sided abs), mis-signing turnarounds.
    Fixed form is (ibq - ibq_lag4) / |ibq_lag4|.
    """
    m = _five_quarters([-10.0, 1.0, 2.0, 3.0, 5.0])  # lag4 of last = -10 (loss) -> +5 (profit)
    eg = compute_characteristics(m)["cs_earnings_growth"].iloc[-1]
    assert abs(eg - 1.5) < 1e-9      # (5 - (-10)) / 10 = +1.5

    m2 = _five_quarters([-10.0, 1.0, 2.0, 3.0, -20.0])  # loss -> deeper loss
    eg2 = compute_characteristics(m2)["cs_earnings_growth"].iloc[-1]
    assert abs(eg2 - (-1.0)) < 1e-9  # (-20 - (-10)) / 10 = -1.0


def test_basic_characteristics_values():
    m = _five_quarters([5.0] * 5)
    out = compute_characteristics(m).iloc[-1]
    assert abs(out["cs_roa"] - (5.0 / 100.0)) < 1e-9      # niq/atq
    assert abs(out["cs_leverage"] - (40.0 / 100.0)) < 1e-9  # ltq/atq


def test_load_and_map_prefers_in_span_permno_regression(tmp_path):
    """REGRESSION: a quarter must attach to a permno whose validity span CONTAINS its
    effective date, not merely the closest-by-midpoint one inside the +/-tol band."""
    # gvkey 1 -> permno 100 (2015-2019) and 200 (2013-2014)
    link = pd.DataFrame({
        "gvkey": [1, 1], "permno": [100, 200],
        "start": pd.to_datetime(["2015-01-01", "2013-01-01"]),
        "end": pd.to_datetime(["2019-12-31", "2014-12-31"]),
    })
    # datadate 2014-11-15 -> eff = +90d ~ 2015-02-13: inside 100's span, but CLOSER to
    # 200's midpoint (so the old midpoint rule would wrongly pick 200).
    fund = pd.DataFrame({
        "gvkey": [1], "datadate": ["20141115"], "rdq": [""],
        **{k: [v] for k, v in _BASE.items()},
    })
    path = tmp_path / "fund.csv"
    fund.to_csv(path, index=False)
    m = load_and_map(str(path), link, universe=set())
    assert set(m["permno"]) == {100}
