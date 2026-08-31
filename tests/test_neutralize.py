"""Tests for alpha_gpt.evaluate.neutralize (cross-sectional & sector neutralization)."""

import numpy as np
import pandas as pd

from alpha_gpt.evaluate.neutralize import (
    characteristic_neutralize,
    sector_neutralize,
    zscore_neutralize,
)


def test_zscore_neutralize_demeans_and_unit_scales():
    idx = pd.bdate_range("2020-01-01", periods=5)
    cols = list(range(10))
    a = pd.DataFrame(np.random.RandomState(0).randn(5, 10) * 3 + 7, index=idx, columns=cols)
    z = zscore_neutralize(a)
    assert np.allclose(z.mean(axis=1).to_numpy(), 0.0, atol=1e-9)
    assert np.allclose(z.std(axis=1, ddof=1).to_numpy(), 1.0, atol=1e-9)


def test_zscore_neutralize_constant_row_is_not_inf():
    idx = pd.bdate_range("2020-01-01", periods=2)
    a = pd.DataFrame({0: [5.0, 1.0], 1: [5.0, 2.0], 2: [5.0, 3.0]}, index=idx)
    z = zscore_neutralize(a)
    assert z.loc[idx[0]].isna().all()  # zero-variance row -> NaN, not inf


def test_sector_neutralize_zeroes_within_sector_means():
    idx = pd.to_datetime(["2020-01-01"])
    cols = [0, 1, 2, 3]
    alpha = pd.DataFrame([[1.0, 3.0, 10.0, 20.0]], index=idx, columns=cols)
    sector = pd.DataFrame([[1, 1, 2, 2]], index=idx, columns=cols)  # two sectors
    resid = sector_neutralize(alpha, sector)
    # within each (day, sector) the residual mean is ~0
    assert abs(resid.iloc[0, 0] + resid.iloc[0, 1]) < 1e-9   # sector 1: [-1, +1]
    assert abs(resid.iloc[0, 2] + resid.iloc[0, 3]) < 1e-9   # sector 2: [-5, +5]
    assert abs(resid.iloc[0, 0] - (-1.0)) < 1e-9


def _char_fixture(n_days=6, n_stocks=40, seed=1):
    idx = pd.bdate_range("2021-01-01", periods=n_days)
    cols = list(range(n_stocks))
    rs = np.random.RandomState(seed)
    char = pd.DataFrame(rs.rand(n_days, n_stocks) * 100, index=idx, columns=cols)
    noise = pd.DataFrame(rs.randn(n_days, n_stocks), index=idx, columns=cols)
    return idx, cols, char, noise


def test_characteristic_neutralize_residual_orthogonal_to_char():
    idx, cols, char, noise = _char_fixture()
    alpha = char.rank(axis=1) * 0.5 + noise  # signal = char tilt + independent part
    resid = characteristic_neutralize(alpha, {"c": char})
    for d in idx:  # per-day residual is uncorrelated with the char's ranks
        r = np.corrcoef(resid.loc[d], char.loc[d].rank())[0, 1]
        assert abs(r) < 1e-8


def test_characteristic_neutralize_kills_pure_factor_signal():
    idx, cols, char, _ = _char_fixture()
    alpha = char.rank(axis=1) * 3.0 + 7.0  # exactly spanned by the char's ranks
    resid = characteristic_neutralize(alpha, {"c": char})
    assert float(resid.abs().max().max()) < 1e-8


def test_characteristic_neutralize_preserves_independent_signal():
    idx, cols, char, noise = _char_fixture()
    resid = characteristic_neutralize(noise, {"c": char})
    for d in idx:  # a char-independent signal survives mostly intact
        r = np.corrcoef(resid.loc[d], noise.loc[d])[0, 1]
        assert r > 0.9


def test_characteristic_neutralize_min_names_returns_nan_days():
    idx, cols, char, noise = _char_fixture(n_stocks=10)
    resid = characteristic_neutralize(noise, {"c": char}, min_names=20)
    assert resid.isna().all().all()  # 10 names < min 20 -> no-position days
