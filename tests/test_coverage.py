"""Tests for alpha_gpt.data.coverage (panel coverage report + terminal qualification)."""

import numpy as np
import pandas as pd

from alpha_gpt.data.coverage import panel_coverage, qualifies


def test_panel_coverage_uses_membership_denominator():
    idx = pd.bdate_range("2020-01-01", periods=4)
    membership = pd.DataFrame(True, index=idx, columns=[1, 2])
    panel = pd.DataFrame({1: [1.0, 1.0, np.nan, np.nan], 2: [1.0, 1.0, 1.0, 1.0]}, index=idx)
    rec = panel_coverage(panel, membership)
    assert abs(rec["coverage"] - 0.75) < 1e-9   # 6 non-NaN / 8 in-universe cells
    assert rec["median_breadth"] == 1.5
    assert rec["n_cols"] == 2


def test_qualifies_thresholds():
    assert qualifies({"coverage": 0.5, "median_breadth": 150}, 0.2, 100) is True
    assert qualifies({"coverage": 0.1, "median_breadth": 150}, 0.2, 100) is False   # too sparse
    assert qualifies({"coverage": 0.5, "median_breadth": 50}, 0.2, 100) is False    # too thin
