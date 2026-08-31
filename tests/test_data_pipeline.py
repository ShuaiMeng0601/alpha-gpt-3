"""Regression tests for scripts/prepare_data.py row handling.

The headline bug of this codebase: `_drop_header_rows` deleted every row with
SecurityHdrFlg=='Y', but those are a security's CURRENT-segment daily observations
(real prices) — dropping them collapsed recent-year coverage to ~0. This guards it.
"""

import importlib.util
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_prepare_data():
    spec = importlib.util.spec_from_file_location(
        "prepare_data_under_test", ROOT / "scripts" / "prepare_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_drop_header_rows_keeps_current_segment_rows():
    pdm = _load_prepare_data()
    ch = pd.DataFrame({
        "PERMNO": [1, 1, 1],
        "SecurityHdrFlg": ["N", "Y", "Y"],   # flips to current-segment mid-history
        "DlyPrc": [10.0, 11.0, 12.0],         # all real daily prices
        "YYYYMMDD": ["20230901", "20231001", "20231002"],
    })
    out = pdm._drop_header_rows(ch)
    # REGRESSION: must keep ALL three rows — the 'Y' rows are not metadata.
    assert len(out) == 3
    assert sorted(out["DlyPrc"].tolist()) == [10.0, 11.0, 12.0]


def test_date_max_covers_recent_years():
    """DATE_MAX must not silently truncate the build at 2023 (the old stale default)."""
    pdm = _load_prepare_data()
    assert pdm.DATE_MAX >= 20251231
