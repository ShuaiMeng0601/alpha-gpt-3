"""Load parquet panel data and split into train/val/test sets."""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DataSplit:
    """Container for panel data split into time periods."""
    panels: dict[str, pd.DataFrame]
    forward_returns: pd.DataFrame


def load_panels(data_dir: str = "data/panels") -> dict[str, pd.DataFrame]:
    """Load all parquet panel files from data_dir.

    Returns dict mapping field name to DataFrame (index=date, columns=PERMNO).
    """
    panels = {}
    for f in os.listdir(data_dir):
        if f.endswith(".parquet"):
            name = f.replace(".parquet", "")
            panels[name] = pd.read_parquet(os.path.join(data_dir, f))
    print(f"Loaded {len(panels)} panels: {list(panels.keys())}")
    return panels


def parse_yyyymmdd(series: pd.Series) -> pd.Series:
    """Parse a YYYYMMDD key whether WRDS wrote 20100104 or 2010-01-04 (-> NaT on junk)."""
    s = series.astype("string").str.replace("-", "", regex=False).str.slice(0, 8)
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce")


def ffill_to_membership(panel: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill a sparse (date x PERMNO) panel onto the point-in-time membership grid.

    Unions the panel's own dates with the membership calendar BEFORE the forward-fill,
    so observations dated on non-trading days (e.g. a month-end WRDS public_date that
    falls on a weekend) are carried forward instead of being silently dropped by a
    direct reindex to the trading calendar. Then restricts to the membership grid and
    masks to in-universe cells. Shared by every PIT panel builder (CRSP ratios,
    Compustat characteristics, OSAP signals) so they cannot drift apart.
    """
    return (
        panel.reindex(index=panel.index.union(membership.index))
        .sort_index()
        .ffill()
        .reindex(index=membership.index, columns=membership.columns)
        .where(membership)
    )


def split_data(
    panels: dict[str, pd.DataFrame],
    is_end: str = "2020-12-31",
) -> tuple[DataSplit, DataSplit]:
    """Split panels into one in-sample block and a held-out test block by date.

    in_sample = [start, is_end]; test = (is_end, end]. The in-sample block is used for
    every selection decision (IC / t-stat gate, orientation sign); test is touched once
    by the final backtest.
    """
    is_end_dt = pd.Timestamp(is_end)

    def _slice(p: dict[str, pd.DataFrame], start, end) -> DataSplit:
        sliced = {}
        for name, df in p.items():
            if name == "forward_returns":
                continue
            if start is None:
                mask = (df.index <= end)
            elif end is None:
                mask = (df.index >= start)
            else:
                mask = (df.index >= start) & (df.index <= end)
            sliced[name] = df.loc[mask]

        fwd = p.get("forward_returns", pd.DataFrame())
        if not fwd.empty:
            if start is None:
                mask = (fwd.index <= end)
            elif end is None:
                mask = (fwd.index >= start)
            else:
                mask = (fwd.index >= start) & (fwd.index <= end)
            fwd = fwd.loc[mask]

        return DataSplit(panels=sliced, forward_returns=fwd)

    in_sample = _slice(panels, None, is_end_dt)
    test = _slice(panels, is_end_dt + pd.Timedelta(days=1), None)  # all remaining dates

    # Embargo the 1-day boundary leak: the in-sample block's last forward_return is the
    # realized return of test's first day (forward = returns.shift(-1)). Blank it so the
    # in-sample labels don't peek across the boundary. (Test has no upper bound, so its
    # last forward_return is already NaN.)
    if not in_sample.forward_returns.empty:
        in_sample.forward_returns = in_sample.forward_returns.copy()
        in_sample.forward_returns.iloc[-1] = np.nan

    print(f"In-sample: {_date_range(in_sample)}, Test: {_date_range(test)}")
    return in_sample, test


def _date_range(split: DataSplit) -> str:
    """Helper to format date range string."""
    sample = next(iter(split.panels.values()), pd.DataFrame())
    if sample.empty:
        return "empty"
    return f"{sample.index.min().date()} to {sample.index.max().date()}"
