"""Alpha quality metrics: IC, ICIR, turnover."""

import numpy as np
import pandas as pd


def compute_ic(alpha_values: pd.DataFrame, forward_returns: pd.DataFrame,
               min_stocks: int = 20) -> pd.Series:
    """Daily cross-sectional Spearman rank IC between alpha values and forward returns.

    Spearman's rho *is* the Pearson correlation of the ranks, so this computes it directly
    and vectorized over the whole panel (rank each day, demean, correlate) with no per-date
    Python loop. It is the exact per-day Spearman (matching ``scipy.stats.spearmanr`` with
    average-rank ties), ~50x faster than the looped form. Ranks use only cells valid in
    BOTH panels, so one-sided NaNs cannot bias the correlation. Days with fewer than
    ``min_stocks`` jointly-valid names, or with no rank variance, are dropped.

    Returns a Series indexed by the surviving dates.
    """
    a, f = alpha_values.align(forward_returns, join="inner")   # common date x stock grid
    valid = a.notna() & f.notna()
    n_valid = valid.sum(axis=1)
    ar = a.where(valid).rank(axis=1)
    fr = f.where(valid).rank(axis=1)
    ar = ar.sub(ar.mean(axis=1), axis=0)
    fr = fr.sub(fr.mean(axis=1), axis=0)
    num = (ar * fr).sum(axis=1)
    den = np.sqrt((ar * ar).sum(axis=1) * (fr * fr).sum(axis=1))
    ic = (num / den.replace(0, np.nan))[n_valid >= min_stocks]
    return ic.dropna().rename("IC")


def compute_icir(ic_series: pd.Series) -> float:
    """Information ratio: IC mean / IC std.

    Uses a tolerance rather than `std == 0`: pandas' variance leaves a ~1e-17 residual
    for an all-identical series (catastrophic cancellation), which would otherwise make
    this return ~1e16 instead of 0. No real daily-IC std is anywhere near 1e-12.
    """
    if ic_series.empty:
        return 0.0
    std = float(ic_series.std())
    if not np.isfinite(std) or std < 1e-12:
        return 0.0
    return float(ic_series.mean() / std)


def compute_turnover(alpha_values: pd.DataFrame) -> float:
    """Average daily rank turnover of the alpha signal.

    Measures how much the cross-sectional ranking changes day to day.
    """
    ranks = alpha_values.rank(axis=1, pct=True)
    daily_changes = ranks.diff().abs().mean(axis=1)
    return float(daily_changes.mean())
