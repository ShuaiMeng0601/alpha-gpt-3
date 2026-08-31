"""Vectorized selection metrics for the factory (no per-date Python loops).

IC/ICIR come from the canonical :func:`alpha_gpt.evaluate.metrics.compute_ic` (the exact
vectorized per-day Spearman). The Sharpe here is a fast PROXY: a rank-weighted,
dollar-neutral "factor return" that stands in for the quintile long-short backtest (highly
correlated with it, much cheaper). The FINAL portfolio result still uses the exact quintile
backtester on test.
"""

import numpy as np
import pandas as pd

from alpha_gpt.evaluate.backtester import RET_CLIP
from alpha_gpt.evaluate.metrics import compute_ic, compute_icir


def factor_returns(panel: pd.DataFrame, fwd: pd.DataFrame, ret_clip=RET_CLIP) -> pd.Series:
    """Daily rank-weighted, dollar-neutral long-short return (quintile proxy)."""
    valid = panel.notna() & fwd.notna()
    f = fwd.where(valid)
    if ret_clip is not None:
        f = f.clip(*ret_clip)  # same outlier handling as the quintile backtester
    r = panel.where(valid).rank(axis=1)
    w = r.sub(r.mean(axis=1), axis=0)
    gross = w.abs().sum(axis=1).replace(0, np.nan)
    w = w.div(gross, axis=0)
    return (w * f).sum(axis=1, min_count=1)


def summarize_full(panel: pd.DataFrame, fwd: pd.DataFrame, ic_days_scale: int = 1):
    """Return (ic, icir, tstat, sharpe, annual_return, max_drawdown).

    IC/ICIR are the exact per-day Spearman (``compute_ic``). ``tstat`` is the t-statistic of
    the mean daily IC = ICIR x sqrt(#IC days) — how significant the signal is, and the
    selection gate in the 2-split design. ``ic_days_scale`` recovers the FULL-sample t-stat
    when IC is measured on a date-subsample (every k-th day): pass k so the day count reflects
    the true in-sample length, else the t-stat is deflated by ~sqrt(k). (ICIR is scale-free,
    so only the sqrt(#days) term needs the correction.) Sharpe / return / drawdown come from
    the fast factor-return proxy above, not the exact quintile backtester.
    """
    ics = compute_ic(panel, fwd)
    n = len(ics)
    ic = float(ics.mean()) if n else 0.0
    icir = compute_icir(ics)
    tstat = float(icir * np.sqrt(n * max(1, ic_days_scale))) if n else 0.0

    fr = factor_returns(panel, fwd).dropna()
    if len(fr) > 1 and fr.std() > 0:
        sharpe = float(np.sqrt(252) * fr.mean() / fr.std())
        cum = (1 + fr).cumprod()
        # Guard annualization against a non-positive terminal wealth (a day with
        # factor return <= -100%): a fractional power of a negative base is NaN.
        last = float(cum.iloc[-1])
        ann = float(last ** (252 / len(fr)) - 1) if last > 0 else -1.0
        dd = float(((cum - cum.cummax()) / cum.cummax()).min())
    else:
        sharpe = ann = dd = 0.0
    return ic, icir, tstat, sharpe, ann, dd


def summarize(panel: pd.DataFrame, fwd: pd.DataFrame) -> tuple[float, float, float, float, float]:
    """(ic, icir, sharpe, annual_return, max_drawdown) — the t-stat-free tuple kept for the
    quick per-alpha metric line in the search loop and its tests."""
    ic, icir, _tstat, sharpe, ann, dd = summarize_full(panel, fwd)
    return ic, icir, sharpe, ann, dd
