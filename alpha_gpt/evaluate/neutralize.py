"""Cross-sectional neutralization helpers.

``zscore_neutralize`` is used by the portfolio constructor; ``characteristic_neutralize``
backs the pipeline's factor-neutral diagnostic row (how much of a book's Sharpe survives
once generic characteristic tilts are regressed out); ``sector_neutralize`` is available
for sector-relative signals (see data/aux/sector.parquet built by prepare_data.py).
"""

import numpy as np
import pandas as pd


def zscore_neutralize(alpha: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional demean + unit-variance per day."""
    mean = alpha.mean(axis=1)
    std = alpha.std(axis=1).replace(0, np.nan)
    return alpha.sub(mean, axis=0).div(std, axis=0)


def characteristic_neutralize(alpha: pd.DataFrame, chars: dict[str, pd.DataFrame],
                              min_names: int = 20) -> pd.DataFrame:
    """Per-day cross-sectional OLS residual of ``alpha`` on the given characteristics.

    Each day, regress the alpha cross-section on the characteristics' cross-sectional
    RANKS (rank-transform so fat-tailed characteristics like market cap can't dominate
    the fit) plus an intercept, and keep the residual. The residual is orthogonal to
    every characteristic that day — i.e. the alpha with those factor tilts removed.

    A stock missing SOME characteristics is kept with those ranks median-filled (0.5),
    not dropped — requiring complete cases across many sparse fundamental panels would
    otherwise empty the cross-section. Days with fewer than ``min_names`` alpha
    observations come back all-NaN (the backtester treats them as no-position days).
    """
    out = pd.DataFrame(np.nan, index=alpha.index, columns=alpha.columns)
    mats = {k: v.reindex(index=alpha.index, columns=alpha.columns) for k, v in chars.items()}
    for d in alpha.index:
        y = alpha.loc[d]
        X = pd.DataFrame({k: m.loc[d] for k, m in mats.items()})
        X = (X.rank() / X.count()).fillna(0.5)
        msk = y.notna()
        if int(msk.sum()) < min_names:
            continue
        Xm = np.column_stack([np.ones(int(msk.sum()))] + [X.loc[msk, k].to_numpy() for k in X])
        ym = y[msk].to_numpy()
        try:
            coef, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
        except np.linalg.LinAlgError:
            continue
        out.loc[d, msk[msk].index] = ym - Xm @ coef
    return out


def sector_neutralize(alpha: pd.DataFrame, sector: pd.DataFrame) -> pd.DataFrame:
    """Demean each alpha cross-section within its sector each day.

    alpha, sector: date x PERMNO frames. `sector` holds integer/category codes.
    Returns alpha minus its per-(day, sector) mean (sector-relative signal).
    """
    sec = sector.reindex(index=alpha.index, columns=alpha.columns)
    # future_stack=True keeps all (date, stock) cells incl. NaN (pandas 2.x dropped the
    # old `dropna=False`); rows with a missing sector are filtered just below.
    a = alpha.stack(future_stack=True)
    g = sec.stack(future_stack=True)
    frame = pd.DataFrame({"a": a, "g": g}).dropna(subset=["g"])
    date_level = frame.index.get_level_values(0)
    grp_mean = frame.groupby([date_level, frame["g"]])["a"].transform("mean")
    resid = (frame["a"] - grp_mean).unstack()
    return resid.reindex(index=alpha.index, columns=alpha.columns)
