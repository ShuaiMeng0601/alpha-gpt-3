"""Alpha operators on panel DataFrames (index=date, columns=PERMNO).

All operators take and return pd.DataFrame. NaN-safe (propagate, don't crash).
Time-series operators are curried with fixed window sizes for DEAP compatibility.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Time-series operators (per-stock, along the time axis)
# ---------------------------------------------------------------------------

def _make_ts_mean(window):
    def ts_mean(x: pd.DataFrame) -> pd.DataFrame:
        return x.rolling(window, min_periods=1).mean()
    ts_mean.__name__ = f"ts_mean_{window}"
    return ts_mean

def _make_ts_std(window):
    def ts_std(x: pd.DataFrame) -> pd.DataFrame:
        return x.rolling(window, min_periods=2).std()
    ts_std.__name__ = f"ts_std_{window}"
    return ts_std

def _make_ts_delta(window):
    def ts_delta(x: pd.DataFrame) -> pd.DataFrame:
        return x - x.shift(window)
    ts_delta.__name__ = f"ts_delta_{window}"
    return ts_delta

def _make_ts_rank(window):
    def ts_rank(x: pd.DataFrame) -> pd.DataFrame:
        return x.rolling(window, min_periods=1).rank(pct=True)
    ts_rank.__name__ = f"ts_rank_{window}"
    return ts_rank

def _make_ts_min(window):
    def ts_min(x: pd.DataFrame) -> pd.DataFrame:
        return x.rolling(window, min_periods=1).min()
    ts_min.__name__ = f"ts_min_{window}"
    return ts_min

def _make_ts_max(window):
    def ts_max(x: pd.DataFrame) -> pd.DataFrame:
        return x.rolling(window, min_periods=1).max()
    ts_max.__name__ = f"ts_max_{window}"
    return ts_max

def _make_ts_returns(window):
    def ts_returns(x: pd.DataFrame) -> pd.DataFrame:
        return x.pct_change(window)
    ts_returns.__name__ = f"ts_returns_{window}"
    return ts_returns

def _make_ts_corr(window):
    def ts_corr(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        return x.rolling(window, min_periods=3).corr(y)
    ts_corr.__name__ = f"ts_corr_{window}"
    return ts_corr


def _make_ts_shift(window):
    def ts_shift(x: pd.DataFrame) -> pd.DataFrame:
        return x.shift(window)
    ts_shift.__name__ = f"ts_shift_{window}"
    return ts_shift

def _make_ts_sum(window):
    def ts_sum(x: pd.DataFrame) -> pd.DataFrame:
        return x.rolling(window, min_periods=1).sum()
    ts_sum.__name__ = f"ts_sum_{window}"
    return ts_sum

def _make_ts_median(window):
    def ts_median(x: pd.DataFrame) -> pd.DataFrame:
        return x.rolling(window, min_periods=1).median()
    ts_median.__name__ = f"ts_median_{window}"
    return ts_median

def _make_ts_skew(window):
    def ts_skew(x: pd.DataFrame) -> pd.DataFrame:
        return x.rolling(window, min_periods=3).skew()
    ts_skew.__name__ = f"ts_skew_{window}"
    return ts_skew

def _make_ts_kurt(window):
    def ts_kurt(x: pd.DataFrame) -> pd.DataFrame:
        return x.rolling(window, min_periods=4).kurt()
    ts_kurt.__name__ = f"ts_kurt_{window}"
    return ts_kurt

def _make_ts_ema(span):
    def ts_ema(x: pd.DataFrame) -> pd.DataFrame:
        return x.ewm(span=span, min_periods=1).mean()
    ts_ema.__name__ = f"ts_ema_{span}"
    return ts_ema

def _make_ts_zscore(window):
    """Rolling z-score: where x sits relative to its OWN recent history (AG1 ts_zscore_scale)."""
    def ts_zscore(x: pd.DataFrame) -> pd.DataFrame:
        mean = x.rolling(window, min_periods=2).mean()
        std = x.rolling(window, min_periods=2).std()
        return (x - mean) / std.replace(0, np.nan)
    ts_zscore.__name__ = f"ts_zscore_{window}"
    return ts_zscore

def _make_ts_ir(window):
    """Rolling mean / rolling std — a per-stock information ratio (AG1 ts_ir)."""
    def ts_ir(x: pd.DataFrame) -> pd.DataFrame:
        mean = x.rolling(window, min_periods=2).mean()
        std = x.rolling(window, min_periods=2).std()
        return mean / std.replace(0, np.nan)
    ts_ir.__name__ = f"ts_ir_{window}"
    return ts_ir

def _make_ts_maxmin_scale(window):
    """Position within the rolling range, in [0, 1] — the 52-week-high family."""
    def ts_maxmin_scale(x: pd.DataFrame) -> pd.DataFrame:
        lo = x.rolling(window, min_periods=1).min()
        hi = x.rolling(window, min_periods=1).max()
        return (x - lo) / (hi - lo).replace(0, np.nan)
    ts_maxmin_scale.__name__ = f"ts_maxmin_scale_{window}"
    return ts_maxmin_scale

def _make_ts_max_diff(window):
    def ts_max_diff(x: pd.DataFrame) -> pd.DataFrame:
        return x - x.rolling(window, min_periods=1).max()
    ts_max_diff.__name__ = f"ts_max_diff_{window}"
    return ts_max_diff

def _make_ts_min_diff(window):
    def ts_min_diff(x: pd.DataFrame) -> pd.DataFrame:
        return x - x.rolling(window, min_periods=1).min()
    ts_min_diff.__name__ = f"ts_min_diff_{window}"
    return ts_min_diff

def _make_ts_delta_ratio(window):
    """Percent change measured against |past| — ts_delta normalised, safe for sign flips."""
    def ts_delta_ratio(x: pd.DataFrame) -> pd.DataFrame:
        past = x.shift(window)
        return (x - past) / past.abs().replace(0, np.nan)
    ts_delta_ratio.__name__ = f"ts_delta_ratio_{window}"
    return ts_delta_ratio

def _make_ts_decayed_linear(window):
    """Linearly weighted moving average — most weight on the most recent observation."""
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()
    def ts_decayed_linear(x: pd.DataFrame) -> pd.DataFrame:
        return sum(w * x.shift(window - 1 - i) for i, w in enumerate(weights))
    ts_decayed_linear.__name__ = f"ts_decayed_linear_{window}"
    return ts_decayed_linear

def _make_ts_linear_reg(window):
    """Slope of an OLS fit on the last `window` observations (trend strength, not level)."""
    t = np.arange(window, dtype=float)
    t_centered = t - t.mean()
    t_var = (t_centered ** 2).sum()
    def ts_linear_reg(x: pd.DataFrame) -> pd.DataFrame:
        num = sum(c * x.shift(window - 1 - i) for i, c in enumerate(t_centered))
        return num / t_var
    ts_linear_reg.__name__ = f"ts_linear_reg_{window}"
    return ts_linear_reg

def _make_ts_cov(window):
    def ts_cov(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        return x.rolling(window, min_periods=3).cov(y)
    ts_cov.__name__ = f"ts_cov_{window}"
    return ts_cov


def _rolling_argext(x: pd.DataFrame, window: int, take_max: bool) -> pd.DataFrame:
    """Index within the window (0 = oldest) of its extreme value — 'how long since the high'.

    Chunked over columns: ``sliding_window_view`` itself is a free view, but the NaN masking
    below materialises a (rows, cols, window) array, which is gigabytes for a full panel.
    pandas' ``rolling().apply()`` would avoid that but runs a Python callback per window —
    millions of them here — so it is far too slow for the factory's throughput.
    """
    a = x.to_numpy(dtype=float)
    n, m = a.shape
    out = np.full((n, m), np.nan)
    if n >= window:
        chunk = max(1, int(5e6 // (n * window)))  # bound each block to ~40 MB
        fill = -np.inf if take_max else np.inf
        for lo in range(0, m, chunk):
            block = np.lib.stride_tricks.sliding_window_view(
                a[:, lo:lo + chunk], window, axis=0)
            nan_mask = np.isnan(block)
            filled = np.where(nan_mask, fill, block)
            idx = (filled.argmax(-1) if take_max else filled.argmin(-1)).astype(float)
            idx[nan_mask.all(axis=-1)] = np.nan
            out[window - 1:, lo:lo + chunk] = idx
    return pd.DataFrame(out, index=x.index, columns=x.columns)

def _make_ts_argmax(window):
    def ts_argmax(x: pd.DataFrame) -> pd.DataFrame:
        return _rolling_argext(x, window, take_max=True)
    ts_argmax.__name__ = f"ts_argmax_{window}"
    return ts_argmax

def _make_ts_argmin(window):
    def ts_argmin(x: pd.DataFrame) -> pd.DataFrame:
        return _rolling_argext(x, window, take_max=False)
    ts_argmin.__name__ = f"ts_argmin_{window}"
    return ts_argmin


# Generate curried operators for standard windows
WINDOWS_SHORT = [5, 10, 20]
WINDOWS_ALL = [5, 10, 20, 60]

ts_mean_5, ts_mean_10, ts_mean_20, ts_mean_60 = [_make_ts_mean(w) for w in WINDOWS_ALL]
ts_std_5, ts_std_10, ts_std_20, ts_std_60 = [_make_ts_std(w) for w in WINDOWS_ALL]
ts_delta_5, ts_delta_10, ts_delta_20 = [_make_ts_delta(w) for w in WINDOWS_SHORT]
ts_rank_5, ts_rank_10, ts_rank_20 = [_make_ts_rank(w) for w in WINDOWS_SHORT]
ts_min_5, ts_min_10, ts_min_20 = [_make_ts_min(w) for w in WINDOWS_SHORT]
ts_max_5, ts_max_10, ts_max_20 = [_make_ts_max(w) for w in WINDOWS_SHORT]
ts_returns_1, ts_returns_5, ts_returns_20 = [_make_ts_returns(w) for w in [1, 5, 20]]
ts_corr_10, ts_corr_20 = [_make_ts_corr(w) for w in [10, 20]]

# Windows for the operators added from the Alpha-GPT 1.0 catalog. Deliberately narrower
# than a full sweep: operator TYPE is the axis that buys expression diversity, while extra
# windows of the same operator mostly multiply near-duplicates and inflate the GP search
# space (and the prompt the model has to read).
ts_shift_1, ts_shift_5, ts_shift_20 = [_make_ts_shift(w) for w in [1, 5, 20]]
ts_sum_5, ts_sum_20 = [_make_ts_sum(w) for w in [5, 20]]
ts_median_20 = _make_ts_median(20)
ts_skew_60 = _make_ts_skew(60)
ts_kurt_60 = _make_ts_kurt(60)
ts_ema_10, ts_ema_60 = [_make_ts_ema(w) for w in [10, 60]]
ts_zscore_20, ts_zscore_60 = [_make_ts_zscore(w) for w in [20, 60]]
ts_ir_20, ts_ir_60 = [_make_ts_ir(w) for w in [20, 60]]
ts_maxmin_scale_20, ts_maxmin_scale_60 = [_make_ts_maxmin_scale(w) for w in [20, 60]]
ts_max_diff_20, ts_max_diff_60 = [_make_ts_max_diff(w) for w in [20, 60]]
ts_min_diff_20, ts_min_diff_60 = [_make_ts_min_diff(w) for w in [20, 60]]
ts_delta_ratio_5, ts_delta_ratio_20 = [_make_ts_delta_ratio(w) for w in [5, 20]]
ts_decayed_linear_5, ts_decayed_linear_20 = [_make_ts_decayed_linear(w) for w in [5, 20]]
ts_linear_reg_20 = _make_ts_linear_reg(20)
ts_argmax_20 = _make_ts_argmax(20)
ts_argmin_20 = _make_ts_argmin(20)
ts_cov_20 = _make_ts_cov(20)


# ---------------------------------------------------------------------------
# Cross-sectional operators (across stocks, per date)
# ---------------------------------------------------------------------------

def cs_rank(x: pd.DataFrame) -> pd.DataFrame:
    """Rank across stocks each day (percentile rank)."""
    return x.rank(axis=1, pct=True)

def cs_zscore(x: pd.DataFrame) -> pd.DataFrame:
    """Z-score across stocks each day."""
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    return x.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)

def cs_winsorize(x: pd.DataFrame) -> pd.DataFrame:
    """Clip each day's cross-section to its 1st-99th percentile (AG1 winsorize_scale).

    Lets a hypothesis say 'this effect is real but not driven by the tails' — previously
    inexpressible, since every path to a bounded signal went through a full rank."""
    lo = x.quantile(0.01, axis=1)
    hi = x.quantile(0.99, axis=1)
    return x.clip(lower=lo, upper=hi, axis=0)


# ---------------------------------------------------------------------------
# Element-wise operators
# ---------------------------------------------------------------------------

def add(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return x + y

def sub(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return x - y

def mul(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return x * y

def safe_div(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Element-wise division, replacing div-by-zero with NaN."""
    return x / y.replace(0, np.nan)

def log_abs(x: pd.DataFrame) -> pd.DataFrame:
    """Log of absolute value (NaN for zero)."""
    return np.log(x.abs().replace(0, np.nan))

def abs_val(x: pd.DataFrame) -> pd.DataFrame:
    return x.abs()

def sign(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x)

def neg(x: pd.DataFrame) -> pd.DataFrame:
    return -x

def relu(x: pd.DataFrame) -> pd.DataFrame:
    """Keep the positive part, zero the rest — one-sided exposure."""
    return x.clip(lower=0)

def square(x: pd.DataFrame) -> pd.DataFrame:
    """x^2 (AG1 `pow`): magnitude regardless of direction, for U-shaped effects."""
    return x ** 2

def signed_sqrt(x: pd.DataFrame) -> pd.DataFrame:
    """sign(x)*sqrt(|x|) (AG1 `pow_sign`): compresses extremes but keeps direction."""
    return np.sign(x) * np.sqrt(x.abs())

def cwise_max(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Element-wise max of two panels."""
    return x.where(x > y, y)

def cwise_min(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Element-wise min of two panels."""
    return x.where(x < y, y)

def greater(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """1.0 where x > y else 0.0 (NaN preserved) — the missing CONDITIONAL primitive.

    Multiplying a signal by one of these is how a hypothesis expresses 'only in this
    regime' or 'only among these names'; the debate has a `filter` formula_role that
    previously had no operator able to implement it."""
    mask = (x > y).astype(float)
    return mask.where(x.notna() & y.notna())

def less(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """1.0 where x < y else 0.0 (NaN preserved)."""
    mask = (x < y).astype(float)
    return mask.where(x.notna() & y.notna())

def normed_rank_diff(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """cs_rank(x) - cs_rank(y): a spread between two characteristics on a common scale."""
    return cs_rank(x) - cs_rank(y)


# ---------------------------------------------------------------------------
# Registry: all operators grouped by type for easy access
# ---------------------------------------------------------------------------

# Unary operators (DataFrame -> DataFrame)
UNARY_OPS = [
    ts_mean_5, ts_mean_10, ts_mean_20, ts_mean_60,
    ts_std_5, ts_std_10, ts_std_20, ts_std_60,
    ts_delta_5, ts_delta_10, ts_delta_20,
    ts_rank_5, ts_rank_10, ts_rank_20,
    ts_min_5, ts_min_10, ts_min_20,
    ts_max_5, ts_max_10, ts_max_20,
    ts_returns_1, ts_returns_5, ts_returns_20,
    # --- added from the Alpha-GPT 1.0 operator catalog ---
    ts_shift_1, ts_shift_5, ts_shift_20,
    ts_sum_5, ts_sum_20,
    ts_median_20, ts_skew_60, ts_kurt_60,
    ts_ema_10, ts_ema_60,
    ts_zscore_20, ts_zscore_60,
    ts_ir_20, ts_ir_60,
    ts_maxmin_scale_20, ts_maxmin_scale_60,
    ts_max_diff_20, ts_max_diff_60,
    ts_min_diff_20, ts_min_diff_60,
    ts_delta_ratio_5, ts_delta_ratio_20,
    ts_decayed_linear_5, ts_decayed_linear_20,
    ts_linear_reg_20, ts_argmax_20, ts_argmin_20,
    cs_rank, cs_zscore, cs_winsorize,
    log_abs, abs_val, sign, neg,
    relu, square, signed_sqrt,
]

# Binary operators (DataFrame, DataFrame -> DataFrame)
BINARY_OPS = [
    add, sub, mul, safe_div,
    ts_corr_10, ts_corr_20,
    # --- added from the Alpha-GPT 1.0 operator catalog ---
    ts_cov_20,
    cwise_max, cwise_min,
    greater, less, normed_rank_diff,
]

# All operators with metadata
ALL_OPS = {op.__name__: op for op in UNARY_OPS + BINARY_OPS}
