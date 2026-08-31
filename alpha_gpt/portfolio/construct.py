"""Combine many stored alphas into one portfolio signal.

Survivors selected on in-sample metrics are re-evaluated on the target split (test),
cross-sectionally z-scored, then combined. Each alpha keeps its NATIVE orientation — we
never sign-flip by in-sample IC. Methods: equal (1/N), orthogonalized.

LOOK-AHEAD CONTROL: the method that *learns* a combination (orthogonalized's
residualization betas) fits those parameters on the in-sample split and applies them
fixed to test — never on test itself. equal uses no fitted parameters at all. So no test
information ever enters the construction.
"""

import logging

import numpy as np
import pandas as pd

from alpha_gpt.evaluate.neutralize import zscore_neutralize
from alpha_gpt.expr.engine import eval_expr
from alpha_gpt.expr.seed_injector import parse_expression

logger = logging.getLogger(__name__)


def _signal_on(rec: dict, pset, panels: dict) -> pd.DataFrame | None:
    """Parse + evaluate one alpha on `panels`; z-score. The signal keeps its NATIVE
    orientation — we never sign-flip by in-sample IC. Flipping would use in-sample returns
    to pick the direction (data-mining the sign); a hypothesis-driven alpha already encodes
    its intended direction in the expression."""
    tree = parse_expression(rec["expression"], pset)
    if tree is None:
        return None
    sig = eval_expr(tree, pset, panels).replace([np.inf, -np.inf], np.nan)
    if sig is None or sig.empty or bool(sig.isna().all().all()):
        return None
    return zscore_neutralize(sig)


def build_signals(survivors: list[dict], pset, panels: dict, progress=None) -> dict[int, pd.DataFrame]:
    """Re-evaluate every survivor on `panels` → {id: z-scored, native-orientation signal}."""
    out = {}
    items = survivors if progress is None else progress(survivors)
    for r in items:
        sig = _signal_on(r, pset, panels)
        if sig is not None:
            out[r["id"]] = sig
    return out


# --- combination primitives ---------------------------------------------------

def _weighted_signals(sigs: list[pd.DataFrame], w: np.ndarray) -> pd.DataFrame:
    num = sum(wi * x.fillna(0.0) for wi, x in zip(w, sigs))
    den = sum(wi * x.notna().astype(float) for wi, x in zip(w, sigs))
    return num.div(den.replace(0, np.nan))


def _mean_signals(sigs: list[pd.DataFrame]) -> pd.DataFrame:
    return _weighted_signals(sigs, np.ones(len(sigs)))


def _ols_beta(s: pd.DataFrame, comp: pd.DataFrame) -> float:
    """Pooled OLS slope of signal `s` on composite `comp` over their common cells."""
    a, b = s.to_numpy().ravel(), comp.to_numpy().ravel()
    msk = ~(np.isnan(a) | np.isnan(b))
    denom = float(np.dot(b[msk], b[msk]))
    return float(np.dot(a[msk], b[msk]) / denom) if denom > 0 else 0.0


def _accum(num, cnt, r):
    rf, rc = r.fillna(0.0), r.notna().astype(float)
    return (rf if num is None else num + rf), (rc if cnt is None else cnt + rc)


def _orthogonalize_betas(signals: dict, weights: dict) -> list:
    """Fit residualization betas over a SIGNALS DICT (self-contained); see note above."""
    order = sorted(signals, key=lambda i: abs(weights.get(i, 0.0)), reverse=True)
    num = cnt = None
    betas = []
    for i in order:
        s = signals[i]
        if num is not None:
            comp = num.div(cnt.replace(0, np.nan))
            beta = _ols_beta(s, comp)
            res = s - beta * comp
        else:
            beta, res = None, s
        betas.append((i, beta))
        num, cnt = _accum(num, cnt, res)
    return betas


# --- fit combiners on the validation split (no look-ahead) ---------------------

def fit_combiners(survivors, pset, fit_panels, weights, progress=None):
    """One streaming pass over the FIT split (validation) → orthogonalization betas.

    Returns ``ortho_betas``: an ordered list of ``(id, beta)`` where ``beta``
    residualizes each signal against the running composite (``None`` for the first /
    no-prior case). Streaming — holds only the running composite frame. These betas are
    later applied (fixed) to the test split, so no test information enters the fit.
    """
    order = sorted(survivors, key=lambda r: abs(weights.get(r["id"], 0.0)), reverse=True)
    if progress is not None:
        order = progress(order)
    num = cnt = None          # orthogonalization running composite
    betas = []
    for r in order:
        sig = _signal_on(r, pset, fit_panels)
        if sig is None:
            continue
        i = r["id"]
        if num is not None:
            comp = num.div(cnt.replace(0, np.nan))
            beta = _ols_beta(sig, comp)
            res = sig - beta * comp
        else:
            beta, res = None, sig
        betas.append((i, beta))
        num, cnt = _accum(num, cnt, res)
    return betas


def _apply_orthogonalization(betas, apply_signals):
    """Apply validation-fit betas to the apply (test) signals, in the fit order."""
    num = cnt = None
    used = []
    for i, beta in betas:
        s = apply_signals.get(i)
        if s is None:
            continue
        if num is not None and beta is not None:
            comp = num.div(cnt.replace(0, np.nan))
            res = s - beta * comp
        else:
            res = s
        used.append(i)
        num, cnt = _accum(num, cnt, res)
    if num is None:
        return None, []
    return num.div(cnt.replace(0, np.nan)), used


# --- apply on the target (test) split ------------------------------------------

def combine(apply_signals: dict, method: str = "equal", weights: dict | None = None,
            betas: list | None = None):
    """Build the composite on `apply_signals`.

    For orthogonalized, pass `betas` fit on a SEPARATE split (validation) to avoid
    look-ahead — the pipeline does exactly this. If they are not provided, combine
    self-fits on `apply_signals` (used by unit tests and standalone calls; that path has
    look-ahead if `apply_signals` is the test split).
    """
    ids = list(apply_signals)
    if not ids:
        return None, []
    weights = weights or {i: 1.0 for i in ids}

    if method == "equal":
        return _mean_signals([apply_signals[i] for i in ids]), ids
    if method == "orthogonalized":
        if betas is None:
            betas = _orthogonalize_betas(apply_signals, weights)
        return _apply_orthogonalization(betas, apply_signals)
    raise ValueError(f"unknown method: {method}")
