"""Robust verification of alpha expressions before storage.

This is the gate that keeps the database clean at scale: an expression must parse,
have sane structure, evaluate without error, and produce a non-degenerate signal
(not all-NaN, not cross-sectionally constant, finite, enough coverage/breadth).
Returns the evaluated signal so callers don't have to re-evaluate.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from deap import gp

from alpha_gpt.expr.engine import eval_expr
from alpha_gpt.expr.seed_injector import normalize_expression, parse_expression


@dataclass
class VerifyResult:
    ok: bool
    reason: str
    normalized: str = ""
    tree: object = None
    signal: pd.DataFrame | None = None
    n_terminals: int = 0
    depth: int = 0
    coverage: float = 0.0


def _structure(tree) -> tuple[int, int]:
    n_terminals = sum(1 for n in tree if isinstance(n, gp.Terminal))
    try:
        depth = tree.height
    except Exception:
        depth = -1
    return n_terminals, depth


# Binary operators whose identical children collapse to a constant panel
# (sub(x,x)=0, safe_div(x,x)=1) or an undefined statistic (ts_corr(x,x) has zero
# variance in most windows). A constant panel then turns ALL-NaN through
# cs_zscore / safe_div / log_abs — the dominant reject class in real debate runs.
_DEGENERATE_IF_SAME_ARGS = ("sub", "safe_div", "ts_corr")
# Unary chains that force a constant panel: sign(abs_val(x)) = 1 wherever valid
# (and abs_val(sign(x)) likewise).
_DEGENERATE_UNARY_CHAINS = {("sign", "abs_val"), ("abs_val", "sign")}


def degenerate_reason(tree) -> str:
    """Static check for expressions that provably collapse to a constant panel.

    Returns a human-readable reason (usable directly as LLM feedback), or "" if
    clean. Pure tree inspection — no data evaluation, so it runs before the
    expensive eval and gives a nameable cause instead of a generic all_nan.
    """
    for i, node in enumerate(tree):
        if not isinstance(node, gp.Primitive):
            continue
        name = node.name
        if node.arity == 2 and name.startswith(_DEGENERATE_IF_SAME_ARGS):
            try:  # defensive: a malformed tree must reject one formula, never crash a run
                left = tree.searchSubtree(i + 1)
                right = tree.searchSubtree(left.stop)
            except IndexError:
                continue  # arity-incomplete tree (parse_expression should have caught it)
            if [n.name for n in tree[left]] == [n.name for n in tree[right]]:
                sub_expr = str(gp.PrimitiveTree(tree[left]))
                return (f"{name}({sub_expr}, {sub_expr}) applies {name} to two identical "
                        "arguments, which collapses to a constant (or undefined) panel")
        if node.arity == 1 and i + 1 < len(tree):
            child = tree[i + 1]
            if isinstance(child, gp.Primitive) and (name, child.name) in _DEGENERATE_UNARY_CHAINS:
                return (f"{name}({child.name}(...)) is constant wherever defined -- "
                        "it carries no cross-sectional information")
    return ""


def verify_expression(
    expr: str,
    pset: gp.PrimitiveSet,
    panels: dict[str, pd.DataFrame],
    *,
    max_depth: int = 8,
    min_coverage: float = 0.05,
    min_breadth: int = 20,
    min_active_days_frac: float = 0.3,
) -> VerifyResult:
    """Verify a single expression against `panels` (typically the train split)."""
    if not expr or not isinstance(expr, str):
        return VerifyResult(False, "empty")

    tree = parse_expression(expr, pset)
    if tree is None:
        return VerifyResult(False, "parse_error")

    # Once the tree parses, every result (rejects included) carries the normalized form,
    # so in-run dedup and reject persistence key on the canonical expression.
    normalized = normalize_expression(expr)
    n_terminals, depth = _structure(tree)
    if n_terminals < 1:
        return VerifyResult(False, "no_terminals", normalized=normalized, tree=tree)
    if depth < 0 or depth > max_depth:
        return VerifyResult(False, "too_deep", normalized=normalized, tree=tree,
                            n_terminals=n_terminals)
    if degenerate_reason(tree):
        return VerifyResult(False, "degenerate", normalized=normalized, tree=tree,
                            n_terminals=n_terminals, depth=depth)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            signal = eval_expr(tree, pset, panels)
    except Exception:
        return VerifyResult(False, "eval_error", normalized=normalized, tree=tree,
                            n_terminals=n_terminals, depth=depth)

    if signal is None or not isinstance(signal, pd.DataFrame) or signal.empty:
        return VerifyResult(False, "empty_signal", normalized=normalized, tree=tree,
                            n_terminals=n_terminals, depth=depth)

    signal = signal.replace([np.inf, -np.inf], np.nan)
    if bool(signal.isna().all().all()):
        return VerifyResult(False, "all_nan", normalized=normalized, tree=tree,
                            n_terminals=n_terminals, depth=depth)

    coverage = float(signal.notna().to_numpy().mean())
    breadth = signal.notna().sum(axis=1)
    meta = dict(normalized=normalized, tree=tree, signal=signal,
                n_terminals=n_terminals, depth=depth, coverage=coverage)
    if coverage < min_coverage or float(breadth.median()) < min_breadth:
        return VerifyResult(False, "sparse", **meta)

    # Reject signals with no cross-sectional variation (constant -> IC undefined).
    active_frac = float((signal.std(axis=1) > 1e-12).mean())
    if active_frac < min_active_days_frac:
        return VerifyResult(False, "constant", **meta)

    return VerifyResult(True, "ok", **meta)


# Reason -> actionable feedback for the debate's revise turn. Return-blind on purpose:
# these describe PROPERTIES of the signal (what it trades, how it churns), never its
# predictive performance, so the debate cannot tune toward the objective.
_REJECT_HINTS = {
    "parse_error": "does not parse: check operator names, allowed windows, and arity "
                   "against the catalog (only listed operators/terminals exist)",
    "too_deep": "is nested too deeply -- simplify the expression",
    "all_nan": "evaluates to all-NaN: usually a sub-expression collapses to a constant "
               "which then breaks cs_zscore / ts_corr / safe_div / log_abs",
    "constant": "has (near-)zero cross-sectional variation on most days -- it cannot rank stocks",
    "sparse": "trades too few stocks (thin coverage) -- use denser fields or fewer sparse terminals",
    "eval_error": "crashes during evaluation",
    "empty": "is empty",
    "no_terminals": "contains no data terminals",
}


def diagnose_expression(
    expr: str,
    pset: gp.PrimitiveSet,
    panels: dict[str, pd.DataFrame],
    *,
    max_depth: int = 20,
) -> str:
    """One-line, return-blind diagnostic for a drafted formula (LLM feedback).

    Failures come back as "REJECTED (<reason>): <what to fix>"; passing formulas get
    measured signal properties (coverage, breadth, rank turnover, distinct values,
    dead days). No forward returns are touched anywhere here.
    """
    vr = verify_expression(expr, pset, panels, max_depth=max_depth)
    if not vr.ok:
        if vr.reason == "degenerate" and vr.tree is not None:
            return f"REJECTED (degenerate): {degenerate_reason(vr.tree)}"
        if vr.reason == "too_deep":
            return (f"REJECTED (too_deep): nesting depth {getattr(vr.tree, 'height', '?')} "
                    f"exceeds the cap of {max_depth} -- simplify the expression")
        hint = _REJECT_HINTS.get(vr.reason, "fails verification")
        return f"REJECTED ({vr.reason}): {hint}"

    signal = vr.signal
    valid = signal.notna()
    breadth = valid.sum(axis=1)
    # Rank turnover: mean |change| in cross-sectional percentile rank between
    # consecutive rows (a rebalance-frequency churn proxy; return-blind).
    u = signal.rank(axis=1, pct=True)
    turnover = float((u - u.shift(1)).abs().mean(axis=1).mean())
    # Ties: median share of valid names carrying a distinct value each day.
    distinct_share = float((signal.nunique(axis=1) / breadth.replace(0, np.nan)).median())
    dead_days = float((signal.std(axis=1).fillna(0.0) <= 1e-12).mean())
    return (f"OK | coverage {vr.coverage:.0%} of cells, ~{int(breadth.median())} stocks/day, "
            f"rank turnover {turnover:.2f}, distinct values {distinct_share:.0%} of names/day, "
            f"dead days {dead_days:.0%}")
