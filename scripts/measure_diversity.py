"""CLI: measure the diversity of a run's alpha book — rho-bar, effective breadth, operator mix.

Effective breadth is what caps a book's Sharpe:

    N_eff = N / (1 + (N-1)*rho_bar)  ->  1/rho_bar   as N grows

so once a book sits near 1/rho_bar, generating MORE alphas buys nothing and the only
lever is lowering the average pairwise correlation. This script reports that ceiling and
how close a run is to it.

Both metrics are label-free (signals only, no forward returns), so this can be run as
often as you like while tuning the generator without contaminating the test split.

    python scripts/measure_diversity.py outputs/probe_subset_v1/alphas.db
    python scripts/measure_diversity.py outputs/*/alphas.db          # compare runs

Always read rho-bar together with mean |IC|: rho-bar alone is gameable, since a generator
emitting pure noise drives rho-bar toward zero while producing nothing worth trading. The
product (skill x sqrt(N_eff)) is the objective.
"""

import argparse
import collections
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from alpha_gpt.config import Config
from alpha_gpt.experiment import load_and_split
from alpha_gpt.factory import db
from alpha_gpt.portfolio.construct import build_signals

# Measured on the 456-idea run (15 hardcoded themes, 3 strategy-prior agents, every idea
# shown the full terminal menu). Printed alongside new runs as the number to beat.
BASELINE = {"rho": 0.151, "neff": 6.6, "mean_abs_sharpe": 0.61}


def pairwise_rho(signals: dict) -> tuple[float, int]:
    """Mean SIGNED pairwise correlation across survivor signals.

    Signed, not absolute: a pair correlated -0.9 is two sides of one bet and cancels in an
    equal-weight book, which is exactly what effective breadth should credit.
    """
    ids = sorted(signals)
    stacked = np.array([signals[i].to_numpy(dtype=float).ravel() for i in ids])
    stacked = stacked[:, np.isfinite(stacked).all(axis=0)]
    stacked -= stacked.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    unit = stacked / np.where(norms == 0, np.nan, norms)
    corr = unit @ unit.T
    vals = corr[np.triu_indices(len(ids), k=1)]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()), len(ids)


def operator_mix(conn) -> tuple[int, int, int]:
    """(alphas using any ts_ operator, total valid alphas, distinct operators used).

    Worth watching next to rho-bar: a book can look diverse in WHICH fields it touches
    while every formula is the same cross-sectional rank composite. That happened — a run
    with per-idea terminal subsets emitted 172 alphas using zero time-series operators.
    """
    exprs = [r[0] for r in conn.execute("SELECT expression FROM alphas WHERE status='ok'")]
    ops = collections.Counter(o for e in exprs for o in re.findall(r"(\w+)\(", e))
    with_ts = sum(1 for e in exprs if re.search(r"\bts_\w+\(", e))
    return with_ts, len(exprs), len(ops)


def report(path: str, pset, panels, min_tstat: float) -> None:
    conn = db.connect(path)
    survivors = db.query(conn, status="ok", min_abs_tstat=min_tstat)
    with_ts, total, n_ops = operator_mix(conn)
    print(f"\n=== {path} ===")
    if not survivors:
        print(f"  no survivors at |t|>={min_tstat} (of {total} valid alphas)")
        return

    signals = build_signals(survivors, pset, panels)
    if len(signals) < 2:
        print(f"  only {len(signals)} signal(s) evaluated — need 2+ for a correlation")
        return
    rho, n = pairwise_rho(signals)
    neff = n / (1 + (n - 1) * rho) if rho > -1 / (n - 1) else float("inf")
    ceiling = 1 / rho if rho > 0 else float("inf")
    mean_ic = float(np.mean([abs(r["is_ic"] or 0.0) for r in survivors]))
    mean_sharpe = float(np.mean([abs(r["is_sharpe"] or 0.0) for r in survivors]))

    print(f"  valid alphas         : {total}")
    print(f"  survivors (|t|>={min_tstat:g})   : {n}  ({100 * n / total:.0f}%)")
    print(f"  rho-bar              : {rho:.4f}   (baseline {BASELINE['rho']})")
    print(f"  effective breadth    : {neff:.2f}   (baseline {BASELINE['neff']})")
    print(f"  ceiling 1/rho-bar    : {ceiling:.2f}   -> {100 * neff / ceiling:.0f}% realized")
    print(f"  mean |in-sample IC|  : {mean_ic:.4f}")
    print(f"  mean |IS Sharpe|     : {mean_sharpe:.3f}   (baseline {BASELINE['mean_abs_sharpe']})")
    print(f"  implied book Sharpe  : {mean_sharpe * neff ** 0.5:.2f}   (skill x sqrt(N_eff))")
    print(f"  alphas using ts_ ops : {with_ts} / {total} ({100 * with_ts / total:.0f}%)")
    print(f"  distinct operators   : {n_ops}")


def main():
    p = argparse.ArgumentParser(description="Measure alpha-book diversity (rho-bar, N_eff)")
    p.add_argument("dbs", nargs="+", help="path(s) to a run's alphas.db")
    p.add_argument("--min-tstat", type=float, default=2.0,
                   help="survivor gate; must match the run you are comparing against")
    args = p.parse_args()

    # Signals are re-evaluated on the TEST split, the same panels the portfolio stage uses,
    # so rho-bar here is the correlation of the book actually traded.
    _in_sample, test, _terminals, pset = load_and_split(Config())
    for path in args.dbs:
        report(path, pset, test.panels, args.min_tstat)


if __name__ == "__main__":
    main()
