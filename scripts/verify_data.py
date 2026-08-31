"""Read-only checks that the rebuilt panels are point-in-time & survivorship-free.

Run after scripts/prepare_data.py:
    python scripts/verify_data.py
"""

import os

import numpy as np
import pandas as pd

PANELS = "data/panels"
MEMBERSHIP = "data/universe/membership.parquet"


def _load(name):
    return pd.read_parquet(os.path.join(PANELS, f"{name}.parquet"))


def main():
    membership = pd.read_parquet(MEMBERSHIP)
    close = _load("close")
    returns = _load("returns")
    fwd = _load("forward_returns")
    checks = []

    def check(name, ok, detail=""):
        checks.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    # 1. Grid consistency
    check("panels share membership grid",
          close.index.equals(membership.index) and list(close.columns) == list(membership.columns),
          f"{close.shape}")
    check("columns are a subset of the universe", set(close.columns) <= set(membership.columns))
    check("panels are float dtype", close.dtypes.eq("float64").all())

    # 2. No non-numeric / non-terminal panel leaked into data/panels
    bad = [f for f in os.listdir(PANELS) if f.endswith(".parquet") and f[:-8] == "sector"]
    check("sector panel kept OUT of data/panels", not bad)

    # 3. forward_returns is the NEXT-day return. On cells where the name is a member
    #    on BOTH day T and T+1 (the unambiguous case, unaffected by entry/exit masking)
    #    forward_returns[T] must equal returns[T+1]. Checking the shift/alignment on
    #    this subset is independent of the production masking formula, so it can catch a
    #    mis-shift or a boundary masking bug (unlike recomputing the exact formula).
    both = membership & membership.shift(-1, fill_value=False)
    nxt = returns.shift(-1)
    aligned = fwd.reindex_like(returns)
    diff = (aligned.where(both) - nxt.where(both)).abs().to_numpy()
    check("forward_returns[T] == returns[T+1] on both-member cells",
          np.nanmax(diff) < 1e-9, f"max abs diff {np.nanmax(diff):.2e}")

    # 3b. No same-day leak: forward must be a FUTURE return, so the shifted panel must
    #     genuinely differ from the same-day returns.
    same_day = np.nanmax((fwd.where(membership) - returns.where(membership)).abs().to_numpy())
    check("forward_returns differs from same-day return (no same-day leak)",
          bool(same_day > 1e-9), f"max abs diff {same_day:.2e}")

    # 4. Survivorship: many names ENTER and EXIT mid-sample (the bias being fixed)
    start, end = membership.index.min(), membership.index.max()
    last_valid = close.apply(lambda c: c.last_valid_index())
    first_valid = close.apply(lambda c: c.first_valid_index())
    exits = int((last_valid < end - pd.Timedelta(days=60)).sum())
    entries = int((first_valid > start + pd.Timedelta(days=60)).sum())
    check("names exit mid-sample (delist/drop-out present)", exits > 50, f"{exits} names exit early")
    check("names enter mid-sample (no static survivor set)", entries > 50, f"{entries} names enter late")
    check("universe larger than legacy 500", close.shape[1] > 800, f"{close.shape[1]} names")

    # 5. Look-ahead canary: quarterly-sourced panels should not start before the
    #    first plausible report date (they ffill from public_date/rdq, ~Q1 lag).
    if os.path.exists(os.path.join(PANELS, "cs_roa.parquet")):
        roa = _load("cs_roa")
        fv = roa.apply(lambda c: c.first_valid_index()).dropna()
        check("cs_* fundamentals start no earlier than price history",
              fv.min() >= start, f"earliest cs_roa {fv.min().date() if len(fv) else 'NA'}")

    # 6. Per-day breadth is enough for IC/quintiles
    breadth = close.notna().sum(axis=1)
    check("median daily breadth >= 200 stocks", breadth.median() >= 200, f"median {int(breadth.median())}")

    print(f"\n{sum(checks)}/{len(checks)} checks passed.")
    raise SystemExit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
