"""Per-panel data-quality / coverage report over the point-in-time universe.

A field should only become a GP terminal if it has enough cross-sectional
breadth (stocks/day) and overall coverage; the IC/backtest code drops NaN
pairwise per day, so sparse fields silently shrink the effective universe.
"""

import os

import pandas as pd

from alpha_gpt.expr.primitives import NON_TERMINAL_PANELS as NON_TERMINAL  # single source of truth


def panel_coverage(panel: pd.DataFrame, membership: pd.DataFrame | None = None) -> dict:
    """Coverage stats for one panel, optionally restricted to a membership mask."""
    if membership is not None:
        panel = panel.reindex(index=membership.index, columns=membership.columns).where(membership)
        denom = int(membership.values.sum())
    else:
        denom = int(panel.notna().size)
    nonnan = panel.notna()
    breadth = nonnan.sum(axis=1)
    valid = breadth[breadth > 0]
    return {
        "coverage": round(int(nonnan.values.sum()) / denom, 4) if denom else 0.0,
        "median_breadth": float(breadth.median()),
        "mean_breadth": round(float(breadth.mean()), 1),
        "first_valid": str(valid.index.min().date()) if len(valid) else None,
        "last_valid": str(valid.index.max().date()) if len(valid) else None,
        "n_cols": int(panel.shape[1]),
    }


def build_coverage_report(panels_dir: str, membership: pd.DataFrame | None = None) -> pd.DataFrame:
    """Coverage stats for every candidate-terminal panel in panels_dir."""
    rows = []
    for f in sorted(os.listdir(panels_dir)):
        if not f.endswith(".parquet") or f.startswith("_"):
            continue
        name = f[:-len(".parquet")]
        if name in NON_TERMINAL:
            continue
        rec = {"field": name}
        rec.update(panel_coverage(pd.read_parquet(os.path.join(panels_dir, f)), membership))
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("coverage", ascending=False).reset_index(drop=True)


def qualifies(rec: dict, min_coverage: float = 0.20, min_breadth: int = 100) -> bool:
    """Whether a field is dense enough to be a useful GP terminal."""
    return rec["coverage"] >= min_coverage and rec["median_breadth"] >= min_breadth


def main():
    import argparse
    from alpha_gpt.config import Config

    parser = argparse.ArgumentParser(description="Panel coverage report")
    parser.add_argument("--data-dir", default="data/panels")
    parser.add_argument("--membership", default="data/universe/membership.parquet")
    args = parser.parse_args()
    config = Config()

    membership = pd.read_parquet(args.membership) if os.path.exists(args.membership) else None
    df = build_coverage_report(args.data_dir, membership)
    df["terminal_ok"] = df.apply(
        lambda r: qualifies(r, config.min_coverage, config.min_breadth), axis=1
    )
    out = os.path.join(args.data_dir, "_coverage_report.csv")
    df.to_csv(out, index=False)
    n_ok = int(df["terminal_ok"].sum())
    print(df.to_string(index=False))
    print(f"\n{n_ok}/{len(df)} fields qualify as terminals "
          f"(coverage>={config.min_coverage}, breadth>={config.min_breadth}/day)")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
