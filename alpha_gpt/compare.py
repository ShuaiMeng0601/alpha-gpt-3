"""Aggregate ``outputs/*/pipeline_summary.json`` into a cross-source comparison.

Each pipeline run (see :mod:`alpha_gpt.pipeline`) writes a ``pipeline_summary.json`` with
its generation source and the out-of-sample portfolio result. This groups those runs by
source (random / gp / llm / debate) and reports the mean ± std across runs of the headline
equal-weight portfolio's Sharpe / IC / annual return, plus the survivor count and LLM cost.
That table is the ablation: does the economic reasoning (llm, debate) beat the mechanical
baselines (random, gp)?
"""


from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np


def _headline_row(portfolio: dict) -> dict | None:
    """The equal-weight portfolio row (the honest, fixed per-arm number).

    Comparing a FIXED method across arms is fair; the pipeline's own ``best`` row is chosen
    per-run on validation and can differ by arm, so we prefer ``equal`` and fall back to
    ``best`` only if it is absent.
    """
    rows = portfolio.get("rows", []) if portfolio else []
    for r in rows:
        if r.get("method") == "equal":
            return r
    return portfolio.get("best") if portfolio else None


def _aggregate(by_source: dict[str, list[dict]]) -> list[dict]:
    out = []
    for source, runs in sorted(by_source.items()):
        sharpes, ics, rets, survs, costs = [], [], [], [], []
        for summary in runs:
            row = _headline_row(summary.get("portfolio", {}))
            if row is None:
                continue
            sharpes.append(row.get("sharpe", 0.0))
            ics.append(row.get("ic", 0.0))
            rets.append(row.get("annual_return", 0.0))
            survs.append(summary.get("n_survivors", 0))
            costs.append(summary.get("factory", {}).get("est_cost_usd", 0.0))
        if not sharpes:
            continue
        out.append({
            "source": source, "n_runs": len(sharpes),
            "sharpe_mean": float(np.mean(sharpes)), "sharpe_std": float(np.std(sharpes)),
            "ic_mean": float(np.mean(ics)), "ic_std": float(np.std(ics)),
            "annual_return_mean": float(np.mean(rets)),
            "n_survivors_mean": float(np.mean(survs)),
            "est_cost_usd_mean": float(np.mean(costs)),
        })
    return out


def _print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 88)
    print("CROSS-SOURCE COMPARISON — equal-weight portfolio, out-of-sample (test)")
    print("=" * 88)
    print(f"{'source':<12}{'runs':>5}{'Sharpe':>18}{'IC':>18}{'AnnRet':>9}{'#surv':>8}{'$cost':>9}")
    print("-" * 88)
    for r in rows:
        print(f"{r['source']:<12}{r['n_runs']:>5}"
              f"{r['sharpe_mean']:>9.2f}+/-{r['sharpe_std']:<6.2f}"
              f"{r['ic_mean']:>9.4f}+/-{r['ic_std']:<6.4f}"
              f"{r['annual_return_mean']:>8.1%}{r['n_survivors_mean']:>8.0f}{r['est_cost_usd_mean']:>9.4f}")


def _write_markdown(path: str, rows: list[dict]) -> None:
    with open(path, "w") as f:
        f.write("# Cross-Source Comparison\n\n")
        f.write("Equal-weight portfolio, out-of-sample (test split), mean ± std across runs.\n\n")
        f.write("| Source | Runs | Sharpe | IC | Ann. Return | # Survivors | Est. Cost |\n")
        f.write("|--------|------|--------|-----|-------------|-------------|-----------|\n")
        for r in rows:
            f.write(f"| {r['source']} | {r['n_runs']} | "
                    f"{r['sharpe_mean']:.2f} ± {r['sharpe_std']:.2f} | "
                    f"{r['ic_mean']:.4f} ± {r['ic_std']:.4f} | "
                    f"{r['annual_return_mean']:.1%} | {r['n_survivors_mean']:.0f} | "
                    f"${r['est_cost_usd_mean']:.4f} |\n")


def run_compare(outputs_dir: str = "outputs") -> None:
    """Scan ``outputs/*/pipeline_summary.json`` and write a cross-source comparison report."""
    files = sorted(glob.glob(os.path.join(outputs_dir, "*", "pipeline_summary.json")))
    if not files:
        print(f"No pipeline_summary.json files found in {outputs_dir}/")
        return

    by_source: dict[str, list[dict]] = defaultdict(list)
    for fpath in files:
        with open(fpath) as f:
            summary = json.load(f)
        source = summary.get("params", {}).get("source", "unknown")
        by_source[source].append(summary)

    rows = _aggregate(by_source)
    _print_summary(rows)

    compare_dir = os.path.join(outputs_dir, "comparison")
    os.makedirs(compare_dir, exist_ok=True)
    _write_markdown(os.path.join(compare_dir, "summary_table.md"), rows)
    with open(os.path.join(compare_dir, "comparison.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nComparison complete. See {compare_dir}/")
