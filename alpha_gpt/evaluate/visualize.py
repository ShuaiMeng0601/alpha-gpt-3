"""Visualization tools for Alpha-GPT pipeline."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def set_style():
    """Set global plotting style."""
    sns.set_theme(style="whitegrid", palette="muted")


def plot_gp_evolution(logbook, out_path: str):
    """Plot maximum and average fitness over GP generations.

    Args:
        logbook: DEAP logbook object or list of generation stats.
        out_path: Path to save the PNG file.
    """
    set_style()

    # Extract data from logbook
    gen = logbook.select("gen")
    fit_max = logbook.select("max")
    fit_avg = logbook.select("avg")

    plt.figure(figsize=(10, 6))
    plt.plot(gen, fit_max, label="Max IC", color="crimson", linewidth=2.5, marker="o")
    plt.plot(gen, fit_avg, label="Average IC", color="royalblue", linewidth=2.5, linestyle="--")

    plt.title("Genetic Programming Evolution: Fitness over Generations", fontsize=14, fontweight="bold")
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Information Coefficient (IC)", fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved GP evolution plot to {out_path}")


def plot_equity_curves(
    backtest_results: dict[str, object],
    out_path: str,
    benchmark: pd.Series | None = None,
):
    """Plot cumulative returns (equity curves) for multiple alphas.

    Args:
        backtest_results: Dictionary mapping alpha names/formulas to BacktestResult objects.
        out_path: Path to save the PNG file.
        benchmark: Optional cumulative return series for a market benchmark.
    """
    set_style()

    plt.figure(figsize=(12, 7))

    for name, btr in backtest_results.items():
        # Plot cumulative returns
        if hasattr(btr, "cumulative_returns") and not btr.cumulative_returns.empty:
            # Ensure the curve starts at 1.0 (or 0% return)
            curve = btr.cumulative_returns

            # Normalize to start at 1.0 if not already
            if curve.iloc[0] != 1.0:
                curve = curve / curve.iloc[0]

            plt.plot(curve.index, curve, label=f"{name} (Ann. Ret: {btr.annual_return:.1%})", linewidth=2)

    if benchmark is not None and not benchmark.empty:
        bm = benchmark / benchmark.iloc[0] if benchmark.iloc[0] != 1.0 else benchmark
        plt.plot(bm.index, bm, label="Market (VW)", linewidth=2,
                 linestyle="--", color="black", alpha=0.7)

    plt.axhline(y=1.0, color='gray', linestyle='-', alpha=0.5)
    plt.title("Top Alpha Signals: Cumulative Out-of-Sample Performance", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return", fontsize=12)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Equity Curves plot to {out_path}")


def plot_gate_ablation(
    curves: dict[str, object],
    out_path: str,
    benchmark: pd.Series | None = None,
):
    """Overlay the test equity curve of the equal-weight portfolio at each selection-gate
    threshold, so the effect of raising the |t-stat| bar on out-of-sample performance is
    visible at a glance. `curves` maps a label (already carrying threshold / n / Sharpe) to a
    BacktestResult.
    """
    set_style()
    plt.figure(figsize=(12, 7))
    for name, btr in curves.items():
        if hasattr(btr, "cumulative_returns") and not btr.cumulative_returns.empty:
            curve = btr.cumulative_returns
            if curve.iloc[0] != 1.0:
                curve = curve / curve.iloc[0]
            plt.plot(curve.index, curve, label=name, linewidth=2)
    if benchmark is not None and not benchmark.empty:
        bm = benchmark / benchmark.iloc[0] if benchmark.iloc[0] != 1.0 else benchmark
        plt.plot(bm.index, bm, label="Market (VW)", linewidth=2, linestyle="--",
                 color="black", alpha=0.7)
    plt.axhline(y=1.0, color="gray", linestyle="-", alpha=0.5)
    plt.title("Selection-gate ablation: test equity by |t-stat| threshold",
              fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return (out-of-sample)", fontsize=12)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved gate-ablation plot to {out_path}")
