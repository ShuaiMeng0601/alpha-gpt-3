"""End-to-end integration over the critical path on tiny synthetic data.

Catches wiring breaks across factory -> DB -> filter -> portfolio -> backtest that
unit tests miss (the kind of thing the data bug masqueraded as). No LLM, no real data.
"""

import numpy as np

from alpha_gpt.evaluate.backtester import backtest_alpha
from alpha_gpt.factory import db
from alpha_gpt.factory.generators import RandomGenerator
from alpha_gpt.factory.parallel import run_factory_parallel
from alpha_gpt.factory.run import run_factory
from alpha_gpt.portfolio.construct import build_signals, combine


def test_factory_to_portfolio_end_to_end(split, pset, conn):
    gen = RandomGenerator(pset, alphas_per_idea=3, seed=0)
    result = run_factory(gen, pset, split, conn, n_ideas=12)
    assert result["stored"] >= 1
    assert {"stored", "dup", "rejects", "seen"}.issubset(result)

    survivors = db.query(conn, status="ok", min_abs_ic=0.0)
    assert len(survivors) >= 1
    assert all("id" in s and "expression" in s for s in survivors)

    signals = build_signals(survivors, pset, split.panels)
    assert len(signals) >= 1

    weights = {s["id"]: s.get("is_ic", 0.0) for s in survivors}
    comp, sel = combine(signals, method="equal", weights=weights)
    assert comp is not None

    bt = backtest_alpha(comp, split.forward_returns)
    assert np.isfinite(bt.sharpe)
    assert np.isfinite(bt.annual_return)


def test_factory_parallel_random_path(split, pset, conn):
    gen = RandomGenerator(pset, alphas_per_idea=3, seed=1)
    res = run_factory_parallel(gen, pset, split, conn, 8, source="random")
    assert res["stored"] >= 1
    assert db.stats(conn)["total"] >= 1


def test_backtest_portfolio_appends_factor_neutral_diagnostic_row(split, pset, conn, tmp_path):
    """The factor-neutral diagnostic (equal book with characteristic tilts regressed out)
    is reported as its own row but can never be chosen as the headline."""
    from alpha_gpt.pipeline import backtest_portfolio
    run_factory(RandomGenerator(pset, alphas_per_idea=4, seed=5), pset, split, conn, n_ideas=10)
    survivors = db.query(conn, status="ok", min_abs_tstat=0.0)
    port = backtest_portfolio(survivors, pset, split, split, out_dir=str(tmp_path),
                              methods=["equal"], show_progress=False)
    methods = {r["method"] for r in port["rows"]}
    assert "equal_factor_neutral" in methods
    assert port["best"]["method"] != "equal_factor_neutral"


def test_gate_ablation_sweeps_thresholds_and_writes_artifacts(split, pset, conn, tmp_path):
    """The gate ablation backtests the equal-weight book at each |t-stat| bar on the (here
    synthetic) test split: raising the bar keeps monotonically fewer alphas, every row has
    a Sharpe, and the summary JSON is written."""
    from alpha_gpt.pipeline import backtest_gate_ablation
    run_factory(RandomGenerator(pset, alphas_per_idea=5, seed=3), pset, split, conn, n_ideas=12)
    rows = backtest_gate_ablation(conn, pset, split, split, str(tmp_path),
                                  thresholds=[0.0, 1.0, 2.0])
    assert rows, "expected at least one threshold with alphas"
    assert all({"min_tstat", "n_alphas", "sharpe", "ic"} <= set(r) for r in rows)
    counts = [r["n_alphas"] for r in sorted(rows, key=lambda r: r["min_tstat"])]
    assert counts == sorted(counts, reverse=True)  # tighter gate -> fewer alphas
    assert (tmp_path / "gate_ablation.json").exists()
