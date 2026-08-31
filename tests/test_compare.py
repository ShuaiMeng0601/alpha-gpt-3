"""Tests for cross-source aggregation (alpha_gpt.compare)."""

import json
import os

from alpha_gpt.compare import run_compare


def _write_run(outputs_dir, name, source, *, n_survivors, equal_sharpe, equal_ic, cost=0.0):
    run_dir = os.path.join(outputs_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    payload = {
        "params": {"source": source},
        "n_survivors": n_survivors,
        "factory": {"est_cost_usd": cost},
        "portfolio": {
            "rows": [
                {"method": "best_single_alpha", "sharpe": 0.1, "ic": 0.005, "annual_return": 0.02},
                {"method": "equal", "sharpe": equal_sharpe, "ic": equal_ic,
                 "annual_return": equal_sharpe * 0.1},
            ],
        },
    }
    with open(os.path.join(run_dir, "pipeline_summary.json"), "w") as f:
        json.dump(payload, f)


def test_run_compare_aggregates_sources_and_writes_artifacts(tmp_path):
    out = str(tmp_path)
    _write_run(out, "run_0_debate", "debate", n_survivors=400, equal_sharpe=1.0, equal_ic=0.02)
    _write_run(out, "run_1_debate", "debate", n_survivors=440, equal_sharpe=0.9, equal_ic=0.022)
    _write_run(out, "run_0_random", "random", n_survivors=10, equal_sharpe=0.1, equal_ic=0.001)

    run_compare(outputs_dir=out)

    comparison_path = os.path.join(out, "comparison", "comparison.json")
    assert os.path.exists(comparison_path)
    assert os.path.exists(os.path.join(out, "comparison", "summary_table.md"))

    rows = {r["source"]: r for r in json.load(open(comparison_path))}
    assert set(rows) == {"debate", "random"}
    assert rows["debate"]["n_runs"] == 2
    assert abs(rows["debate"]["sharpe_mean"] - 0.95) < 1e-9  # mean of 1.0 and 0.9
    assert rows["debate"]["n_survivors_mean"] == 420
    assert rows["random"]["n_runs"] == 1


def test_run_compare_no_results_is_noop(tmp_path, capsys):
    run_compare(outputs_dir=str(tmp_path))
    assert "No pipeline_summary.json" in capsys.readouterr().out
    assert not os.path.exists(os.path.join(str(tmp_path), "comparison"))
