"""Tests for the verifier's static degeneracy check, the return-blind diagnostics line,
and reject persistence (rejects stored with status=reason; readers filter status='ok').

Motivated by run_20260623_162032: 326/1475 formulas were all_nan, dominated by
self-cancelling expressions (sub(x,x), safe_div(x,x), sign(abs_val(x))) collapsing to a
constant and turning all-NaN through cs_zscore / ts_corr / safe_div / log_abs.
"""

from alpha_gpt.factory import db
from alpha_gpt.factory.generators import IdeaBatch
from alpha_gpt.factory.run import run_factory
from alpha_gpt.factory.verifier import degenerate_reason, diagnose_expression, verify_expression
from alpha_gpt.expr.seed_injector import parse_expression


# ---------------------------------------------------------------------------
# degenerate_reason (static, no data)
# ---------------------------------------------------------------------------

def _tree(expr, pset):
    tree = parse_expression(expr, pset)
    assert tree is not None, f"test expression failed to parse: {expr}"
    return tree


def test_degenerate_sub_of_identical_args(pset):
    assert "sub" in degenerate_reason(_tree("sub(close, close)", pset))


def test_degenerate_safe_div_of_identical_args(pset):
    assert "safe_div" in degenerate_reason(_tree("cs_rank(safe_div(volume, volume))", pset))


def test_degenerate_ts_corr_of_identical_args(pset):
    assert "ts_corr" in degenerate_reason(_tree("ts_corr(close, close, 20)", pset))


def test_degenerate_identical_nontrivial_subtrees(pset):
    expr = "sub(cs_rank(ts_delta(close, 5)), cs_rank(ts_delta(close, 5)))"
    assert "sub" in degenerate_reason(_tree(expr, pset))


def test_degenerate_sign_abs_chain(pset):
    assert "constant" in degenerate_reason(_tree("cs_zscore(sign(abs_val(close)))", pset))


def test_clean_expressions_not_degenerate(pset):
    for expr in [
        "cs_rank(ts_delta(close, 5))",
        "sub(cs_rank(close), cs_rank(volume))",                    # different args: fine
        "sub(cs_rank(ts_delta(close, 5)), cs_rank(ts_delta(close, 10)))",  # different windows
        "safe_div(ts_returns(close, 20), ts_std(returns, 20))",
        "sign(ts_delta(close, 5))",                                # sign alone is fine
    ]:
        assert degenerate_reason(_tree(expr, pset)) == "", expr


def test_verify_rejects_degenerate_before_eval(pset, panels):
    vr = verify_expression("sub(close, close)", pset, panels)
    assert not vr.ok and vr.reason == "degenerate"
    assert vr.signal is None  # rejected statically, no evaluation happened
    assert vr.normalized      # rejects carry the canonical form for dedup/persistence


def test_arity_incomplete_expressions_are_parse_errors_not_crashes(pset, panels):
    """REGRESSION: DEAP's from_string mis-parses wrong-arity expressions into truncated
    trees (e.g. ts_corr(close, 10) -> binary ts_corr_10 with ONE child), which crashed
    degenerate_reason with IndexError and killed the whole factory run. They must come
    back as ordinary parse_error rejects."""
    for expr in ["ts_corr(close, 10)", "sub(close)", "safe_div(close)"]:
        vr = verify_expression(expr, pset, panels)   # must not raise
        assert not vr.ok and vr.reason == "parse_error", expr
        line = diagnose_expression(expr, pset, panels)
        assert line.startswith("REJECTED (parse_error)"), expr


# ---------------------------------------------------------------------------
# diagnose_expression (return-blind feedback line for the debate)
# ---------------------------------------------------------------------------

def test_diagnose_ok_reports_measured_properties(pset, panels):
    line = diagnose_expression("cs_rank(ts_delta(close, 5))", pset, panels)
    assert line.startswith("OK")
    for token in ["coverage", "stocks/day", "rank turnover", "distinct values", "dead days"]:
        assert token in line
    assert "IC" not in line  # return-blind: no performance metric ever appears


def test_diagnose_degenerate_names_the_cause(pset, panels):
    line = diagnose_expression("cs_zscore(sub(close, close))", pset, panels)
    assert line.startswith("REJECTED (degenerate)")
    assert "sub" in line


def test_diagnose_parse_error_is_actionable(pset, panels):
    line = diagnose_expression("not_an_op(close)", pset, panels)
    assert line.startswith("REJECTED (parse_error)")


def test_diagnose_too_deep_mentions_cap(pset, panels):
    line = diagnose_expression("cs_rank(ts_delta(close, 5))", pset, panels, max_depth=1)
    assert line.startswith("REJECTED (too_deep)")


# ---------------------------------------------------------------------------
# Reject persistence (serial factory path)
# ---------------------------------------------------------------------------

class _ScriptedGen:
    """No-API generator emitting one degenerate and one healthy expression."""
    total_prompt_tokens = 0
    total_completion_tokens = 0
    n_calls = 0
    model = None

    def generate(self, n_ideas):
        yield IdeaBatch(idea="test-idea", expressions=[
            "sub(close, close)",              # degenerate -> rejected + persisted
            "cs_rank(ts_delta(close, 5))",    # healthy -> stored ok
        ], source="test")


def test_rejects_are_persisted_with_reason(pset, split, conn):
    summary = run_factory(_ScriptedGen(), pset, split, conn, n_ideas=1)
    assert summary["rejects"] == {"degenerate": 1}
    assert summary["stored"] == 1

    rows = {r["status"]: r for r in conn.execute("SELECT * FROM alphas").fetchall()}
    assert set(rows) == {"degenerate", "ok"}
    reject = rows["degenerate"]
    assert reject["raw_expression"] == "sub(close, close)"
    assert reject["is_ic"] is None  # no metrics for rejects

    # Every downstream reader filters status='ok' -> rejects never leak into selection.
    ok_rows = db.query(conn, status="ok")
    assert len(ok_rows) == 1
    assert ok_rows[0]["raw_expression"] == "cs_rank(ts_delta(close, 5))"


def test_reject_row_never_shadows_a_later_ok_row(conn):
    """REGRESSION: reject hashes live in a 'reject:' namespace. With a shared namespace,
    a stale reject row in a reused alphas.db would silently swallow a later ok row for
    the same expression (hash UNIQUE + INSERT OR IGNORE -> counted as dup, never stored)."""
    expr = "cs_rank(ts_delta(close, 5))"
    reject = {"hash": db.expr_hash("reject:" + expr), "expression": expr,
              "raw_expression": expr, "status": "sparse"}
    assert db.insert_alpha(conn, reject)
    ok = {"hash": db.expr_hash(expr), "expression": expr, "raw_expression": expr,
          "status": "ok", "is_ic": 0.01, "is_tstat": 3.0}
    assert db.insert_alpha(conn, ok)  # must NOT be swallowed by the stale reject
    ok_rows = db.query(conn, status="ok")
    assert len(ok_rows) == 1 and ok_rows[0]["expression"] == expr
