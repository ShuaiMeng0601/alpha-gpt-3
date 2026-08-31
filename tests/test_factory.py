"""Tests for the alpha factory: DB dedup/query and the verifier gate."""

import numpy as np
import pandas as pd
import pytest

from alpha_gpt.factory import db
from alpha_gpt.factory.verifier import verify_expression
from alpha_gpt.expr.primitives import create_primitive_set


def _rec(h, is_ic, status="ok"):
    return {"hash": h, "expression": f"expr_{h}", "status": status, "is_ic": is_ic,
            "coverage": 0.9, "source": "random"}


def test_db_dedup_and_query():
    conn = db.connect(":memory:")
    assert db.insert_alpha(conn, _rec("aaa", 0.05)) is True
    assert db.insert_alpha(conn, _rec("aaa", 0.99)) is False  # duplicate hash ignored
    db.insert_alpha(conn, _rec("bbb", -0.04))                 # negative IC kept
    db.insert_alpha(conn, _rec("ccc", 0.001))
    # |IC| filter keeps the strong positive and strong negative, drops the weak one
    rows = db.query(conn, min_abs_ic=0.02)
    hashes = {r["hash"] for r in rows}
    assert hashes == {"aaa", "bbb"}
    assert db.stats(conn)["total"] == 3


def _toy_panels(n_days=40, n_stocks=25):
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    cols = list(range(1, n_stocks + 1))
    rng = np.random.RandomState(0)
    return {
        "close": pd.DataFrame(rng.randn(n_days, n_stocks).cumsum(0) + 50, index=idx, columns=cols),
        "volume": pd.DataFrame(rng.rand(n_days, n_stocks) * 1e6, index=idx, columns=cols),
    }


def test_verifier_accepts_valid_and_rejects_garbage():
    pset = create_primitive_set(["close", "volume"])
    panels = _toy_panels()
    ok = verify_expression("cs_rank(ts_delta(close, 5))", pset, panels,
                           min_breadth=5, min_active_days_frac=0.1)
    assert ok.ok and ok.reason == "ok" and ok.signal is not None

    bad = verify_expression("nonexistent_op(close)", pset, panels)
    assert not bad.ok and bad.reason == "parse_error"

    empty = verify_expression("", pset, panels)
    assert not empty.ok


def test_db_query_order_by_allowlist():
    conn = db.connect(":memory:")
    db.insert_alpha(conn, _rec("aaa", 0.05))
    db.query(conn, order_by="abs(is_ic)")    # allowed
    db.query(conn, order_by="is_sharpe")     # allowed
    with pytest.raises(ValueError):          # REGRESSION: reject SQL-injection / junk
        db.query(conn, order_by="is_ic; DROP TABLE alphas")


def test_connect_migrates_pre_refactor_db(tmp_path):
    """REGRESSION: opening a PRE-refactor DB (old train_ic/val_ic schema, no is_* columns)
    must not crash on CREATE INDEX ... ON alphas(is_ic). connect() migrates the columns
    (old rows -> NULL is_tstat, excluded by the gate) instead of dying with an opaque error."""
    import sqlite3
    path = str(tmp_path / "old.db")
    old = sqlite3.connect(path)
    old.executescript("CREATE TABLE alphas (id INTEGER PRIMARY KEY, hash TEXT UNIQUE, "
                      "expression TEXT, status TEXT, train_ic REAL, val_ic REAL);")
    old.execute("INSERT INTO alphas (hash, expression, status, train_ic, val_ic) "
                "VALUES ('h1','e1','ok',0.05,0.05)")
    old.commit(); old.close()

    conn = db.connect(path)  # must not raise
    cols = {r[1] for r in conn.execute("PRAGMA table_info(alphas)").fetchall()}
    assert {"is_ic", "is_tstat"} <= cols
    # the stale row has NULL is_tstat, so the t-stat gate drops it (no contamination)
    assert db.query(conn, min_abs_tstat=2.0) == []
    assert db.stats(conn)["total"] == 1


def test_db_query_tstat_gate():
    conn = db.connect(":memory:")
    db.insert_alpha(conn, {"hash": "sig", "expression": "e1", "status": "ok",
                           "is_ic": 0.03, "is_tstat": 3.5})
    db.insert_alpha(conn, {"hash": "noise", "expression": "e2", "status": "ok",
                           "is_ic": 0.03, "is_tstat": 1.2})   # same IC, insignificant
    db.insert_alpha(conn, {"hash": "shortneg", "expression": "e3", "status": "ok",
                           "is_ic": -0.03, "is_tstat": -3.0})  # strong negative kept (|t|)
    rows = db.query(conn, min_abs_tstat=2.0)
    assert {r["hash"] for r in rows} == {"sig", "shortneg"}


def test_db_query_abs_value_filters_keep_strong_negative():
    conn = db.connect(":memory:")
    db.insert_alpha(conn, _rec("pos", 0.05))
    db.insert_alpha(conn, _rec("neg", -0.06))   # strong negative = sign-flip away
    db.insert_alpha(conn, _rec("weak", 0.001))
    rows = db.query(conn, min_abs_ic=0.02)
    assert {r["hash"] for r in rows} == {"pos", "neg"}


def test_verifier_reports_structure_and_rejects_too_deep():
    pset = create_primitive_set(["close", "volume"])
    panels = _toy_panels()
    ok = verify_expression("cs_rank(ts_delta(close, 5))", pset, panels,
                           min_breadth=5, min_active_days_frac=0.1)
    assert ok.ok and ok.n_terminals >= 1 and ok.depth >= 1 and 0.0 <= ok.coverage <= 1.0
    # an absurdly low depth cap rejects a nested expression as too_deep
    deep = verify_expression("cs_rank(ts_delta(close, 5))", pset, panels, max_depth=1)
    assert not deep.ok and deep.reason == "too_deep"
