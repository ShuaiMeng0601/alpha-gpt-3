"""SQLite store for generated alphas.

Stores EXPRESSIONS + scalar metrics only (never signals — 1M signals would be
terabytes). Signals are re-evaluated on demand for the filtered survivors during
portfolio construction. SQLite handles millions of rows comfortably and is
zero-config. Selection metrics are train/val only; test is held out for the
final portfolio backtest.
"""

import hashlib
import os
import sqlite3

# Whitelist of permissible ORDER BY expressions (order_by is interpolated into SQL,
# so it must never be caller-controlled free text). Metrics are IN-SAMPLE (`is_*`).
ALLOWED_ORDER_BY = {
    "abs(is_ic)", "abs(is_icir)", "abs(is_tstat)", "abs(is_sharpe)",
    "is_ic", "is_icir", "is_tstat", "is_sharpe",
    "turnover", "coverage", "created_at", "id",
}

# Kept separate from the indexes: `CREATE TABLE IF NOT EXISTS` is a no-op on an existing
# (possibly pre-refactor) table, so the is_* columns must be added by migration in connect()
# BEFORE the indexes reference them — otherwise CREATE INDEX ... ON alphas(is_ic) crashes.
CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS alphas (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    hash            TEXT UNIQUE NOT NULL,
    expression      TEXT NOT NULL,
    raw_expression  TEXT,
    idea            TEXT,
    hypothesis      TEXT,
    source          TEXT,
    model           TEXT,
    is_ic           REAL,
    is_icir         REAL,
    is_tstat        REAL,
    is_sharpe       REAL,
    is_annual_return REAL,
    is_max_drawdown  REAL,
    turnover        REAL,
    coverage        REAL,
    n_terminals     INTEGER,
    depth           INTEGER,
    offered_terminals     TEXT,
    confidence            INTEGER,
    gen_prompt_tokens     INTEGER,
    gen_completion_tokens INTEGER,
    status          TEXT NOT NULL,
    created_at      TEXT
);
"""

INDEXES = """
CREATE INDEX IF NOT EXISTS idx_status   ON alphas(status);
CREATE INDEX IF NOT EXISTS idx_is_ic    ON alphas(is_ic);
CREATE INDEX IF NOT EXISTS idx_is_tstat ON alphas(is_tstat);
"""

FIELDS = [
    "hash", "expression", "raw_expression", "idea", "hypothesis", "source", "model",
    "is_ic", "is_icir", "is_tstat", "is_sharpe", "is_annual_return", "is_max_drawdown",
    "turnover", "coverage", "n_terminals", "depth",
    "offered_terminals", "confidence",
    "gen_prompt_tokens", "gen_completion_tokens", "status", "created_at",
]


def expr_hash(normalized_expression: str) -> str:
    """Stable short hash of a normalized expression (for dedup)."""
    return hashlib.sha1(normalized_expression.encode("utf-8")).hexdigest()[:16]


def connect(path: str = "data/alphas.db") -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(CREATE_TABLE)
    # Migrate older DBs (incl. pre-refactor ones with train_ic/val_ic) so the is_* columns
    # exist before the indexes reference them. Old rows get NULL is_* and are correctly
    # excluded by the abs(is_tstat) >= ? gate (abs(NULL) >= x is NULL, i.e. not selected).
    existing = {r[1] for r in conn.execute("PRAGMA table_info(alphas)").fetchall()}
    for col in ("is_ic", "is_icir", "is_tstat", "is_sharpe", "is_annual_return", "is_max_drawdown"):
        if col not in existing:
            conn.execute(f"ALTER TABLE alphas ADD COLUMN {col} REAL")
    for col in ("gen_prompt_tokens", "gen_completion_tokens", "confidence"):
        if col not in existing:
            conn.execute(f"ALTER TABLE alphas ADD COLUMN {col} INTEGER")
    if "offered_terminals" not in existing:
        conn.execute("ALTER TABLE alphas ADD COLUMN offered_terminals TEXT")
    conn.executescript(INDEXES)
    conn.commit()
    return conn


def insert_alpha(conn: sqlite3.Connection, rec: dict) -> bool:
    """Insert one alpha record. Returns True if new, False if duplicate hash.

    Does NOT commit — the caller commits once per batch. A commit per row fsyncs on
    every insert, which is catastrophic at the factory's ~1M-alpha scale.
    """
    cols = ",".join(FIELDS)
    placeholders = ",".join("?" * len(FIELDS))
    vals = [rec.get(f) for f in FIELDS]
    cur = conn.execute(
        f"INSERT OR IGNORE INTO alphas ({cols}) VALUES ({placeholders})", vals
    )
    return cur.rowcount > 0


def query(
    conn: sqlite3.Connection,
    status: str = "ok",
    min_abs_ic: float | None = None,
    min_abs_icir: float | None = None,
    min_abs_tstat: float | None = None,
    min_abs_sharpe: float | None = None,
    max_turnover: float | None = None,
    min_coverage: float | None = None,
    order_by: str = "abs(is_tstat)",
    desc: bool = True,
    limit: int | None = None,
) -> list[dict]:
    """Filter stored alphas on IN-SAMPLE metrics. IC/ICIR/t-stat/Sharpe filters use
    ABSOLUTE value, since a strongly negative-IC alpha is just a sign flip away from a
    strong signal.

    ``min_abs_tstat`` is the primary overfitting gate: the t-stat of the mean in-sample IC
    (= ICIR x sqrt(#days)) must clear the bar, so a signal that only *looks* good by luck on
    a short window is filtered out. It replaces the old train/val sign-consistency check."""
    where, args = ["status = ?"], [status]
    if min_abs_ic is not None:
        where.append("abs(is_ic) >= ?"); args.append(min_abs_ic)
    if min_abs_icir is not None:
        where.append("abs(is_icir) >= ?"); args.append(min_abs_icir)
    if min_abs_tstat is not None:
        where.append("abs(is_tstat) >= ?"); args.append(min_abs_tstat)
    if min_abs_sharpe is not None:
        where.append("abs(is_sharpe) >= ?"); args.append(min_abs_sharpe)
    if max_turnover is not None:
        where.append("turnover <= ?"); args.append(max_turnover)
    if min_coverage is not None:
        where.append("coverage >= ?"); args.append(min_coverage)
    if order_by not in ALLOWED_ORDER_BY:
        raise ValueError(f"order_by must be one of {sorted(ALLOWED_ORDER_BY)}, got {order_by!r}")
    sql = f"SELECT * FROM alphas WHERE {' AND '.join(where)}"
    sql += f" ORDER BY {order_by} {'DESC' if desc else 'ASC'}"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return [dict(r) for r in conn.execute(sql, args).fetchall()]


def stats(conn: sqlite3.Connection) -> dict:
    total = conn.execute("SELECT COUNT(*) FROM alphas").fetchone()[0]
    by_status = {
        r[0]: r[1]
        for r in conn.execute("SELECT status, COUNT(*) FROM alphas GROUP BY status").fetchall()
    }
    return {"total": total, "by_status": by_status}
