"""DEAP PrimitiveSet definition for alpha expression trees.

Uses a single type (panel DataFrame) for all inputs/outputs.
Operators are curried with fixed window sizes so DEAP doesn't need
to evolve integer constants.
"""

import os

from deap import gp

from alpha_gpt.expr.alpha_ops import BINARY_OPS, UNARY_OPS


def create_primitive_set(terminal_names: list[str]) -> gp.PrimitiveSet:
    """Build a DEAP PrimitiveSet from the operator registry + the given terminals.

    Operators come straight from ``expr.alpha_ops`` (the single source of
    truth); terminals are added as named constants whose actual data is injected at
    evaluation time (see ``expr.engine.eval_expr``).
    """
    pset = gp.PrimitiveSet("alpha", arity=0)
    for op in UNARY_OPS:
        pset.addPrimitive(op, 1)
    for op in BINARY_OPS:
        pset.addPrimitive(op, 2)
    for name in terminal_names:
        pset.addTerminal(name, name)
    return pset


# Stable core terminals (CRSP + key ratios), always kept by the terminal screen.
DEFAULT_TERMINALS = [
    "close", "open", "high", "low", "volume", "returns",
    "shrout", "market_cap",
    # Compustat ratios
    "bm", "roe", "roa", "pe_op_dil", "ptb", "npm", "gpm",
    "debt_at", "curr_ratio", "accrual",
]

# Panels that are never terminals (targets / benchmarks / categorical aux).
NON_TERMINAL_PANELS = {"forward_returns", "vw_market_return", "sector"}


def discover_terminals(data_dir: str, exclude: set[str] | None = None) -> list[str]:
    """List candidate terminal field names from the panels directory.

    Returns every <field>.parquet basename except non-terminal panels and any
    underscore-prefixed bookkeeping files (manifests, coverage reports).
    """
    skip = set(NON_TERMINAL_PANELS) | (exclude or set())
    names = []
    for f in sorted(os.listdir(data_dir)):
        if f.endswith(".parquet") and not f.startswith("_"):
            name = f[: -len(".parquet")]
            if name not in skip:
                names.append(name)
    return names
