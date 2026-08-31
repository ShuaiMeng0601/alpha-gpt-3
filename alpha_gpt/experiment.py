"""Shared run setup: load panels, split train/val/test, screen GP terminals.

The pipeline (see ``pipeline.py``) and the ``portfolio`` command both start the same
way — load the panel library, split it on the configured dates, pick the terminal
menu, and build the primitive set. Keeping that wiring here means callers don't have
to reassemble it.
"""

from __future__ import annotations

import logging

import pandas as pd

from alpha_gpt.config import Config
from alpha_gpt.data.loader import DataSplit, load_panels, split_data
from alpha_gpt.expr.primitives import (
    DEFAULT_TERMINALS,
    create_primitive_set,
    discover_terminals,
)
from alpha_gpt.expr.terminal_selection import select_terminals

logger = logging.getLogger(__name__)


def load_and_split(config: Config):
    """Load panels, split into in-sample/test, and resolve the GP terminal menu.

    Returns ``(in_sample, test, available_terminals, pset)``. The terminal screen keeps the
    stable core plus the highest-|IC| extras up to ``config.max_terminals``; the resulting
    ``pset`` is shared for parsing seeds and evaluating expressions.
    """
    panels = load_panels(config.data_dir)
    in_sample, test = split_data(panels, config.is_end)
    candidates = [t for t in discover_terminals(config.data_dir) if t in in_sample.panels]
    if config.terminal_screen == "ic_capped":
        available_terminals = select_terminals(
            in_sample.panels, in_sample.forward_returns, candidates, DEFAULT_TERMINALS,
            max_terminals=config.max_terminals, min_breadth=config.min_breadth,
            min_coverage=config.min_coverage, rank_by_ic=True,
        )
    else:  # "breadth_only" — debate-native: keep every field clearing the quality floor
        available_terminals = select_terminals(
            in_sample.panels, in_sample.forward_returns, candidates, DEFAULT_TERMINALS,
            max_terminals=None, min_breadth=config.min_breadth,
            min_coverage=config.min_coverage, rank_by_ic=False,
        )
    pset = create_primitive_set(available_terminals)
    logger.info(
        f"Resolved {len(available_terminals)}/{len(candidates)} candidate terminals "
        f"(screen={config.terminal_screen})"
    )
    return in_sample, test, available_terminals, pset


def compute_vw_benchmark(test: DataSplit) -> pd.Series:
    """Value-weighted (or equal-weight fallback) market equity curve over the test split.

    Recomputed each call (a single matrix op) — there is deliberately no on-disk
    cache, which previously could be silently reused across a different universe or
    date split whenever >10 dates happened to overlap.
    """
    fwd = test.forward_returns
    if "market_cap" in test.panels:
        mcap = test.panels["market_cap"].loc[fwd.index]
        weights = mcap.div(mcap.sum(axis=1), axis=0)
        vw_ret = (fwd * weights).sum(axis=1)
    else:
        vw_ret = fwd.mean(axis=1)  # equal-weight fallback
    return (1 + vw_ret).cumprod()
