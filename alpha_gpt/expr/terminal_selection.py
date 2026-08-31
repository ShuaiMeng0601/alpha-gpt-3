"""Rank candidate terminals by univariate IC and cap the set fed to GP/LLM.

With ~200+ candidate panels, feeding all of them to DEAP explodes the search
space. We keep a stable core (DEFAULT_TERMINALS) plus the highest |IC| extras up
to `max_terminals`. The full panel library stays on disk; this screen selects the
most predictive subset per run. IC here is the canonical vectorized Spearman
(evaluate/metrics.compute_ic) — the same metric used everywhere else.
"""

import numpy as np
import pandas as pd

from alpha_gpt.evaluate.metrics import compute_ic


def fast_ic(panel: pd.DataFrame, forward_returns: pd.DataFrame) -> float:
    """Mean daily cross-sectional Spearman IC (delegates to the canonical compute_ic)."""
    return float(compute_ic(panel, forward_returns).mean())


def _passes_quality(panel: pd.DataFrame, min_breadth: int, min_coverage: float) -> bool:
    """Data-quality floor: a field must be broad (enough stocks/day) and dense enough.

    An extremely sparse field silently shrinks the per-day universe every alpha that uses
    it can trade, so it is unusable regardless of generator (debate, GP, or random).
    """
    if min_breadth and float(panel.notna().sum(axis=1).median()) < min_breadth:
        return False
    if min_coverage and float(panel.notna().to_numpy().mean()) < min_coverage:
        return False
    return True


def select_terminals(
    panels: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    candidates: list[str],
    core: list[str],
    max_terminals: int | None = None,
    min_breadth: int = 0,
    min_coverage: float = 0.0,
    rank_by_ic: bool = True,
) -> list[str]:
    """Core terminals (always kept) + extras that clear the data-quality floor.

    Two regimes (selected via config.terminal_screen):
      * debate-native (``rank_by_ic=False`` and ``max_terminals=None``): keep EVERY extra
        that passes the breadth/coverage floor — no univariate-IC pre-screen, no cap. The
        debate reasons about which features matter, so we do not narrow the menu with the
        heuristic it is meant to beat, nor starve it with a cap.
      * legacy GP (``rank_by_ic=True`` with a ``max_terminals`` cap): among the qualifying
        extras, keep the highest-|IC| ones up to the cap, to bound the GP search space.
    """
    core_present = [c for c in core if c in candidates]
    extras = [c for c in candidates if c not in core_present]

    # Data-quality floor (always applied).
    qualified = []
    for name in extras:
        try:
            panel = panels[name].reindex_like(forward_returns)
        except Exception:
            continue
        if _passes_quality(panel, min_breadth, min_coverage):
            qualified.append(name)

    # Debate-native: no IC pre-screen -> keep all qualifying extras (cap is ignored when
    # rank_by_ic is False, since a cap only makes sense alongside an IC ranking to cut by).
    if not rank_by_ic:
        return core_present + qualified

    # Legacy: rank qualifying extras by |IC|, cap to max_terminals.
    scored = []
    for name in qualified:
        try:
            ic = fast_ic(panels[name].reindex_like(forward_returns), forward_returns)
            if not np.isnan(ic):
                scored.append((abs(ic), name))
        except Exception:
            continue
    scored.sort(reverse=True)
    cap = len(scored) if max_terminals is None else max_terminals
    n_extra = max(0, cap - len(core_present))
    selected = [name for _, name in scored[:n_extra]]
    return core_present + selected
