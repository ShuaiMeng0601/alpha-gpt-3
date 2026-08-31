"""Guard: the operator catalog shown to the model must match the code registry.

`OPERATOR_CATALOG` (debate/prompts.py) is hand-written prose shown to every debate agent
and to the LLM idea generator. The operators it advertises must stay in sync with the real
registry (`expr/alpha_ops.py`): if they drift, the model proposes operators that don't exist
(wasted calls / unparseable formulas) or never uses ones that do. These tests fail on drift.

(Terminals are NOT checked here: the authoritative terminal set is injected into the prompt
at runtime from the resolved panels, so it can't drift from the code the way a hard-coded
catalog can.)
"""

import re

from alpha_gpt.debate.prompts import OPERATOR_CATALOG
from alpha_gpt.expr.alpha_ops import ALL_OPS


def _registry_base_names() -> set[str]:
    """Registry op names in the form the catalog advertises them.

    Time-series ops are curried per window in the registry (``ts_mean_20``) but shown in the
    catalog as a single parameterized entry (``ts_mean(x, window)``), so collapse them to the
    base name. Everything else (``cs_rank``, ``add``, ``safe_div``, ...) is shown as-is.
    """
    bases = set()
    for name in ALL_OPS:
        base, _, window = name.rpartition("_")
        bases.add(base if base.startswith("ts_") and window.isdigit() else name)
    return bases


def test_every_registry_operator_is_advertised_in_the_catalog():
    """Forward drift: add an operator to the registry, forget the prompt -> caught here."""
    missing = sorted(b for b in _registry_base_names() if b not in OPERATOR_CATALOG)
    assert not missing, f"operators in the registry but absent from OPERATOR_CATALOG: {missing}"


def test_catalog_advertises_no_operators_absent_from_the_registry():
    """Reverse drift: the catalog can't promise the model an operator the code lacks."""
    # Operators appear in the catalog as `name(` calls (ts_mean(, cs_rank(, add(, ...).
    advertised = set(re.findall(r"\b([a-z][a-z0-9_]*)\(", OPERATOR_CATALOG))
    known = _registry_base_names() | set(ALL_OPS)  # accept base or curried form
    unknown = sorted(advertised - known)
    assert not unknown, f"OPERATOR_CATALOG advertises operators not in the registry: {unknown}"


def test_curried_time_series_windows_are_advertised():
    """Each concrete ts window in the registry must be listed so the model uses valid ones."""
    for name in ALL_OPS:
        base, _, window = name.rpartition("_")
        if base.startswith("ts_") and window.isdigit():
            # The catalog lists windows per family, e.g. "ts_mean(x, window) ... (windows: 5, 10, 20, 60)".
            line = next((ln for ln in OPERATOR_CATALOG.splitlines() if base + "(" in ln), "")
            assert re.search(rf"\b{window}\b", line), f"window {window} for {base} missing from its catalog line"
