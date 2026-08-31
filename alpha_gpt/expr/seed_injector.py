"""Parse LLM-generated alpha expression strings into DEAP PrimitiveTree objects.

Handles the bridge between natural-language-style expressions like
  cs_rank(ts_delta(close, 5))
and DEAP's curried form:
  cs_rank(ts_delta_5(close))
"""

import re
import logging

from deap import gp

from alpha_gpt.expr.alpha_ops import ALL_OPS

logger = logging.getLogger(__name__)

# (base_op, window) -> curried operator name, e.g. ("ts_mean", 20) -> "ts_mean_20".
# Derived from the operator registry so it can never drift from the implementations.
CURRIED_MAP = {
    (base, int(window)): name
    for name in ALL_OPS
    for base, _, window in [name.rpartition("_")]
    if base.startswith("ts_") and window.isdigit()
}

# Default window when the LLM omits it (a deliberate choice, not derivable).
DEFAULT_WINDOWS = {
    "ts_mean": 20, "ts_std": 20, "ts_min": 20, "ts_max": 20,
    "ts_delta": 5, "ts_rank": 20, "ts_returns": 1, "ts_corr": 20,
}


def normalize_expression(expr: str) -> str:
    """Convert LLM-style expression to DEAP-compatible curried form.

    Examples:
        "cs_rank(ts_delta(close, 5))" -> "cs_rank(ts_delta_5(close))"
        "ts_mean(volume, 20)"        -> "ts_mean_20(volume)"
        "rank(returns)"              -> "cs_rank(returns)"
    """
    # Common LLM aliases — use \w lookbehind to avoid matching inside
    # prefixed names (e.g. ts_rank, cs_rank, safe_div, etc.)
    expr = re.sub(r'(?<!\w)rank\(', 'cs_rank(', expr)
    expr = re.sub(r'(?<!\w)zscore\(', 'cs_zscore(', expr)
    expr = re.sub(r'(?<!\w)delta\(', 'ts_delta(', expr)
    expr = re.sub(r'(?<!\w)std\(', 'ts_std(', expr)
    expr = re.sub(r'(?<!\w)mean\(', 'ts_mean(', expr)
    expr = re.sub(r'(?<!\w)abs\(', 'abs_val(', expr)
    expr = re.sub(r'(?<!\w)log\(', 'log_abs(', expr)
    expr = re.sub(r'(?<!\w)div\(', 'safe_div(', expr)
    expr = expr.replace("correlation(", "ts_corr_20(")
    expr = re.sub(r'(?<!\w)corr\(', 'ts_corr_20(', expr)

    # Handle ts_op(args..., window) -> ts_op_window(args...) using balanced-paren aware splitting
    def _curry_once(text: str) -> str:
        """Find and curry one ts_op(..., window) call, innermost first."""
        # Find all ts_op( positions
        for m in re.finditer(r'(ts_\w+)\(', text):
            op_name = m.group(1)
            # Already curried (has _\d+ suffix)?
            if re.match(r'ts_\w+_\d+$', op_name):
                continue
            start = m.end()  # position right after the '('
            # Walk forward with a paren counter to find the matching ')'
            depth = 1
            i = start
            while i < len(text) and depth > 0:
                if text[i] == '(':
                    depth += 1
                elif text[i] == ')':
                    depth -= 1
                i += 1
            if depth != 0:
                continue
            inner = text[start:i - 1]  # content between the outer parens
            # Split on the LAST comma at depth 0 to separate (field_args, window)
            last_comma = -1
            d = 0
            for j, ch in enumerate(inner):
                if ch == '(':
                    d += 1
                elif ch == ')':
                    d -= 1
                elif ch == ',' and d == 0:
                    last_comma = j
            if last_comma == -1:
                continue
            field_part = inner[:last_comma].strip()
            window_part = inner[last_comma + 1:].strip()
            if not window_part.isdigit():
                continue
            window = int(window_part)
            curried = CURRIED_MAP.get((op_name, window))
            if not curried:
                default_w = DEFAULT_WINDOWS.get(op_name)
                if default_w:
                    curried = CURRIED_MAP.get((op_name, default_w))
            if curried:
                replacement = f"{curried}({field_part})"
                return text[:m.start()] + replacement + text[i:]
        return text

    # Apply repeatedly (handles nested expressions from inside out)
    for _ in range(10):
        new_expr = _curry_once(expr)
        if new_expr == expr:
            break
        expr = new_expr

    # Handle ts_op(field) without window -> add default window
    for op_name, default_w in DEFAULT_WINDOWS.items():
        # Match op_name( but not op_name_\d+( (already curried)
        pattern_no_window = rf'(?<!\w){op_name}\((?!\d)'
        curried = CURRIED_MAP.get((op_name, default_w), f"{op_name}_{default_w}")
        expr = re.sub(pattern_no_window, f"{curried}(", expr)

    return expr


def _is_complete_prefix_tree(tree: gp.PrimitiveTree) -> bool:
    """True iff the node list forms exactly one complete prefix-order tree.

    DEAP's ``from_string`` does not validate arity: a wrong-arity expression like
    ``ts_corr(close, 10)`` (normalized to the binary ``ts_corr_10`` with ONE child)
    parses into a truncated node list whose traversal then crashes any caller that
    walks subtrees. An exact prefix walk (not a sum-of-arities shortcut, which is
    fooled by trailing junk after an early-completing root) catches both truncated
    and over-full node lists."""
    need = 1
    for i, node in enumerate(tree):
        need += node.arity - 1
        if need == 0:
            return i == len(tree) - 1  # complete exactly at the last node
    return False  # ran out of nodes while children were still expected


def parse_expression(expr_str: str, pset: gp.PrimitiveSet) -> gp.PrimitiveTree | None:
    """Parse an expression string into a DEAP PrimitiveTree.

    Args:
        expr_str: Expression string (LLM-generated or normalized).
        pset: The PrimitiveSet to use for parsing.

    Returns:
        PrimitiveTree if parsing succeeds, None if it fails (including arity-incomplete
        expressions that DEAP would happily mis-parse into a truncated tree).
    """
    try:
        normalized = normalize_expression(expr_str)
        tree = gp.PrimitiveTree.from_string(normalized, pset)
        if len(tree) == 0:           # malformed input can yield an empty tree, not an error
            return None
        if not _is_complete_prefix_tree(tree):
            logger.warning(f"Arity-incomplete expression rejected at parse: {expr_str!r}")
            return None
        return tree
    except Exception as e:
        logger.warning(f"Failed to parse expression: {expr_str!r} -> {e}")
        return None


def inject_seeds(
    seed_expressions: list[str],
    pset: gp.PrimitiveSet,
) -> list[gp.PrimitiveTree]:
    """Parse a list of LLM-generated expressions into DEAP trees.

    Expressions that fail to parse are logged and skipped.

    Returns:
        List of successfully parsed PrimitiveTree objects.
    """
    trees = []
    for expr in seed_expressions:
        tree = parse_expression(expr, pset)
        if tree is not None:
            trees.append(tree)
            logger.info(f"Parsed seed: {expr!r} -> {tree}")
        else:
            logger.warning(f"Skipped unparseable seed: {expr!r}")

    logger.info(f"Injected {len(trees)}/{len(seed_expressions)} seed expressions")
    return trees
