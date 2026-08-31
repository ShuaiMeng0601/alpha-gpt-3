"""Family-blocked terminal subsampling: the diversity seed for LLM idea generation.

Every idea used to be shown the whole terminal menu, so the model's prior concentrated
on a few famous fields (``cs_roa`` appeared in 49% of surviving alphas) and independent
ideas converged on the same formulas. Handing each idea a small random subset instead
makes decorrelation mechanical rather than statistical: an idea never shown
``market_cap`` cannot rediscover ``cs_rank(market_cap)``.

Draws are blocked by economic family (round-robin over shuffled families, so no family
contributes a second field before every other has contributed one). That stops a draw
from spending its budget on three flavours of the same characteristic — ``roa``,
``cs_roa`` and ``cs_gp_at`` are one bet, not three.

Default size is ~sqrt(n_terminals), which keeps the expected number of shared fields
between two independent draws near one: two draws of size y from x fields overlap in
y^2/x fields on average, so y ~ sqrt(x) is where subsets stop being near-copies of each
other without becoming too thin to express an idea.
"""

from __future__ import annotations

import random

# Economic families, not source families: what a field MEASURES decides whether two
# fields are the same bet. `roa` (WRDS ratio) and `cs_roa` (Compustat characteristic)
# come from different source files but are one profitability bet, so they share a family
# and a draw takes at most one of them per round.
FAMILIES: dict[str, tuple[str, ...]] = {
    "price": ("close", "open", "high", "low", "price", "returns"),
    "volume": ("volume", "dollar_volume", "num_trades"),
    "spread": ("bid", "ask"),
    "size": ("market_cap", "shrout"),
    "value": ("bm", "ptb", "pe_exi", "pe_op_dil", "pcf", "ps", "capei",
              "cs_bm", "cs_ep", "cs_sp"),
    "profitability": ("roe", "roa", "roce", "npm", "gpm", "gprof", "opmbd",
                      "cs_roa", "cs_roe", "cs_gp_at", "cs_gpmargin", "cs_opmargin"),
    "accruals": ("accrual", "cs_accruals"),
    "growth": ("cs_asset_growth", "cs_earnings_growth", "cs_sales_growth", "cs_rd_sale"),
    "leverage": ("de_ratio", "debt_at", "cs_debt_at", "cs_leverage"),
    "solvency": ("cash_ratio", "curr_ratio", "quick_ratio", "cs_current_ratio", "cs_cash_at"),
    "efficiency": ("inv_turn", "at_turn", "rect_turn"),
    "payout": ("divyield",),
}

# Families carrying a genuinely DAILY observation. Fundamentals are quarterly values
# forward-filled to daily, so a subset drawn entirely from them makes every ts_* operator
# degenerate (see OPERATOR_CATALOG's pitfalls) and the debate burns ~20 paid calls on
# formulas the verifier will reject. Every draw is guaranteed one of these.
DAILY_FAMILIES = ("price", "volume", "spread")

_TERMINAL_FAMILY = {t: fam for fam, members in FAMILIES.items() for t in members}

# Terminals added to the panel library after this map was written (e.g. an OSAP ingest)
# land here rather than being dropped: unknown fields still get sampled, they just don't
# block against each other.
OTHER_FAMILY = "other"


def family_of(terminal: str) -> str:
    return _TERMINAL_FAMILY.get(terminal, OTHER_FAMILY)


def default_subset_size(n_terminals: int) -> int:
    """~sqrt(n): the size at which two independent draws share ~1 field on average."""
    return max(3, round(n_terminals ** 0.5))


def _weighted_order(buckets: dict[str, list[str]], families: list[str],
                    rng: random.Random) -> list[str]:
    """Shuffle families with probability proportional to family size.

    Weighting by size is what keeps INDIVIDUAL fields near-uniformly exposed. Under a flat
    shuffle every family gets the same chance to contribute, so the lone member of a
    one-field family (``divyield``) turns up in ~55% of draws while each of profitability's
    twelve members turns up in ~4% — which just swaps the model's concentration for one of
    our own. Weighting by ``len(bucket)`` makes P(family) proportional to its size and
    P(field) roughly flat. (Efraimidis-Spirakis: key = u**(1/w), sorted descending, is
    weighted sampling without replacement in one pass.)
    """
    return sorted(families, key=lambda f: rng.random() ** (1.0 / len(buckets[f])), reverse=True)


def sample_subset(terminals: list[str], size: int | None = None,
                  rng: random.Random | None = None) -> list[str]:
    """Draw a family-spread subset of ``terminals``.

    Returns the full menu (sorted) when it is already at or below ``size`` — subsetting a
    menu that small would only starve the idea.
    """
    rng = rng or random.Random()
    pool = list(dict.fromkeys(terminals))  # de-dup, keep caller order stable
    if size is None:
        size = default_subset_size(len(pool))
    if size >= len(pool):
        return sorted(pool)

    buckets: dict[str, list[str]] = {}
    for name in pool:
        buckets.setdefault(family_of(name), []).append(name)

    families = _weighted_order(buckets, list(buckets), rng)
    # Lead with a daily family so ts_* operators always have something to bite on. WHICH
    # daily family leads (and which field inside it) stays random, so this guarantees a
    # usable field without handing every idea the same one. Size-weighted here too, else
    # the 2-field spread family would contribute as often as the 6-field price family.
    daily = [f for f in families if f in DAILY_FAMILIES]
    if daily:
        lead = _weighted_order(buckets, daily, rng)[0]
        families.remove(lead)
        families.insert(0, lead)

    picked: list[str] = []
    while len(picked) < size and any(buckets.values()):
        for family in families:
            if len(picked) >= size:
                break
            bucket = buckets[family]
            if bucket:
                picked.append(bucket.pop(rng.randrange(len(bucket))))
    return sorted(picked)
