"""Tests for the family-blocked terminal sampler.

This is the diversity seed for every paid idea, so the properties that matter are the ones
that make decorrelation MECHANICAL: a draw spreads across economic families, two draws
rarely overlap, and no draw is left without a daily field to apply ts_* operators to.
"""

import random

import pytest

from alpha_gpt.expr.terminal_subset import (
    DAILY_FAMILIES,
    FAMILIES,
    default_subset_size,
    family_of,
    sample_subset,
)

ALL_TERMINALS = sorted({t for members in FAMILIES.values() for t in members})


def test_default_size_is_sqrt_of_menu():
    """y ~ sqrt(x) is where two independent draws share ~1 field on average."""
    assert default_subset_size(54) == 7
    assert default_subset_size(4) == 3  # floor: a 2-field idea has nothing to work with


def test_draw_has_requested_size_and_no_repeats():
    rng = random.Random(0)
    for _ in range(50):
        picked = sample_subset(ALL_TERMINALS, size=7, rng=rng)
        assert len(picked) == 7
        assert len(set(picked)) == 7
        assert set(picked) <= set(ALL_TERMINALS)


def test_draw_spreads_across_families():
    """One field per family while families last — 'roa', 'cs_roa' and 'cs_gp_at' are one
    profitability bet, so a draw that spent 3 of its 7 slots on them would be near-blind."""
    rng = random.Random(1)
    for _ in range(50):
        picked = sample_subset(ALL_TERMINALS, size=7, rng=rng)
        families = [family_of(t) for t in picked]
        assert len(set(families)) == len(families)


def test_every_draw_contains_a_daily_field():
    """Fundamentals are quarterly values forward-filled to daily; a subset drawn entirely
    from them makes every ts_* operator degenerate and wastes a full paid debate."""
    rng = random.Random(2)
    for _ in range(100):
        picked = sample_subset(ALL_TERMINALS, size=7, rng=rng)
        assert any(family_of(t) in DAILY_FAMILIES for t in picked)


def test_draws_rarely_overlap():
    """The whole point: independent ideas must not be shown the same fields. Two draws of
    7 from 54 should share ~1 field (y^2/x), and certainly not be near-copies."""
    rng = random.Random(3)
    draws = [set(sample_subset(ALL_TERMINALS, size=7, rng=rng)) for _ in range(60)]
    overlaps = [len(a & b) for i, a in enumerate(draws) for b in draws[i + 1:]]
    assert max(overlaps) < 7                       # no two draws are identical
    assert sum(overlaps) / len(overlaps) < 2.0     # average overlap stays near one field


def test_no_field_is_wildly_over_exposed():
    """REGRESSION: family selection must be weighted by family SIZE.

    Under a flat family shuffle, `payout` (one member) got the same turn as `profitability`
    (twelve), so `divyield` appeared in 55% of draws and `cs_gp_at` in 2% — the old
    concentration problem rebuilt from the other end. Perfect flatness is unreachable while
    keeping one field per family (7 of 12 families must be picked, so a 12-member family
    cannot be picked 12x as often as a 1-member one); this pins the achievable range.
    """
    rng = random.Random(11)
    draws = [sample_subset(ALL_TERMINALS, size=7, rng=rng) for _ in range(2000)]
    counts = {t: 0 for t in ALL_TERMINALS}
    for draw in draws:
        for t in draw:
            counts[t] += 1
    share = {t: c / len(draws) for t, c in counts.items()}
    assert min(share.values()) > 0.03, f"starved field: {min(share, key=share.get)}"
    assert max(share.values()) < 0.35, f"over-exposed field: {max(share, key=share.get)}"


def test_menu_smaller_than_subset_is_returned_whole():
    """Subsetting a menu that is already tiny would only starve the idea."""
    assert sample_subset(["close", "volume"], size=7) == ["close", "volume"]
    assert sample_subset(["close", "volume"], size=2) == ["close", "volume"]


def test_unknown_terminals_are_still_sampled():
    """A newly ingested field (e.g. an osap_* panel) must not silently drop out of the menu
    just because it predates the family map."""
    menu = ["close", "volume", "osap_mom12m", "osap_ivol", "bm"]
    picked = sample_subset(menu, size=3, rng=random.Random(0))
    assert len(picked) == 3
    assert family_of("osap_mom12m") == "other"
    seen = set()
    for seed in range(40):
        seen |= set(sample_subset(menu, size=3, rng=random.Random(seed)))
    assert {"osap_mom12m", "osap_ivol"} <= seen


def test_same_seed_gives_same_draw():
    """A run's draws must be reproducible from its seed alone (they are logged per alpha)."""
    a = sample_subset(ALL_TERMINALS, size=7, rng=random.Random(42))
    b = sample_subset(ALL_TERMINALS, size=7, rng=random.Random(42))
    assert a == b


@pytest.mark.parametrize("size", [3, 5, 7, 12, 20])
def test_sizes_beyond_family_count_still_fill(size):
    """With 12 families, a size-20 draw must wrap around and take seconds, not come up short."""
    picked = sample_subset(ALL_TERMINALS, size=size, rng=random.Random(7))
    assert len(picked) == size and len(set(picked)) == size
