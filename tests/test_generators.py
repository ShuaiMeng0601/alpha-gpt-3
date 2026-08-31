"""Tests for alpha_gpt.factory.generators (random + LLM generators).

Includes a regression for honest token accounting across LLM retries.
"""

import re
from types import SimpleNamespace

import pytest

from alpha_gpt.factory.generators import (
    LLMGenerator, RandomGenerator, RandomPriceGenerator, PRICE_VOLUME_TERMINALS)

VALID = '{"idea": "i", "hypothesis": "h", "expressions": ["cs_rank(close)"]}'


class _Transient(Exception):
    """A retryable (429-shaped) API error, so the generator backs off instead of failing fast.
    (A bare Exception is now treated as a permanent error and is NOT retried — see _is_retryable.)"""
    status_code = 429


class _ScriptedClient:
    """OpenAI-shaped client whose create() replays a scripted list of contents/errors."""

    def __init__(self, scripted):
        self._s = list(scripted)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        item = self._s[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
            choices=[SimpleNamespace(message=SimpleNamespace(content=item))],
        )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    # don't actually back off during retry tests
    monkeypatch.setattr("alpha_gpt.factory.generators.time.sleep", lambda *a, **k: None)


def test_llm_single_success_accounting():
    g = LLMGenerator(_ScriptedClient([VALID]), "m", ["close"], seed=0)
    b = g._one(["close"])
    assert b is not None
    assert b.prompt_tokens == 100
    assert g.total_prompt_tokens == 100 and g.total_completion_tokens == 50
    assert g.n_calls == 1


def test_llm_retry_counts_only_billed_calls_regression():
    """REGRESSION: a transient (retryable) error on the first attempt errors before billing;
    totals reflect only the billed (successful) call, and the batch carries that call's tokens."""
    g = LLMGenerator(_ScriptedClient([_Transient("rate limit"), VALID]), "m", ["close"], seed=0)
    b = g._one(["close"])
    assert b is not None and b.prompt_tokens == 100
    assert g.total_prompt_tokens == 100


def test_llm_total_failure_returns_none():
    g = LLMGenerator(_ScriptedClient([RuntimeError("x")] * 3), "m", ["close"], seed=0)
    assert g._one(["close"]) is None
    assert g.total_prompt_tokens == 0
    assert g.n_calls == 1


def test_usage_meter_records_real_tokens_through_call_json():
    """call_json(usage=meter) must record every billed response's real usage — the debate's
    cost accounting is built on this (no more flat per-debate estimates)."""
    from alpha_gpt.llm import UsageMeter, call_json
    meter = UsageMeter()
    client = _ScriptedClient(['{"ok": 1}'])
    out = call_json(client, "m", "sys", "user", usage=meter)
    assert out == {"ok": 1}
    assert (meter.prompt_tokens, meter.completion_tokens, meter.calls) == (100, 50, 1)


def test_usage_meter_counts_billed_repair_attempts():
    """A first response with unparseable JSON is still billed; the repair retry is a second
    billed call. The meter must count BOTH (honest over-report, not just successes)."""
    from alpha_gpt.llm import UsageMeter, call_json
    meter = UsageMeter()
    client = _ScriptedClient(["not json at all", '{"ok": 1}'])
    out = call_json(client, "m", "sys", "user", retries=2, usage=meter)
    assert out == {"ok": 1}
    assert meter.calls == 2 and meter.prompt_tokens == 200


def test_llm_budget_guard_stops_new_generation():
    """max_cost_usd: once the running estimated spend crosses the cap, _one refuses to start
    new paid work (returns None without calling the API)."""
    client = _ScriptedClient([VALID])
    g = LLMGenerator(client, "m", ["close"], seed=0, max_cost_usd=0.000001)
    g.total_prompt_tokens = 10_000_000  # pretend we've already spent a lot
    assert g._one(["close"]) is None
    assert client.calls == 0  # no new API call was made


def test_debate_crash_still_records_billed_usage(monkeypatch):
    """REGRESSION: a debate that pays for calls then crashes mid-way must still fold those
    billed tokens into the generator totals (was: silently $0)."""
    from alpha_gpt.factory.generators import DebateGenerator

    def _fake_idea_debate(terminals, client, model, config=None, usage=None):
        usage.record(SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1234, completion_tokens=567)))
        raise RuntimeError("boom mid-debate")

    monkeypatch.setattr("alpha_gpt.debate.moderator.run_idea_debate", _fake_idea_debate)
    g = DebateGenerator(client=None, model="m", available_terminals=["close"],
                        config=SimpleNamespace(save_debate_artifacts=False, terminal_subset_size=None))
    assert g._one(["close"]) is None
    assert g.total_prompt_tokens == 1234 and g.total_completion_tokens == 567
    assert g.n_calls == 1


def test_random_generator_yields_expressions(pset):
    gen = RandomGenerator(pset, alphas_per_idea=4, seed=0)
    batches = list(gen.generate(3))
    assert len(batches) == 3
    assert all(b.source == "random" for b in batches)
    assert sum(len(b.expressions) for b in batches) > 0


def test_random_price_generator_emits_only_ohlcv_no_fundamentals():
    """The price/volume-only control MUST never touch fundamentals (that's its whole point:
    isolate price factors + sidestep reporting-date look-ahead). Builds its own OHLCV pset,
    so needs no pipeline pset."""
    gen = RandomPriceGenerator(alphas_per_idea=5, seed=0)
    batches = list(gen.generate(8))
    exprs = [e for b in batches for e in b.expressions]
    assert exprs and all(b.source == "random_price" for b in batches)

    tokens = set(re.findall(r"[A-Za-z_]\w*", " ".join(exprs)))
    fundamentals = {"roe", "roa", "bm", "market_cap", "shrout", "returns", "ptb", "npm",
                    "gpm", "debt_at", "curr_ratio", "accrual", "pe_op_dil", "divyield", "ps"}
    assert tokens & fundamentals == set(), f"leaked fundamentals: {tokens & fundamentals}"
    assert tokens & set(PRICE_VOLUME_TERMINALS), "expected at least one OHLCV terminal"


def test_insufficient_quota_is_not_retryable():
    """REGRESSION: OpenAI reports an empty balance as a 429 with code insufficient_quota;
    backoff can never fix it, so it must fail fast instead of burning the retry budget."""
    from alpha_gpt.llm import _is_retryable

    class _Quota(Exception):
        status_code = 429
    exc = _Quota("Error code: 429 - {'error': {'code': 'insufficient_quota'}}")
    assert _is_retryable(exc) is False
    assert _is_retryable(_Transient("plain rate limit")) is True  # ordinary 429 still retries
