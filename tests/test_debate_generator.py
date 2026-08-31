"""Tests for DebateGenerator (the multi-agent debate wired as a factory generator).

The debate itself is stubbed (no network) — we only verify the generator contract that
``run_factory_parallel`` relies on: ``_one(terminals)`` returns an ``IdeaBatch`` of the
debate's selected formulas (or ``None`` when the debate yields nothing / errors), and the
cost counters advance.
"""

import alpha_gpt.debate.moderator as moderator
from alpha_gpt.config import Config
from alpha_gpt.debate.models import FormulaProposal, ResearchHypothesisSpec, SeedFormulaPack
from alpha_gpt.factory.generators import DebateGenerator, IdeaBatch


def _stub_debate(monkeypatch, expressions):
    def fake_idea(terms, client, model, config=None, usage=None):
        return [ResearchHypothesisSpec(hypothesis_id="h1", title="t", idea_label="gross margin drift",
                                       confidence=4, summary="a grounded story")], {}

    def fake_formula(hyps, terms, client, model, config=None, diagnose_fn=None, usage=None):
        formulas = [FormulaProposal(formula_id=f"f{i}", hypothesis_id="h1", agent_name="A",
                                    expression=e) for i, e in enumerate(expressions)]
        return SeedFormulaPack(pack_id="p", selected_formulas=formulas), {}

    monkeypatch.setattr(moderator, "run_idea_debate", fake_idea)
    monkeypatch.setattr(moderator, "run_formula_debate", fake_formula)


def _gen():
    return DebateGenerator(client=object(), model="m", available_terminals=["close"], config=Config())


def test_one_returns_batch_of_debate_formulas(monkeypatch):
    _stub_debate(monkeypatch, ["cs_rank(ts_returns(close, 20))", "cs_zscore(close)"])
    gen = _gen()
    batch = gen._one(["close", "cs_gpmargin"])
    assert isinstance(batch, IdeaBatch)
    assert batch.source == "debate"
    # No theme is handed down: the batch is named by the label the MODEL chose, and carries
    # the subset it was shown plus its self-rated confidence.
    assert batch.idea == "gross margin drift"
    assert batch.terminals == ["close", "cs_gpmargin"] and batch.confidence == 4
    assert batch.expressions == ["cs_rank(ts_returns(close, 20))", "cs_zscore(close)"]
    assert batch.hypothesis == "a grounded story"
    # cost counters advanced (so the factory's cost accounting is non-trivial)
    assert gen.n_calls == 1 and gen.total_prompt_tokens > 0 and gen.total_completion_tokens > 0


def test_one_returns_none_when_debate_yields_no_formulas(monkeypatch):
    _stub_debate(monkeypatch, [])
    assert _gen()._one(["close"]) is None


def test_one_returns_none_on_debate_exception(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("debate failed")
    monkeypatch.setattr(moderator, "run_idea_debate", boom)
    assert _gen()._one(["close"]) is None


def test_generate_yields_batches(monkeypatch):
    _stub_debate(monkeypatch, ["cs_rank(close)"])
    batches = list(_gen().generate(3))
    assert len(batches) == 3
    assert all(b.source == "debate" and b.expressions == ["cs_rank(close)"] for b in batches)
