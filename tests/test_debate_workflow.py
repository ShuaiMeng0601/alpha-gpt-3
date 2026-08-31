import json
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from alpha_gpt.config import Config
from alpha_gpt.debate.models import (
    FormulaProposal,
    FormulaRevision,
    FormulaReview,
    IdeaProposal,
    IdeaRevision,
    IdeaReview,
    ResearchHypothesisSpec,
    make_id,
)
from alpha_gpt.debate.moderator import run_formula_debate, run_idea_debate


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


class _FakeCompletions:
    def __init__(self, contents):
        self.contents = list(contents)

    def create(self, **kwargs):
        return _FakeResponse(self.contents.pop(0))


class _FakeClient:
    def __init__(self, contents):
        self.chat = SimpleNamespace(completions=_FakeCompletions(contents))


class _StubAgent:
    def __init__(self, name: str):
        self.name = name

    def draft_idea(self, brief):
        return IdeaProposal(
            proposal_id=make_id("idea-draft", self.name),
            agent_name=self.name,
            title=f"{self.name} hypothesis",
            mechanism=f"{self.name} mechanism",
            signal_type="cross_sectional_alpha",
            payoff_definition="Predicts relative returns.",
            directionality="directional",
            direction_separation_plan="Not needed.",
            data_definition="Use price and volume proxies.",
            candidate_proxies=["close", "volume"],
            subfactor_design=[f"{self.name} subfactor"],
            filter_policy="No extra filter.",
            normalization_policy="cs_rank",
            neutralization_policy="none",
            implementability="directly_supported",
            open_risks=["crowding"],
            stage2_constraints=["Keep formulas simple."],
            summary=f"{self.name} summary",
        )

    def review_ideas(self, brief, proposals):
        return [
            IdeaReview(
                review_id=make_id("idea-review", self.name, proposal.proposal_id),
                reviewer_agent_name=self.name,
                target_proposal_id=proposal.proposal_id,
                mechanism_quality=4,
                signal_type_clarity=4,
                payoff_clarity=4,
                directionality_clarity=4,
                subfactor_quality=4,
                filter_logic=4,
                normalization_soundness=4,
                implementability=4,
                decision="accept_with_revision",
                comments=[f"Review from {self.name}"],
            )
            for proposal in proposals
        ]

    def revise_idea(self, brief, proposal, reviews):
        return IdeaRevision(
            revision_id=make_id("idea-revision", self.name),
            agent_name=self.name,
            base_proposal_id=proposal.proposal_id,
            accepted_feedback=[review.comments[0] for review in reviews],
            revision_summary=f"{self.name} revised",
            revised_proposal=proposal,
        )

    def draft_formulas(self, briefs):
        proposals = []
        for idx, brief in enumerate(briefs, start=1):
            if self.name == "Statistician" and idx == 1:
                expression = "not_a_formula(close)"
            elif self.name == "Trader" and idx == 1:
                expression = "cs_rank(volume)"
            else:
                expression = "cs_rank(close)"
            role = "filter" if self.name == "Trader" and idx == 1 else "main_alpha"
            proposals.append(
                FormulaProposal(
                    formula_id=make_id("formula", self.name, brief.hypothesis_id, str(idx)),
                    hypothesis_id=brief.hypothesis_id,
                    agent_name=self.name,
                    formula_role=role,
                    expression=expression,
                    plain_language_mapping=f"{self.name} formula for {brief.hypothesis_id}",
                    terminals_used=["close"],
                    operators_used=["cs_rank"],
                    expected_signal_direction="positive",
                    rationale="Stub rationale",
                )
            )
        return proposals

    def review_formulas(self, proposals):
        return [
            FormulaReview(
                review_id=make_id("formula-review", self.name, proposal.formula_id),
                reviewer_agent_name=self.name,
                target_formula_id=proposal.formula_id,
                faithfulness=4,
                implementability=4,
                robustness=4,
                novelty=4,
                simplicity=4,
                decision="accept_with_revision",
                comments=[f"Formula review from {self.name}"],
            )
            for proposal in proposals
        ]

    def revise_formulas(self, proposals, reviews, briefs_by_id):
        return [
            FormulaRevision(
                revision_id=make_id("formula-revision", self.name, proposal.formula_id),
                agent_name=self.name,
                base_formula_id=proposal.formula_id,
                accepted_feedback=[review.comments[0] for review in reviews if review.target_formula_id == proposal.formula_id],
                revision_summary=f"{self.name} revised formula",
                revised_formula=proposal,
            )
            for proposal in proposals
        ]


class DebateWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.config.idea_hypothesis_target = 2
        self.config.seed_formula_target = 5
        self.stub_agents = [
            _StubAgent("Economist"),
            _StubAgent("Statistician"),
            _StubAgent("Trader"),
        ]

    def test_run_idea_debate_produces_expected_artifact_counts(self):
        client = _FakeClient([
            """
            [
              {
                "title": "Hypothesis Alpha",
                "source_agents": ["Economist", "Trader"],
                "mechanism": "Mechanism A",
                "signal_type": "cross_sectional_alpha",
                "payoff_definition": "Relative return",
                "directionality": "directional",
                "direction_separation_plan": "Not needed",
                "data_definition": "close and volume",
                "candidate_proxies": ["close"],
                "subfactor_design": ["sub_a"],
                "filter_policy": "none",
                "normalization_policy": "cs_rank",
                "neutralization_policy": "none",
                "implementability": "directly_supported",
                "open_risks": ["crowding"],
                "stage2_constraints": ["simple"],
                "summary": "A"
              },
              {
                "title": "Hypothesis Beta",
                "source_agents": ["Statistician"],
                "mechanism": "Mechanism B",
                "signal_type": "filter",
                "payoff_definition": "Conditioning",
                "directionality": "non_directional",
                "direction_separation_plan": "Use as filter",
                "data_definition": "returns dispersion",
                "candidate_proxies": ["returns"],
                "subfactor_design": ["sub_b"],
                "filter_policy": "regime gate",
                "normalization_policy": "zscore",
                "neutralization_policy": "none",
                "implementability": "directly_supported",
                "open_risks": ["lag"],
                "stage2_constraints": ["allow filter role"],
                "summary": "B"
              }
            ]
            """
        ])

        with patch("alpha_gpt.debate.moderator.create_agents", return_value=self.stub_agents):
            hypotheses, artifacts = run_idea_debate(
                available_terminals=["close", "volume", "returns"],
                client=client,
                model="fake-model",
                config=self.config,
            )

        self.assertEqual(len(artifacts["idea_drafts.json"]), 3)
        self.assertEqual(len(artifacts["idea_reviews.json"]), 6)
        self.assertEqual(len(artifacts["idea_revisions.json"]), 3)
        self.assertEqual(len(hypotheses), 2)

    def test_run_formula_debate_keeps_parseable_filter_formulas(self):
        hypotheses = [
            ResearchHypothesisSpec(
                hypothesis_id="hypothesis-alpha",
                title="Hypothesis Alpha",
                summary="Summary A",
                stage2_constraints=["simple"],
            ),
            ResearchHypothesisSpec(
                hypothesis_id="hypothesis-beta",
                title="Hypothesis Beta",
                summary="Summary B",
                stage2_constraints=["allow filter role"],
            ),
        ]
        selected_ids = [
            make_id("formula", "Economist", "hypothesis-alpha", "1"),
            make_id("formula", "Economist", "hypothesis-beta", "2"),
            make_id("formula", "Statistician", "hypothesis-beta", "2"),
            make_id("formula", "Trader", "hypothesis-alpha", "1"),
            make_id("formula", "Trader", "hypothesis-beta", "2"),
        ]
        selected_filter_id = make_id("formula", "Trader", "hypothesis-alpha", "1")
        client = _FakeClient([
            f"""
            {{
              "selected_formula_ids": {json.dumps(selected_ids)},
              "selection_rationale": ["diverse roles", "cross-hypothesis coverage"]
            }}
            """
        ])

        with patch("alpha_gpt.debate.moderator.create_agents", return_value=self.stub_agents):
            seed_pack, artifacts = run_formula_debate(
                hypotheses=hypotheses,
                available_terminals=["close", "volume", "returns"],
                client=client,
                model="fake-model",
                config=self.config,
            )

        self.assertTrue(seed_pack.selected_formulas)
        self.assertEqual(len(seed_pack.selected_formulas), 5)
        self.assertTrue(all(formula.parseable for formula in seed_pack.selected_formulas))
        self.assertTrue(any(formula.formula_role == "filter" for formula in seed_pack.selected_formulas))
        self.assertIn(selected_filter_id, [formula.formula_id for formula in seed_pack.selected_formulas])
        self.assertTrue(any(not revision.revised_formula.parseable for revision in artifacts["formula_revisions.json"]))

    def test_run_formula_debate_diagnostics_annotate_and_gate(self):
        """With diagnose_fn wired in: drafts get measured diagnostics before review, and a
        verifier-REJECTED formula is dropped before moderator selection even if the
        moderator asks for it."""
        def diagnose(expr):
            if expr == "cs_rank(volume)":
                return "REJECTED (all_nan): evaluates to all-NaN"
            return ("OK | coverage 46% of cells, ~1100 stocks/day, rank turnover 0.31, "
                    "distinct values 98% of names/day, dead days 0%")

        hypotheses = [
            ResearchHypothesisSpec(hypothesis_id="hypothesis-alpha", title="Hypothesis Alpha",
                                   summary="Summary A", stage2_constraints=["simple"]),
            ResearchHypothesisSpec(hypothesis_id="hypothesis-beta", title="Hypothesis Beta",
                                   summary="Summary B", stage2_constraints=["allow filter role"]),
        ]
        rejected_id = make_id("formula", "Trader", "hypothesis-alpha", "1")  # cs_rank(volume)
        healthy_id = make_id("formula", "Economist", "hypothesis-alpha", "1")      # cs_rank(close)
        client = _FakeClient([
            f"""
            {{
              "selected_formula_ids": {json.dumps([healthy_id, rejected_id])},
              "selection_rationale": ["evidence-based"]
            }}
            """
        ])

        with patch("alpha_gpt.debate.moderator.create_agents", return_value=self.stub_agents):
            seed_pack, artifacts = run_formula_debate(
                hypotheses=hypotheses,
                available_terminals=["close", "volume", "returns"],
                client=client,
                model="fake-model",
                config=self.config,
                diagnose_fn=diagnose,
            )

        # Drafts were measured before critique (reviewers see evidence, not just prose).
        drafts = artifacts["formula_drafts.json"]
        self.assertTrue(all(d.diagnostics for d in drafts if d.expression))
        # The verifier-rejected formula never reaches selection, despite the moderator
        # naming its id; it lands in dropped_formulas with the rejection attached.
        selected_ids = [f.formula_id for f in seed_pack.selected_formulas]
        self.assertIn(healthy_id, selected_ids)
        self.assertNotIn(rejected_id, selected_ids)
        dropped_by_id = {f.formula_id: f for f in seed_pack.dropped_formulas}
        self.assertIn(rejected_id, dropped_by_id)
        self.assertTrue(dropped_by_id[rejected_id].diagnostics.startswith("REJECTED"))
