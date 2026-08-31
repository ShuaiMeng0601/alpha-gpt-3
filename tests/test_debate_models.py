import json
import unittest

from alpha_gpt.debate.models import (
    FormulaProposal,
    IdeaProposal,
    IdeaRevision,
    SeedFormulaPack,
    to_jsonable,
)


class DebateModelsTest(unittest.TestCase):
    def test_nested_dataclasses_are_json_serializable(self):
        proposal = IdeaProposal(
            proposal_id="idea-1",
            agent_name="Momentum",
            title="Draft",
            mechanism="Mechanism",
            candidate_proxies=["proxy_a", "proxy_b"],
        )
        revision = IdeaRevision(
            revision_id="rev-1",
            agent_name="Momentum",
            base_proposal_id="idea-1",
            accepted_feedback=["good point"],
            revised_proposal=proposal,
        )
        formula = FormulaProposal(
            formula_id="formula-1",
            hypothesis_id="hyp-1",
            agent_name="Momentum",
            formula_role="filter",
            expression="cs_rank(close)",
        )
        pack = SeedFormulaPack(
            pack_id="pack-1",
            selected_formulas=[formula],
            traceability_map={"formula-1": "hyp-1"},
        )

        payload = to_jsonable({"revision": revision, "pack": pack})
        serialized = json.dumps(payload)

        self.assertIn('"proposal_id": "idea-1"', serialized)
        self.assertIn('"formula_role": "filter"', serialized)
        self.assertEqual(payload["pack"]["traceability_map"]["formula-1"], "hyp-1")

