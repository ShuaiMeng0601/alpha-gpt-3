"""Two-stage debate orchestration and moderator synthesis."""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
from typing import Any

from openai import OpenAI

from alpha_gpt.config import Config
from alpha_gpt.llm import call_json
from alpha_gpt.debate.agents import create_agents
from alpha_gpt.debate.models import (
    FormulaDebateBrief,
    FormulaProposal,
    FormulaRevision,
    FormulaReview,
    IdeaDebateBrief,
    IdeaProposal,
    IdeaRevision,
    IdeaReview,
    ResearchHypothesisSpec,
    SeedFormulaPack,
    coerce_list_of_str,
    coerce_score,
    coerce_str,
    make_id,
    to_jsonable,
)
from alpha_gpt.debate.prompts import (
    MODERATOR_FORMULA_SYSTEM,
    MODERATOR_FORMULA_USER,
    MODERATOR_IDEA_SYSTEM,
    MODERATOR_IDEA_USER,
    OPERATOR_CATALOG,
)

logger = logging.getLogger(__name__)


def _config_value(config: Config | None, name: str, default: Any) -> Any:
    return getattr(config, name, default) if config is not None else default


def _run_parallel(thunks: list) -> list:
    """Run zero-arg callables concurrently, returning results in INPUT order.

    Used to fan the 3 debate personas across a phase (draft / review / revise): they are
    independent within a phase (only stages depend on each other), so running them in
    parallel cuts a debate's wall-clock without changing its logic. Falls back to serial
    for a single thunk.
    """
    if len(thunks) <= 1:
        return [t() for t in thunks]
    with cf.ThreadPoolExecutor(max_workers=len(thunks)) as ex:
        return list(ex.map(lambda f: f(), thunks))


def _moderator_json(client, model, system_prompt, user_prompt, empty_factory,
                    retries=2, max_tokens=4000, usage=None) -> Any:
    """Moderator JSON call (lower temperature, larger budget than agent calls)."""
    return call_json(
        client, model, system_prompt, user_prompt,
        empty_factory=empty_factory, retries=retries, temperature=0.2, max_tokens=max_tokens,
        label="moderator", usage=usage,
    )


def _proposal_to_hypothesis(proposal: IdeaProposal, hypothesis_id: str | None = None) -> ResearchHypothesisSpec:
    return ResearchHypothesisSpec(
        hypothesis_id=hypothesis_id or make_id("hypothesis", proposal.agent_name, proposal.proposal_id),
        idea_label=proposal.idea_label,
        confidence=proposal.confidence,
        title=proposal.title,
        source_agents=[proposal.agent_name],
        mechanism=proposal.mechanism,
        signal_type=proposal.signal_type,
        payoff_definition=proposal.payoff_definition,
        directionality=proposal.directionality,
        direction_separation_plan=proposal.direction_separation_plan,
        data_definition=proposal.data_definition,
        candidate_proxies=list(proposal.candidate_proxies),
        subfactor_design=list(proposal.subfactor_design),
        filter_policy=proposal.filter_policy,
        normalization_policy=proposal.normalization_policy,
        neutralization_policy=proposal.neutralization_policy,
        implementability=proposal.implementability,
        open_risks=list(proposal.open_risks),
        stage2_constraints=list(proposal.stage2_constraints),
        summary=proposal.summary,
    )


def _hypothesis_from_payload(payload: Any, idx: int) -> ResearchHypothesisSpec:
    payload = payload if isinstance(payload, dict) else {}
    return ResearchHypothesisSpec(
        hypothesis_id=make_id("hypothesis", coerce_str(payload.get("title")) or str(idx + 1)),
        idea_label=coerce_str(payload.get("idea_label")),
        confidence=coerce_score(payload.get("confidence")),
        title=coerce_str(payload.get("title")) or f"Hypothesis {idx + 1}",
        source_agents=coerce_list_of_str(payload.get("source_agents")),
        mechanism=coerce_str(payload.get("mechanism")),
        signal_type=coerce_str(payload.get("signal_type")),
        payoff_definition=coerce_str(payload.get("payoff_definition")),
        directionality=coerce_str(payload.get("directionality")),
        direction_separation_plan=coerce_str(payload.get("direction_separation_plan")),
        data_definition=coerce_str(payload.get("data_definition")),
        candidate_proxies=coerce_list_of_str(payload.get("candidate_proxies")),
        subfactor_design=coerce_list_of_str(payload.get("subfactor_design")),
        filter_policy=coerce_str(payload.get("filter_policy")),
        normalization_policy=coerce_str(payload.get("normalization_policy")),
        neutralization_policy=coerce_str(payload.get("neutralization_policy")),
        implementability=coerce_str(payload.get("implementability")),
        open_risks=coerce_list_of_str(payload.get("open_risks")),
        stage2_constraints=coerce_list_of_str(payload.get("stage2_constraints")),
        summary=coerce_str(payload.get("summary")),
    )


def _build_idea_brief(available_terminals: list[str]) -> IdeaDebateBrief:
    constraints = [
        "Do not use future information or look-ahead logic.",
        "Do not write formulas during Stage 1.",
        "Use only concepts that can be supported directly or through proxies from the available terminals.",
    ]
    data_notes = [
        f"Available terminals for this run: {', '.join(available_terminals) or 'none'}.",
        "This is a random subset of the full field library, drawn for this debate only.",
        "Cross-sectional and time-series roles should be made explicit in the hypothesis.",
    ]
    return IdeaDebateBrief(
        available_terminals=available_terminals,
        operator_catalog=OPERATOR_CATALOG,
        constraints=constraints,
        data_notes=data_notes,
    )


def _build_formula_briefs(
    hypotheses: list[ResearchHypothesisSpec],
    available_terminals: list[str],
) -> list[FormulaDebateBrief]:
    briefs = []
    for hypothesis in hypotheses:
        constraints = list(hypothesis.stage2_constraints)
        constraints.append(f"Use only these terminals: {', '.join(available_terminals) or 'none'}.")
        briefs.append(
            FormulaDebateBrief(
                hypothesis_id=hypothesis.hypothesis_id,
                hypothesis_title=hypothesis.title,
                hypothesis_summary=hypothesis.summary,
                available_terminals=available_terminals,
                operator_catalog=OPERATOR_CATALOG,
                formula_constraints=constraints,
            )
        )
    return briefs


def _annotate_diagnostics(formulas: list[FormulaProposal], diagnose_fn) -> None:
    """Attach measured, return-blind signal diagnostics to each drafted formula.

    ``diagnose_fn`` maps an expression string to a one-line diagnostic (see
    factory.verifier.diagnose_expression). The line rides along inside each proposal's
    JSON, so reviewers, revisers, and the moderator all see the same evidence. A
    diagnostics failure never interrupts the debate."""
    if diagnose_fn is None:
        return
    for formula in formulas:
        if not formula.expression:
            continue
        try:
            formula.diagnostics = diagnose_fn(formula.expression)
        except Exception:
            logger.warning("diagnose_fn failed for %r", formula.expression, exc_info=True)
            formula.diagnostics = ""


def _annotate_formula_parse_status(formulas: list[FormulaProposal], available_terminals: list[str]) -> None:
    try:
        from alpha_gpt.expr.primitives import create_primitive_set
        from alpha_gpt.expr.seed_injector import normalize_expression, parse_expression
    except ModuleNotFoundError:
        for formula in formulas:
            formula.normalized_expression = formula.expression
            formula.parseable = bool(formula.expression) and "not_a_formula" not in formula.expression
            formula.parse_error = "" if formula.parseable else "Parser dependencies are unavailable."
        return

    if not available_terminals:
        for formula in formulas:
            formula.parseable = False
            formula.parse_error = "No available terminals for parse gate."
            formula.normalized_expression = formula.expression
        return

    pset = create_primitive_set(available_terminals)
    for formula in formulas:
        if not formula.expression:
            formula.parseable = False
            formula.parse_error = "Empty expression."
            formula.normalized_expression = ""
            continue
        normalized = normalize_expression(formula.expression)
        formula.normalized_expression = normalized
        tree = parse_expression(formula.expression, pset)
        if tree is None:
            formula.parseable = False
            formula.parse_error = "Parser returned None."
        else:
            formula.parseable = True
            formula.parse_error = ""


def _synthesize_hypotheses(
    available_terminals: list[str],
    revisions: list[IdeaRevision],
    client: OpenAI,
    model: str,
    target_count: int,
    retries: int,
    max_tokens: int = 4000,
    usage=None,
) -> list[ResearchHypothesisSpec]:
    user_msg = MODERATOR_IDEA_USER.format(
        available_terminals=", ".join(available_terminals) or "none",
        revisions_json=json.dumps(to_jsonable(revisions), indent=2, ensure_ascii=False),
    )
    payload = _moderator_json(
        client=client,
        model=model,
        system_prompt=MODERATOR_IDEA_SYSTEM,
        user_prompt=user_msg,
        empty_factory=list,
        retries=retries,
        max_tokens=max_tokens,
        usage=usage,
    )
    items = payload if isinstance(payload, list) else []
    hypotheses = [
        _hypothesis_from_payload(item, idx)
        for idx, item in enumerate(items[:target_count])
    ]
    if hypotheses:
        return hypotheses

    fallback = []
    for idx, revision in enumerate(revisions[:target_count]):
        fallback.append(
            _proposal_to_hypothesis(
                revision.revised_proposal,
                hypothesis_id=make_id("hypothesis", revision.agent_name, str(idx + 1)),
            )
        )
    return fallback


def _select_seed_formulas(
    hypotheses: list[ResearchHypothesisSpec],
    formulas: list[FormulaProposal],
    client: OpenAI,
    model: str,
    target_count: int,
    retries: int,
    max_tokens: int = 4000,
    usage=None,
) -> SeedFormulaPack:
    # Verify-gate: a formula must parse AND survive the signal verifier (a formula whose
    # diagnostics say REJECTED would be discarded downstream anyway — dropping it here
    # keeps the moderator choosing among viable candidates only).
    def _selectable(formula: FormulaProposal) -> bool:
        return formula.parseable and not formula.diagnostics.startswith("REJECTED")

    parseable = [formula for formula in formulas if _selectable(formula)]
    dropped = [formula for formula in formulas if not _selectable(formula)]
    if not parseable:
        return SeedFormulaPack(
            pack_id=make_id("seed-pack", "empty"),
            selected_formulas=[],
            selection_rationale=["No parseable formulas were available after Stage 2."],
            traceability_map={},
            dropped_formulas=dropped,
        )

    user_msg = MODERATOR_FORMULA_USER.format(
        hypotheses_json=json.dumps(to_jsonable(hypotheses), indent=2, ensure_ascii=False),
        formula_candidates_json=json.dumps(to_jsonable(parseable), indent=2, ensure_ascii=False),
        target_count=target_count,
    )
    payload = _moderator_json(
        client=client,
        model=model,
        system_prompt=MODERATOR_FORMULA_SYSTEM,
        user_prompt=user_msg,
        empty_factory=dict,
        retries=retries,
        max_tokens=max_tokens,
        usage=usage,
    )

    parseable_by_id = {formula.formula_id: formula for formula in parseable}
    selected_ids = []
    selection_rationale = []
    if isinstance(payload, dict):
        selected_ids = [
            formula_id
            for formula_id in coerce_list_of_str(payload.get("selected_formula_ids"))
            if formula_id in parseable_by_id
        ]
        selection_rationale = coerce_list_of_str(payload.get("selection_rationale"))
        selected_ids = list(dict.fromkeys(selected_ids))

    if not selected_ids:
        selected_ids = [formula.formula_id for formula in parseable[:target_count]]
        selection_rationale = ["Fallback selection: first parseable formulas by revised order."]

    selected = [parseable_by_id[formula_id] for formula_id in selected_ids[:target_count]]
    selected_id_set = {formula.formula_id for formula in selected}
    dropped.extend(formula for formula in parseable if formula.formula_id not in selected_id_set)
    return SeedFormulaPack(
        pack_id=make_id("seed-pack", "selected"),
        selected_formulas=selected,
        selection_rationale=selection_rationale,
        traceability_map={formula.formula_id: formula.hypothesis_id for formula in selected},
        dropped_formulas=dropped,
    )


def run_idea_debate(
    available_terminals: list[str],
    client: OpenAI,
    model: str,
    config: Config | None = None,
    usage=None,
) -> tuple[list[ResearchHypothesisSpec], dict[str, Any]]:
    """Run Stage 1 idea debate and return hypotheses plus saved artifacts.

    ``available_terminals`` is the debate's seed: a random subset of the run's field
    library (see ``expr.terminal_subset``). There is no assigned theme — the agents invent
    whatever hypothesis those fields can support, which is what keeps independent debates
    from converging on the same handful of famous anomalies.

    ``usage`` (optional llm.UsageMeter): records the real billed tokens of every agent and
    moderator call in this debate, for honest cost accounting."""
    brief = _build_idea_brief(available_terminals)
    json_retries = _config_value(config, "debate_json_retries", 2)
    target_count = _config_value(config, "idea_hypothesis_target", 3)
    agent_max_tokens = _config_value(config, "agent_max_tokens", 4000)
    moderator_max_tokens = _config_value(config, "moderator_max_tokens", 4000)
    agents = create_agents(client, model, json_retries=json_retries, max_tokens=agent_max_tokens,
                           usage=usage)

    # Each phase runs the 3 personas in parallel (they're independent within a phase).
    idea_drafts = _run_parallel([lambda a=agent: a.draft_idea(brief) for agent in agents])

    review_lists = _run_parallel([
        lambda a=agent: a.review_ideas(brief, [p for p in idea_drafts if p.agent_name != a.name])
        for agent in agents
    ])
    idea_reviews: list[IdeaReview] = [review for lst in review_lists for review in lst]

    def _revise_one(agent):
        proposal = next(p for p in idea_drafts if p.agent_name == agent.name)
        reviews = [r for r in idea_reviews if r.target_proposal_id == proposal.proposal_id]
        return agent.revise_idea(brief, proposal, reviews)
    idea_revisions: list[IdeaRevision] = _run_parallel(
        [lambda a=agent: _revise_one(a) for agent in agents])

    hypotheses = _synthesize_hypotheses(
        available_terminals=available_terminals,
        revisions=idea_revisions,
        client=client,
        model=model,
        target_count=target_count,
        retries=json_retries,
        max_tokens=moderator_max_tokens,
        usage=usage,
    )

    artifacts = {
        "idea_brief.json": brief,
        "idea_drafts.json": idea_drafts,
        "idea_reviews.json": idea_reviews,
        "idea_revisions.json": idea_revisions,
        "hypotheses.json": hypotheses,
    }
    return hypotheses, artifacts


def run_formula_debate(
    hypotheses: list[ResearchHypothesisSpec],
    available_terminals: list[str],
    client: OpenAI,
    model: str,
    config: Config | None = None,
    diagnose_fn=None,
    usage=None,
) -> tuple[SeedFormulaPack, dict[str, Any]]:
    """Run Stage 2 formula debate and return the seed formula pack.

    ``diagnose_fn`` (optional): expression -> one-line, return-blind diagnostic. When
    provided, drafts are measured on historical data before cross-review (so critique
    is grounded in evidence), revisions are re-measured, and verifier-rejected formulas
    are dropped before moderator selection.
    ``usage`` (optional llm.UsageMeter): records real billed tokens of every call."""
    json_retries = _config_value(config, "debate_json_retries", 2)
    target_count = _config_value(config, "seed_formula_target", 8)
    agent_max_tokens = _config_value(config, "agent_max_tokens", 4000)
    moderator_max_tokens = _config_value(config, "moderator_max_tokens", 4000)
    agents = create_agents(client, model, json_retries=json_retries, max_tokens=agent_max_tokens,
                           usage=usage)
    formula_briefs = _build_formula_briefs(hypotheses, available_terminals)
    brief_map = {brief.hypothesis_id: brief for brief in formula_briefs}

    # Each phase runs the 3 personas in parallel (independent within a phase).
    draft_lists = _run_parallel(
        [lambda a=agent: a.draft_formulas(formula_briefs) for agent in agents])
    formula_drafts: list[FormulaProposal] = [f for lst in draft_lists for f in lst]
    _annotate_diagnostics(formula_drafts, diagnose_fn)  # measure drafts before critique

    review_lists = _run_parallel([
        lambda a=agent: a.review_formulas([f for f in formula_drafts if f.agent_name != a.name])
        for agent in agents
    ])
    formula_reviews: list[FormulaReview] = [review for lst in review_lists for review in lst]

    def _revise_one(agent):
        own_formulas = [f for f in formula_drafts if f.agent_name == agent.name]
        own_reviews = [
            r for r in formula_reviews
            if any(f.formula_id == r.target_formula_id for f in own_formulas)
        ]
        return agent.revise_formulas(own_formulas, own_reviews, brief_map)
    revision_lists = _run_parallel([lambda a=agent: _revise_one(a) for agent in agents])
    formula_revisions: list[FormulaRevision] = [rev for lst in revision_lists for rev in lst]

    revised_formulas = [revision.revised_formula for revision in formula_revisions]
    _annotate_formula_parse_status(revised_formulas, available_terminals)
    _annotate_diagnostics(revised_formulas, diagnose_fn)  # re-measure after revision
    seed_pack = _select_seed_formulas(
        hypotheses=hypotheses,
        formulas=revised_formulas,
        client=client,
        model=model,
        target_count=target_count,
        retries=json_retries,
        max_tokens=moderator_max_tokens,
        usage=usage,
    )

    artifacts = {
        "formula_briefs.json": formula_briefs,
        "formula_drafts.json": formula_drafts,
        "formula_reviews.json": formula_reviews,
        "formula_revisions.json": formula_revisions,
        "seed_formula_pack.json": seed_pack,
        "seed_formulas.json": seed_pack.selected_formulas,
    }
    return seed_pack, artifacts
