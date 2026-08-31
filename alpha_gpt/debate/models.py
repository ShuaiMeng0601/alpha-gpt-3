"""Structured artifacts for the two-stage debate framework."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any
import re


VALID_FORMULA_ROLES = {
    "main_alpha",
    "directional_alpha",
    "filter",
    "composite",
}

VALID_REVIEW_DECISIONS = {
    "accept",
    "accept_with_revision",
    "reject",
}


def slugify(value: str) -> str:
    """Convert an arbitrary string to a stable identifier fragment."""
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return value.strip("-") or "item"


def make_id(prefix: str, *parts: str) -> str:
    """Build a stable identifier from a prefix and a sequence of labels."""
    fragments = [slugify(prefix)]
    fragments.extend(slugify(part) for part in parts if part)
    return "-".join(fragments)


def to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses into JSON-serializable values."""
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


# --- payload coercion (model JSON is untrusted; normalize into our schema) -----

def coerce_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def coerce_list_of_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [coerce_str(item) for item in value if coerce_str(item)]
    text = coerce_str(value)
    return [text] if text else []


def coerce_score(value: Any) -> int:
    """Clamp a model-supplied review score into the valid 0-5 range."""
    try:
        return max(0, min(5, int(value)))
    except (TypeError, ValueError):
        return 0


@dataclass
class IdeaDebateBrief:
    """Stage-1 brief. The sampled terminal subset IS the seed — there is no assigned
    theme, so what the agents invent is bounded only by what these fields can measure."""
    available_terminals: list[str] = field(default_factory=list)
    operator_catalog: str = ""
    constraints: list[str] = field(default_factory=list)
    data_notes: list[str] = field(default_factory=list)


@dataclass
class IdeaProposal:
    proposal_id: str
    agent_name: str
    # Model-supplied name for the idea it invented. With no theme handed down, these
    # labels ARE the taxonomy: cluster them to count how many distinct directions a run
    # actually found, instead of assuming the directions we supplied up front.
    idea_label: str = ""
    # Model's own 1-5 rating of how promising the idea is. Recorded, never acted on:
    # low-confidence drafts are kept and gated on measured t-stat like everything else,
    # which leaves "does self-rated confidence predict out-of-sample IC?" answerable.
    confidence: int = 0
    title: str = ""
    mechanism: str = ""
    signal_type: str = ""
    payoff_definition: str = ""
    directionality: str = ""
    direction_separation_plan: str = ""
    data_definition: str = ""
    candidate_proxies: list[str] = field(default_factory=list)
    subfactor_design: list[str] = field(default_factory=list)
    filter_policy: str = ""
    normalization_policy: str = ""
    neutralization_policy: str = ""
    implementability: str = ""
    open_risks: list[str] = field(default_factory=list)
    stage2_constraints: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class IdeaReview:
    review_id: str
    reviewer_agent_name: str
    target_proposal_id: str
    mechanism_quality: int = 0
    signal_type_clarity: int = 0
    payoff_clarity: int = 0
    directionality_clarity: int = 0
    subfactor_quality: int = 0
    filter_logic: int = 0
    normalization_soundness: int = 0
    implementability: int = 0
    decision: str = "accept_with_revision"
    comments: list[str] = field(default_factory=list)


@dataclass
class IdeaRevision:
    revision_id: str
    agent_name: str
    base_proposal_id: str
    accepted_feedback: list[str] = field(default_factory=list)
    rejected_feedback: list[str] = field(default_factory=list)
    revision_summary: str = ""
    revised_proposal: IdeaProposal = field(default_factory=lambda: IdeaProposal("", ""))


@dataclass
class ResearchHypothesisSpec:
    hypothesis_id: str
    idea_label: str = ""
    confidence: int = 0
    title: str = ""
    source_agents: list[str] = field(default_factory=list)
    mechanism: str = ""
    signal_type: str = ""
    payoff_definition: str = ""
    directionality: str = ""
    direction_separation_plan: str = ""
    data_definition: str = ""
    candidate_proxies: list[str] = field(default_factory=list)
    subfactor_design: list[str] = field(default_factory=list)
    filter_policy: str = ""
    normalization_policy: str = ""
    neutralization_policy: str = ""
    implementability: str = ""
    open_risks: list[str] = field(default_factory=list)
    stage2_constraints: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class FormulaDebateBrief:
    hypothesis_id: str
    hypothesis_title: str = ""
    hypothesis_summary: str = ""
    available_terminals: list[str] = field(default_factory=list)
    operator_catalog: str = ""
    formula_constraints: list[str] = field(default_factory=list)


@dataclass
class FormulaProposal:
    formula_id: str
    hypothesis_id: str
    agent_name: str
    formula_role: str = "main_alpha"
    expression: str = ""
    plain_language_mapping: str = ""
    terminals_used: list[str] = field(default_factory=list)
    operators_used: list[str] = field(default_factory=list)
    expected_signal_direction: str = ""
    embedded_filter_logic: str = ""
    normalization_in_formula: str = ""
    neutralization_in_formula_or_postprocess: str = ""
    rationale: str = ""
    parseable: bool = False
    normalized_expression: str = ""
    parse_error: str = ""
    # Measured, return-blind signal properties (or the verifier's rejection reason),
    # computed on historical data between debate phases; "" when no verifier is wired in.
    diagnostics: str = ""


@dataclass
class FormulaReview:
    review_id: str
    reviewer_agent_name: str
    target_formula_id: str
    faithfulness: int = 0
    implementability: int = 0
    robustness: int = 0
    novelty: int = 0
    simplicity: int = 0
    decision: str = "accept_with_revision"
    comments: list[str] = field(default_factory=list)


@dataclass
class FormulaRevision:
    revision_id: str
    agent_name: str
    base_formula_id: str
    accepted_feedback: list[str] = field(default_factory=list)
    rejected_feedback: list[str] = field(default_factory=list)
    revision_summary: str = ""
    revised_formula: FormulaProposal = field(default_factory=lambda: FormulaProposal("", "", ""))


@dataclass
class SeedFormulaPack:
    pack_id: str
    selected_formulas: list[FormulaProposal] = field(default_factory=list)
    selection_rationale: list[str] = field(default_factory=list)
    traceability_map: dict[str, str] = field(default_factory=dict)
    dropped_formulas: list[FormulaProposal] = field(default_factory=list)
