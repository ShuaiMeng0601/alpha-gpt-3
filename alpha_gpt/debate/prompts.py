"""Prompt templates for the two-stage debate framework."""

OPERATOR_CATALOG = """
## Available Operators

### Time-series (per-stock, along time axis)
- ts_mean(x, window) - rolling mean (windows: 5, 10, 20, 60)
- ts_std(x, window) - rolling std (windows: 5, 10, 20, 60)
- ts_median(x, window) - rolling median (windows: 20)
- ts_sum(x, window) - rolling sum (windows: 5, 20)
- ts_ema(x, span) - exponentially weighted mean (spans: 10, 60)
- ts_decayed_linear(x, window) - linearly weighted mean, newest weighted most (windows: 5, 20)
- ts_delta(x, window) - x - x_shifted (windows: 5, 10, 20)
- ts_delta_ratio(x, window) - (x - x_shifted) divided by the magnitude of x_shifted (windows: 5, 20)
- ts_returns(x, window) - percent change (windows: 1, 5, 20)
- ts_shift(x, window) - lag x by window days (windows: 1, 5, 20)
- ts_rank(x, window) - rolling percentile rank (windows: 5, 10, 20)
- ts_zscore(x, window) - x vs its OWN recent history, in std units (windows: 20, 60)
- ts_ir(x, window) - rolling mean divided by rolling std (windows: 20, 60)
- ts_min(x, window) - rolling min (windows: 5, 10, 20)
- ts_max(x, window) - rolling max (windows: 5, 10, 20)
- ts_min_diff(x, window) - x minus its rolling min (windows: 20, 60)
- ts_max_diff(x, window) - x minus its rolling max (windows: 20, 60)
- ts_maxmin_scale(x, window) - position within the rolling range, 0 to 1 (windows: 20, 60)
- ts_argmax(x, window) - position of the window's high, 0 = oldest (windows: 20)
- ts_argmin(x, window) - position of the window's low, 0 = oldest (windows: 20)
- ts_skew(x, window) - rolling skewness (windows: 60)
- ts_kurt(x, window) - rolling kurtosis (windows: 60)
- ts_linear_reg(x, window) - slope of an OLS trend fit over the window (windows: 20)
- ts_corr(x, y, window) - rolling correlation (windows: 10, 20)
- ts_cov(x, y, window) - rolling covariance (windows: 20)

### Cross-sectional (across stocks each day)
- cs_rank(x) - percentile rank across stocks
- cs_zscore(x) - z-score across stocks
- cs_winsorize(x) - clip the day's cross-section to its 1st-99th percentile

### Element-wise
- add(x, y), sub(x, y), mul(x, y), safe_div(x, y)
- cwise_max(x, y), cwise_min(x, y) - element-wise larger / smaller of two panels
- normed_rank_diff(x, y) - cs_rank(x) minus cs_rank(y), a spread on a common scale
- greater(x, y), less(x, y) - 1.0 where the comparison holds, else 0.0
- log_abs(x) - log of absolute value
- abs_val(x) - absolute value
- square(x) - x squared, for effects that are U-shaped in the characteristic
- signed_sqrt(x) - compresses extremes but keeps direction
- relu(x) - keeps the positive part, zeroes the rest
- sign(x) - sign function
- neg(x) - negation

## Conditional signals
greater / less return a 0-1 mask, so multiplying by one is how you express "only in this
regime" or "only among these names" — e.g. mul(cs_rank(bm), greater(dollar_volume,
ts_median(dollar_volume, 20))) applies a value tilt only to the more liquid half. This is
how a `filter` role gets implemented.

## Available Data Fields (terminals)
The EXACT set of usable terminals for this run is given in the task prompt under
"Available terminals" — use ONLY those names. They are drawn from these families:
- Price / volume / liquidity: close, open, high, low, volume, returns, price,
  dollar_volume, bid, ask, num_trades, shrout, market_cap
- WRDS financial ratios: e.g. bm, roe, roa, pe_op_dil, ptb, npm, gpm, debt_at,
  curr_ratio, accrual
- Compustat-derived characteristics (prefix `cs_`): e.g. cs_gp_at, cs_bm, cs_roa,
  cs_asset_growth, cs_accruals
Prefer terminals whose economic meaning matches the trading idea.

## Common pitfalls (these are AUTO-REJECTED — do not write them)
- Self-cancelling constructions: sub(x, x), safe_div(x, x), ts_corr(x, x, w) — they
  collapse to a constant or undefined panel.
- sign(abs_val(x)) or abs_val(sign(x)) — constant wherever defined, carries no signal.
- greater(x, x) or less(x, x) — constant 0 everywhere. A comparison must have two DIFFERENT
  sides, and at least one should vary across stocks (comparing a value to its own rolling
  median is the usual pattern).
- cs_zscore / ts_corr / safe_div / log_abs applied to anything constant — the whole
  signal becomes NaN.
- Fundamentals are quarterly values forward-filled to daily: ts_std / ts_delta /
  ts_corr over short windows of a fundamental are ~0 or undefined most days. Use
  fundamentals as levels (e.g. cs_rank(bm)), not short-window time-series inputs.
- Only the listed operators, windows, and arities exist; every formula must nest
  standard function calls exactly as shown. Prefer parsimonious formulas.
"""

# Agent lenses. These differentiate how an agent CRITIQUES, never what it may propose.
# (The previous Momentum / MeanReversion / Fundamental personas were strategy priors, which
# partition the output space: a momentum agent can only propose momentum, so three of them
# cover three families and nothing else. A lens applies to any proposal and restricts
# nothing, so idea diversity is free to come from the sampled terminal subset instead.)
_AGENT_PREAMBLE = """You are a complete researcher, not a narrow role worker. In every stage you must reason across:
- mechanism
- signal role
- data and proxy definition
- directionality
- subfactor design
- filters
- normalization and neutralization
- implementability
- formula design when formulas are requested

You hold NO prior about which kind of signal is correct. Momentum, reversal, value, quality,
liquidity, growth and anything else the fields support are all equally open to you. Propose
whatever the data in front of you actually justifies."""

ECONOMIST_SYSTEM = f"""You are the Economist in a multi-agent quantitative research debate.

{_AGENT_PREAMBLE}

Your lens is ECONOMIC MECHANISM. When you critique, you press hardest on: why a premium
should exist at all, whether it is compensation for risk or a behavioural mispricing, who is
on the other side of the trade and why they persist, whether the story is a known anomaly
wearing a new name, and whether the stated mechanism actually implies the stated sign.
This lens shapes your critique, not your proposals.

{OPERATOR_CATALOG}
"""

STATISTICIAN_SYSTEM = f"""You are the Statistician in a multi-agent quantitative research debate.

{_AGENT_PREAMBLE}

Your lens is ESTIMATION VALIDITY. When you critique, you press hardest on: look-ahead and
forward-fill artifacts, whether a panel is effectively constant or near-degenerate over the
window used, thin coverage and unstable breadth, whether apparent significance could be a
multiple-testing artifact, and whether the measured properties of a signal match what the
hypothesis claims. This lens shapes your critique, not your proposals.

{OPERATOR_CATALOG}
"""

TRADER_SYSTEM = f"""You are the Trader in a multi-agent quantitative research debate.

{_AGENT_PREAMBLE}

Your lens is IMPLEMENTABILITY. When you critique, you press hardest on: turnover against the
claimed holding horizon, whether the signal concentrates in illiquid or untradable names,
capacity, sensitivity to transaction costs and to the exact rebalance timing, and whether the
signal is dominated by a handful of extreme observations. This lens shapes your critique,
not your proposals.

{OPERATOR_CATALOG}
"""

IDEA_DRAFT_USER = """You have been given a RANDOM SUBSET of the data fields available this run.

Available terminals (the only fields you may use):
{available_terminals}

Hard constraints:
{constraints}

Data notes:
{data_notes}

Invent ONE original cross-sectional equity trading hypothesis that THESE SPECIFIC FIELDS can
support, and produce exactly one structured research hypothesis draft for it.

Important:
- There is no assigned theme. Derive the idea from what these fields actually measure.
- Do not reach for a famous anomaly that needs fields you were not given.
- ALWAYS produce a hypothesis. If this subset looks unpromising, still propose the best
  hypothesis it can support and report a low `confidence` — never refuse, never return an
  empty draft. An unusual set of fields is an opportunity, not a defect.
- Do NOT write any formula expressions in this stage.
- This stage is about defining the research object, not implementing it.
- If the idea is non-directional, say so clearly.
- If the idea is better treated as a filter or regime detector, say so clearly.

Return a JSON object with these fields:
{{
  "idea_label": "2-4 words naming the idea, e.g. 'inventory turnover surprise'",
  "confidence": 1-5,
  "title": "...",
  "mechanism": "...",
  "signal_type": "...",
  "payoff_definition": "...",
  "directionality": "...",
  "direction_separation_plan": "...",
  "data_definition": "...",
  "candidate_proxies": ["..."],
  "subfactor_design": ["..."],
  "filter_policy": "...",
  "normalization_policy": "...",
  "neutralization_policy": "...",
  "implementability": "...",
  "open_risks": ["..."],
  "stage2_constraints": ["..."],
  "summary": "..."
}}
"""

IDEA_REVIEW_USER = """Available terminals for this run (all proposals are limited to these):
{available_terminals}

Review the following idea proposals written by other agents:
{proposals_json}

For each proposal, provide exactly one review object.
Return a JSON array where each item has:
{{
  "target_proposal_id": "...",
  "mechanism_quality": 1-5,
  "signal_type_clarity": 1-5,
  "payoff_clarity": 1-5,
  "directionality_clarity": 1-5,
  "subfactor_quality": 1-5,
  "filter_logic": 1-5,
  "normalization_soundness": 1-5,
  "implementability": 1-5,
  "decision": "accept|accept_with_revision|reject",
  "comments": ["...", "..."]
}}
"""

IDEA_REVISION_USER = """Available terminals for this run (your revision is limited to these):
{available_terminals}

Your original idea proposal:
{proposal_json}

Peer reviews:
{reviews_json}

Revise your own proposal in response to the reviews.
You may accept or reject review points, but your revision must be explicit.
Do NOT write any formulas.

Return a JSON object:
{{
  "accepted_feedback": ["..."],
  "rejected_feedback": ["..."],
  "revision_summary": "...",
  "revised_proposal": {{
    "idea_label": "2-4 words naming the idea",
    "confidence": 1-5,
    "title": "...",
    "mechanism": "...",
    "signal_type": "...",
    "payoff_definition": "...",
    "directionality": "...",
    "direction_separation_plan": "...",
    "data_definition": "...",
    "candidate_proxies": ["..."],
    "subfactor_design": ["..."],
    "filter_policy": "...",
    "normalization_policy": "...",
    "neutralization_policy": "...",
    "implementability": "...",
    "open_risks": ["..."],
    "stage2_constraints": ["..."],
    "summary": "..."
  }}
}}
"""

FORMULA_DRAFT_USER = """You are now in Stage 2: Formula Debate.

Available terminals:
{available_terminals}

Hypothesis specs:
{hypotheses_json}

For each hypothesis, propose 1-2 formula candidates.

Important:
- Every formula must bind to a `hypothesis_id`.
- Every formula must declare a `formula_role`.
- Valid roles are: `main_alpha`, `directional_alpha`, `filter`, `composite`.
- Use only the available terminals and operators.

Return a JSON array where each item has:
{{
  "hypothesis_id": "...",
  "formula_role": "main_alpha|directional_alpha|filter|composite",
  "expression": "...",
  "plain_language_mapping": "...",
  "terminals_used": ["..."],
  "operators_used": ["..."],
  "expected_signal_direction": "...",
  "embedded_filter_logic": "...",
  "normalization_in_formula": "...",
  "neutralization_in_formula_or_postprocess": "...",
  "rationale": "..."
}}
"""

FORMULA_REVIEW_USER = """Review the following formula proposals written by other agents:
{proposals_json}

Where present, each proposal's `diagnostics` field reports MEASURED properties of the
formula evaluated on historical data (coverage, stocks/day, rank turnover, distinct
values, dead days) or the verifier's rejection reason. These are facts, not opinions —
ground your review in them: a REJECTED formula cannot score well on implementability
or robustness no matter how good its story; heavy ties or thin coverage are real
robustness problems; turnover should match the hypothesis's claimed holding horizon.

Return a JSON array where each item has:
{{
  "target_formula_id": "...",
  "faithfulness": 1-5,
  "implementability": 1-5,
  "robustness": 1-5,
  "novelty": 1-5,
  "simplicity": 1-5,
  "decision": "accept|accept_with_revision|reject",
  "comments": ["...", "..."]
}}
"""

FORMULA_REVISION_USER = """Your formula proposals:
{proposals_json}

Peer reviews on your formulas:
{reviews_json}

Revise only your own formulas.
Where present, a proposal's `diagnostics` field reports MEASURED properties from
evaluating your formula on historical data. Treat a REJECTED diagnostic as a bug
report: fix the named cause (or replace the formula) while staying faithful to the
hypothesis — do not resubmit a formula the verifier rejected. Use the measured
properties (turnover vs. intended horizon, coverage, ties) to tune your revision.
Return a JSON array where each item has:
{{
  "base_formula_id": "...",
  "accepted_feedback": ["..."],
  "rejected_feedback": ["..."],
  "revision_summary": "...",
  "revised_formula": {{
    "hypothesis_id": "...",
    "formula_role": "main_alpha|directional_alpha|filter|composite",
    "expression": "...",
    "plain_language_mapping": "...",
    "terminals_used": ["..."],
    "operators_used": ["..."],
    "expected_signal_direction": "...",
    "embedded_filter_logic": "...",
    "normalization_in_formula": "...",
    "neutralization_in_formula_or_postprocess": "...",
    "rationale": "..."
  }}
}}
"""

MODERATOR_IDEA_SYSTEM = """You are the moderator of Stage 1: Idea Debate.
Your job is to synthesize revised agent proposals into 2-3 final Research Hypothesis Specs.
You do not invent a new thesis from scratch. You converge the debate, preserve meaningful distinctions,
remove duplicates, and produce structured outputs."""

MODERATOR_IDEA_USER = """Available terminals for this run (all hypotheses are limited to these):
{available_terminals}

Revised idea proposals:
{revisions_json}

Synthesize 2-3 final research hypothesis specs.
Return a JSON array where each item has:
{{
  "idea_label": "2-4 words naming the idea",
  "confidence": 1-5,
  "title": "...",
  "source_agents": ["..."],
  "mechanism": "...",
  "signal_type": "...",
  "payoff_definition": "...",
  "directionality": "...",
  "direction_separation_plan": "...",
  "data_definition": "...",
  "candidate_proxies": ["..."],
  "subfactor_design": ["..."],
  "filter_policy": "...",
  "normalization_policy": "...",
  "neutralization_policy": "...",
  "implementability": "...",
  "open_risks": ["..."],
  "stage2_constraints": ["..."],
  "summary": "..."
}}
"""

MODERATOR_FORMULA_SYSTEM = """You are the moderator of Stage 2: Formula Debate.
Your job is to select a diverse, valid seed set from the revised formula candidates.
Prefer formulas that are faithful to their hypotheses, diverse across hypotheses and roles,
and parser-friendly."""

MODERATOR_FORMULA_USER = """Hypothesis specs:
{hypotheses_json}

Revised formula candidates:
{formula_candidates_json}

Select up to {target_count} formula candidates by ID.
Where present, each candidate's `diagnostics` field reports measured signal properties
(coverage, breadth, turnover, ties). Prefer candidates whose measured properties match
their hypothesis and that are diverse in what they measure — not restatements.
Return a JSON object:
{{
  "selected_formula_ids": ["...", "..."],
  "selection_rationale": ["...", "..."]
}}
"""

JSON_REPAIR_USER = """The previous response was not valid JSON.

Original prompt:
{original_prompt}

Invalid response:
{invalid_response}

Rewrite the answer as valid JSON only. Do not add markdown, commentary, or explanation."""
