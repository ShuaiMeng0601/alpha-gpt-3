# Alpha-GPT Debate Framework

## Title and Motivation

Alpha-GPT is intended to use a genuine multi-agent debate process to transform a vague natural-language trading idea into a small set of strong, testable seed formulas. The earlier framing of `raw idea -> direct seed formula generation` is useful for quick ideation, but it is not a true debate framework. In that setup, agents mostly jump straight to expressions before the research hypothesis itself has been clarified, challenged, or refined.

The design goal of the new framework is to make the agents debate the idea before they debate the formula. Agents should first converge on what the idea means, what kind of signal it is, what payoff it is trying to capture, whether it is directional, whether it should instead act as a filter, what data proxies define it, and what constraints or filters are needed to make it tradable. Only after that convergence should they debate how to express the agreed hypotheses as formulas.

## Framework Overview

The framework has two stages:

1. `Stage 1: Idea Debate`
2. `Stage 2: Formula Debate`

Each stage uses the same high-level structure:

1. Three full research agents independently produce proposals.
2. The agents peer review and judge one another's proposals.
3. Each agent revises its own work in response to critique.
4. A moderator converges the debate into a smaller set of outputs.

The output of Stage 1 is `2-3 Research Hypothesis Specs`. The output of Stage 2 is a validated `Seed Formula Pack` whose formulas are traceable to those hypothesis specs.

## Agent Design

The three agents are complete researchers, not narrow role workers. Each agent performs the full reasoning process in each stage. No agent is limited to only mechanism reasoning, only data definition, or only formula writing. Instead, each agent is expected to produce a complete research view from its own prior and style.

The recommended identities remain style-based rather than task-based:

- `Momentum Agent`
- `Mean-Reversion Agent`
- `Fundamental Agent`

These identities give the agents different priors and biases, which is what makes the debate productive. All three agents independently analyze the same raw idea, critique one another, and revise in response.

The moderator is responsible for orchestration and convergence. The moderator does not invent the core research content on its own. Its job is to synthesize the results of the debate, remove duplication, preserve meaningful disagreement when needed, and compress the process into a manageable number of structured outputs.

## Stage 1: Idea Debate

### Purpose

Stage 1 converts a vague natural-language idea into `2-3 Research Hypothesis Specs`. It does not generate formulas. It defines what the research object is and how it should later be translated into formulas.

### Required Analysis Dimensions

Each agent must independently produce a complete hypothesis draft that addresses the following dimensions in general research terms:

- `Mechanism understanding`
  What economic, behavioral, structural, or trading mechanism could make this idea work? Under what conditions should it hold, and when should it fail?

- `Signal type classification`
  Is the idea primarily a cross-sectional alpha, a time-series alpha, a regime detector, a filter, an entry/exit timing condition, a risk overlay, or a composite concept that mixes several roles?

- `Payoff definition`
  What is the expected source of returns? Is the idea supposed to predict direction, relative performance, volatility expansion, trend persistence, mean reversion, drawdown avoidance, or something else?

- `Directionality`
  Does the idea itself imply a directional forecast, or is it non-directional? If it is non-directional, it must not be treated as a directional alpha by default.

- `Direction separation plan`
  If the idea is not inherently directional, how should direction be separated? The agent must specify whether the signal should become a parent hypothesis plus directional children, or whether it should be treated as a filter that gates other directional signals.

- `Data definition and observable proxies`
  How should the natural-language concepts be mapped into measurable objects? The agent must define candidate proxies, the expected data family, and any timing, smoothing, lagging, or aggregation assumptions needed for later implementation.

- `Subfactor design`
  If the raw idea is too coarse, how should it be decomposed into subfactors? The decomposition may separate direction, strength, confirmation, state, or noise control, but each subfactor must have a clear role.

- `Filter need and filter policy`
  Does the idea require filtering to reduce false positives, avoid regime mismatch, or improve tradability? If so, under what conditions should the signal be filtered and why?

- `Normalization and neutralization policy`
  Should the idea use ranking, z-scoring, volatility adjustment, winsorization, industry neutralization, size neutralization, or other normalization and neutralization steps? The agent must explain whether these belong to the signal itself or to a later portfolio layer.

- `Implementability under current data`
  Is the idea directly supported by current data, only partially supported through proxies, or not currently supported? The agent must identify data gaps and timing risks before the idea can move forward.

### Debate Rounds

Stage 1 follows a fixed sequence:

1. `Independent proposal`
   Each of the three agents submits a full `Research Hypothesis Spec Draft`.

2. `Peer review / judging`
   Each agent reviews the other two drafts and scores them on mechanism quality, signal-type clarity, payoff clarity, directionality clarity, subfactor quality, filter logic, normalization and neutralization soundness, and implementability.

3. `Self-revision`
   Each agent revises only its own draft in response to critique. Revisions must explicitly address accepted and rejected review points.

4. `Moderator synthesis`
   The moderator converges the revised drafts into `2-3 Research Hypothesis Specs`.

### Final Artifact

The final Stage 1 artifact is a structured `Research Hypothesis Spec`, not a sentence and not a formula. Each spec should contain, in concept:

- a core mechanism
- signal type
- payoff definition
- directionality
- a direction separation plan when needed
- data and proxy definitions
- subfactor design
- filter policy
- normalization policy
- neutralization policy
- implementability status
- open risks and Stage 2 constraints

## Stage 2: Formula Debate

### Purpose

Stage 2 translates the agreed `Research Hypothesis Specs` into seed formulas. Formula generation starts only after Stage 1 has converged. Stage 2 is about expression design, not mechanism discovery.

### Debate Rounds

Stage 2 also follows a fixed sequence:

1. `Independent formula proposal`
   Each agent independently proposes formula implementations for each hypothesis spec.

2. `Peer review / judging`
   Each agent reviews the other agents' formulas for faithfulness, implementability, robustness, novelty, simplicity, and parser friendliness.

3. `Self-revision`
   Each agent revises only its own formulas in response to critique.

4. `Moderator seed selection`
   The moderator selects a small, diverse, validated seed pool.

### Formula Roles

A formula is not always the same kind of object. Stage 2 should explicitly allow formulas to play different roles:

- `Main alpha`
- `Directional alpha`
- `Filter`
- `Composite implementation`

Every final formula must be traceable back to a specific hypothesis. If a hypothesis describes a non-directional regime or filter, the corresponding formula should preserve that role rather than being misrepresented as a standalone directional alpha.

## Artifacts and Traceability

The framework should be organized around a small set of conceptual artifacts:

- `Idea Debate Brief`
  The brief built from the raw natural-language idea plus the current data, operator, and implementation constraints.

- `Research Hypothesis Spec`
  The structured output of Stage 1 that defines what the idea means and how it should be studied.

- `Formula Debate Brief`
  The Stage 2 brief derived from a specific hypothesis spec and the allowable implementation constraints.

- `Formula Proposal`
  A candidate expression together with its signal role, rationale, and relation to the underlying hypothesis.

- `Seed Formula Pack`
  The final selected set of formulas after proposal, peer review, revision, and moderator convergence.

The traceability chain should be explicit:

`raw idea -> hypothesis spec -> formula proposal -> selected seed formula`

This traceability is important for debugging, interpretability, auditability, and future explanation of why a final seed exists.

## Acceptance Criteria

The new framework is successful if the following are true:

- Stage 1 outputs `Research Hypothesis Specs`, not formulas.
- Non-directional ideas can be classified as filters or regime signals instead of being forced into directional alpha form.
- Each hypothesis spec defines signal role, payoff, directionality, filters, normalization, neutralization, and implementability.
- Stage 2 debates formulas only after Stage 1 has converged on the underlying research objects.
- Final formulas are diverse, parser-ready, and traceable to specific hypotheses.

