"""Idea + alpha generators for the factory.

Three interchangeable strategies yielding `IdeaBatch`es. They all expose the same
``_one(terminals)`` method, so the parallel factory harness (``run_factory_parallel``)
thread-pools any of them across ideas identically:
  * RandomGenerator — samples valid random expression trees (no API; lets you
    stress-test the full loop/DB/portfolio pipeline for free at scale).
  * LLMGenerator   — one LLM call per idea: invents a diversified idea and emits
    a few concrete alpha expressions as JSON.
  * DebateGenerator — runs the full two-stage multi-agent debate per idea (~20 calls
    each, sequential WITHIN an idea but parallel ACROSS ideas) and returns the debate's
    selected seed formulas. This is how the debate scales to thousands of economically
    GROUNDED alphas through the same generate → filter → gate → portfolio path.
"""

import logging
import random
import threading
import time
from dataclasses import dataclass, field

from deap import gp

from alpha_gpt.llm import UsageMeter, extract_json_payload, _is_retryable, _completion_params
from alpha_gpt.debate.models import coerce_score
from alpha_gpt.debate.prompts import OPERATOR_CATALOG
from alpha_gpt.expr.primitives import create_primitive_set
from alpha_gpt.expr.terminal_subset import sample_subset

logger = logging.getLogger(__name__)


@dataclass
class IdeaBatch:
    idea: str
    expressions: list[str]
    hypothesis: str = ""
    source: str = "unknown"
    model: str | None = None
    # The terminal subset this idea was SHOWN (not the fields its formulas ended up using).
    # Persisted so a run can be audited for whether subsetting actually flattened terminal
    # usage, and so low-confidence draws can be checked for being systematically unusual.
    terminals: list[str] = field(default_factory=list)
    # Model's own 1-5 rating of the idea. Recorded, never used to filter.
    confidence: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ---------------------------------------------------------------------------
# Random generator (no API)
# ---------------------------------------------------------------------------

class RandomGenerator:
    def __init__(self, pset, alphas_per_idea: int = 5, min_depth: int = 1,
                 max_depth: int = 4, seed: int = 0, source: str = "random"):
        self.pset = pset
        self.alphas_per_idea = alphas_per_idea
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.source = source
        random.seed(seed)

    def generate(self, n_ideas: int):
        for i in range(n_ideas):
            exprs = []
            for _ in range(self.alphas_per_idea):
                try:
                    tree = gp.PrimitiveTree(
                        gp.genHalfAndHalf(self.pset, self.min_depth, self.max_depth)
                    )
                    exprs.append(str(tree))
                except Exception:
                    continue
            yield IdeaBatch(idea=f"{self.source}#{i}", expressions=exprs, source=self.source)


# Pure market-data terminals (no fundamentals). Restricting the generator to these is the
# price/volume-only CONTROL: any factor exposure a random book then earns can come ONLY from
# price/volume structure (short-term reversal, low-vol), never from fundamental characteristics
# (value, quality, profitability) — which are also where reporting-date look-ahead could hide.
PRICE_VOLUME_TERMINALS = ["open", "high", "low", "close", "volume"]


class RandomPriceGenerator(RandomGenerator):
    """RandomGenerator restricted to OHLCV terminals — no fundamentals. Builds its own
    price/volume-only primitive set, so every emitted expression is a function of raw market
    data alone. Run as ``--source random_price`` to isolate "is the random Sharpe just
    price-based factors?" from the fundamentals-are-factors story (and to sidestep any
    look-ahead that fundamental report dates could introduce)."""

    def __init__(self, alphas_per_idea: int = 5, min_depth: int = 1, max_depth: int = 4,
                 seed: int = 0, terminals: list[str] | None = None):
        pset = create_primitive_set(terminals or PRICE_VOLUME_TERMINALS)
        super().__init__(pset, alphas_per_idea, min_depth, max_depth, seed,
                         source="random_price")


# ---------------------------------------------------------------------------
# LLM generator (autonomous, diversified)
# ---------------------------------------------------------------------------

IDEA_GEN_SYSTEM = f"""You are a quantitative researcher who invents ORIGINAL cross-sectional
equity alpha ideas and expresses them as formulas. You are precise and only use the
operators and data fields provided.

{OPERATOR_CATALOG}
"""

IDEA_GEN_USER = """You have been given a RANDOM SUBSET of the data fields available this run.

Available terminals (use ONLY these names):
{terminals}

Invent ONE original cross-sectional equity trading idea that THESE SPECIFIC FIELDS can
support, then express it as {k} DISTINCT alpha formulas capturing different facets of it.

Rules:
- There is no assigned theme. Derive the idea from what these fields actually measure.
- Do not reach for a famous anomaly that needs fields you were not given.
- ALWAYS produce an idea. If this subset looks unpromising, still give the best idea it can
  support and report a low `confidence` — never refuse, never return empty expressions.
- Use ONLY the operators and terminals listed above.
- Write each formula in functional form, e.g. cs_rank(ts_delta(close, 5)).
- Every formula must be a concrete, valid expression (no pseudocode, no prose).
- Make the {k} formulas genuinely different from each other.

Return ONLY a JSON object:
{{"idea": "2-4 words naming the idea", "confidence": 1-5, "hypothesis": "...",
  "expressions": ["...", "..."]}}"""


class _PaidGeneratorMixin:
    """What both paid (LLM-backed) generators share: a budget stop and the subset seed.

    ``max_cost_usd`` (None = uncapped) is checked at the START of each ``_one`` call against
    the running token totals, so once the cap is crossed no NEW paid work begins. In-flight
    work still completes: with ``gen_workers`` concurrent calls the overshoot is bounded by
    ~gen_workers x cost-per-idea."""

    max_cost_usd: float | None = None

    def sample_subset(self) -> list[str]:
        """Draw this idea's terminal menu from the run's full field library.

        Sampling lives on the generator (not the caller) because the generator owns both the
        field library and the seeded RNG, so a run's draws stay reproducible from its seed.
        """
        return sample_subset(self.available_terminals,
                             size=getattr(self.config, "terminal_subset_size", None),
                             rng=self._rng)

    def _budget_exhausted(self) -> bool:
        if self.max_cost_usd is None:
            return False
        from alpha_gpt.factory.run import estimate_cost  # local import: avoids import cycle
        with self._lock:
            cost = estimate_cost(self.total_prompt_tokens, self.total_completion_tokens,
                                 model=self.model)
            if cost < self.max_cost_usd:
                return False
            if not getattr(self, "_budget_logged", False):
                self._budget_logged = True
                logger.warning("cost budget reached (est. $%.2f >= cap $%.2f); "
                               "skipping all remaining generation", cost, self.max_cost_usd)
        return True


class LLMGenerator(_PaidGeneratorMixin):
    def __init__(self, client, model: str, available_terminals: list[str],
                 alphas_per_idea: int = 3, temperature: float = 0.9, seed: int = 0,
                 max_cost_usd: float | None = None, config=None):
        self.client = client
        self.model = model
        self.available_terminals = available_terminals
        self.config = config
        self.alphas_per_idea = alphas_per_idea
        self.temperature = temperature
        self._rng = random.Random(seed)
        self.max_cost_usd = max_cost_usd
        # Running totals across ALL calls (incl. parse failures) for honest cost
        # accounting. Lock-protected so it's safe under a thread pool.
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.n_calls = 0
        self._lock = threading.Lock()

    def _one(self, terminals: list[str] | None = None) -> IdeaBatch | None:
        if self._budget_exhausted():
            return None
        subset = list(terminals) if terminals else self.sample_subset()
        user = IDEA_GEN_USER.format(k=self.alphas_per_idea, terminals=", ".join(subset))
        pt = ct = 0                   # successful attempt's tokens (per-batch attribution)
        billed_pt = billed_ct = 0     # tokens billed across ALL attempts (honest totals)
        n_api_calls = 0
        payload = None
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": IDEA_GEN_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    # 4000 (was 1400) so idea+hypothesis+k formulas can't truncate mid-JSON;
                    # helper sends max_completion_tokens / drops temperature for GPT-5 models.
                    **_completion_params(self.model, temperature=self.temperature, max_tokens=4000),
                )
                n_api_calls += 1
                usage = getattr(resp, "usage", None)
                pt = int(getattr(usage, "prompt_tokens", 0) or 0)
                ct = int(getattr(usage, "completion_tokens", 0) or 0)
                billed_pt += pt   # count this billed call even if JSON parsing below fails
                billed_ct += ct
                payload = extract_json_payload(resp.choices[0].message.content)
                break
            except Exception as exc:
                if not _is_retryable(exc):
                    break  # permanent error (400 bad param / 401 auth / 402 no credits): fail fast
                time.sleep(1.0 + attempt)  # back off (handles transient rate limits)
        # Honest cost accounting: totals include EVERY billed attempt (incl. ones whose
        # JSON failed to parse), while the IdeaBatch carries only the successful
        # attempt's tokens for per-expression attribution.
        with self._lock:
            self.n_calls += max(1, n_api_calls)
            self.total_prompt_tokens += billed_pt
            self.total_completion_tokens += billed_ct
        if not isinstance(payload, dict):
            return None
        exprs = [e for e in payload.get("expressions", []) if isinstance(e, str) and e.strip()]
        if not exprs:
            return None
        return IdeaBatch(
            idea=str(payload.get("idea", ""))[:500],
            hypothesis=str(payload.get("hypothesis", ""))[:1000],
            expressions=exprs,
            source="llm",
            model=self.model,
            terminals=subset,
            confidence=coerce_score(payload.get("confidence")),
            prompt_tokens=pt,
            completion_tokens=ct,
        )

    def generate(self, n_ideas: int):
        for _ in range(n_ideas):
            batch = self._one(self.sample_subset())
            if batch is not None:
                yield batch


# ---------------------------------------------------------------------------
# Debate generator (multi-agent debate as the factory's generator)
# ---------------------------------------------------------------------------

# Fallback per-debate token estimate, used ONLY if the provider returned no usage data for
# any call in a debate (rare). Real usage is metered per call via llm.UsageMeter; the flat
# estimate keeps cost telemetry over-reporting rather than silently zero in that edge case.
_DEBATE_PROMPT_TOKENS = 30_000
_DEBATE_COMPLETION_TOKENS = 12_000


class DebateGenerator(_PaidGeneratorMixin):
    """Runs the two-stage multi-agent debate per idea, behind the same ``_one(terminals)``
    interface as :class:`LLMGenerator` so it drops straight into ``run_factory_parallel``'s
    thread pool. Each call argues out a hypothesis and returns the debate's selected seed
    formulas as an :class:`IdeaBatch` — so ``--source debate`` produces many economically
    grounded alphas through the existing factory → filter → gate → portfolio pipeline.

    Each debate is seeded with its OWN random subset of the terminal menu, and no theme:
    what the agents invent is bounded only by what those fields can measure.

    The debate is sequential *within* one idea (each agent responds to the last), but ideas
    are independent, so the pool runs ``gen_workers`` debates concurrently — the cost is
    API latency × calls / workers, not any design limit.
    """

    def __init__(self, client, model: str, available_terminals: list[str], config, seed: int = 0,
                 pset=None, panels=None, out_dir: str | None = None, diag_subsample: int = 3,
                 max_cost_usd: float | None = None):
        self.client = client
        self.model = model
        self.available_terminals = available_terminals
        self.config = config
        self._rng = random.Random(seed)
        self.max_cost_usd = max_cost_usd
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.n_calls = 0
        self._lock = threading.Lock()
        self._idea_counter = 0
        # Artifact dir (draft/review/revision JSON per idea) — only when configured.
        self.out_dir = out_dir if getattr(config, "save_debate_artifacts", False) else None
        # Return-blind diagnostics for drafted formulas (see verifier.diagnose_expression),
        # evaluated on a date-subsampled copy of the panels so a debate-time verification
        # costs ~1/diag_subsample of a full evaluation. Optional: without pset/panels the
        # debate runs exactly as before.
        self._diagnose_fn = None
        if pset is not None and panels:
            step = max(1, int(diag_subsample or 1))
            sub = {name: df.iloc[::step] for name, df in panels.items()}
            max_depth = getattr(config, "verify_max_depth", 20)
            from alpha_gpt.factory.verifier import diagnose_expression
            # Honest labelling: on every-k-th-day data, ts windows span ~k x the calendar
            # days and "rank turnover" is per k-day step, so say so in the feedback line.
            note = f" (measured on every-{step}th-day data)" if step > 1 else ""

            def _diagnose(expr):
                line = diagnose_expression(expr, pset, sub, max_depth=max_depth)
                return line + note if line.startswith("OK") else line
            self._diagnose_fn = _diagnose

    def _save_artifacts(self, label: str, terminals: list[str], idea_idx: int,
                        stage1: dict, stage2: dict) -> None:
        """Write one JSON per idea with the full debate trace (drafts/reviews/revisions).

        The filename carries the model's OWN label for the idea, so browsing the artifact
        directory shows the taxonomy a run discovered rather than the one we handed it.
        Best-effort: an artifact write must never kill a paid debate."""
        if self.out_dir is None:
            return
        try:
            import json as _json
            import os as _os
            from alpha_gpt.debate.models import to_jsonable
            debate_dir = _os.path.join(self.out_dir, "debate")
            _os.makedirs(debate_dir, exist_ok=True)
            safe_label = "".join(c if c.isalnum() else "_" for c in label)[:40] or "unlabelled"
            path = _os.path.join(debate_dir, f"idea_{idea_idx:04d}_{safe_label}.json")
            with open(path, "w") as f:
                _json.dump(to_jsonable({"idea_label": label, "terminals": terminals,
                                        **stage1, **stage2}), f, indent=2)
        except Exception:
            logger.warning("failed to save debate artifacts for %r", label, exc_info=True)

    def _record_usage(self, meter: UsageMeter, completed: bool) -> tuple[int, int]:
        """Fold one debate's REAL billed tokens into the running totals. Falls back to the
        flat estimate only if a COMPLETED debate somehow surfaced no usage data (telemetry
        must over-report, never silently zero). Returns (pt, ct) for batch attribution."""
        pt, ct = meter.prompt_tokens, meter.completion_tokens
        if completed and pt == 0 and ct == 0:
            pt, ct = _DEBATE_PROMPT_TOKENS, _DEBATE_COMPLETION_TOKENS
        with self._lock:
            self.n_calls += meter.calls or (1 if completed else 0)
            self.total_prompt_tokens += pt
            self.total_completion_tokens += ct
        return pt, ct

    def _one(self, terminals: list[str] | None = None) -> "IdeaBatch | None":
        if self._budget_exhausted():
            return None
        # Imported here (not at module load) to keep the factory→debate dependency lazy.
        from alpha_gpt.debate.moderator import run_formula_debate, run_idea_debate
        subset = list(terminals) if terminals else self.sample_subset()
        with self._lock:
            self._idea_counter += 1
            idea_idx = self._idea_counter
        meter = UsageMeter()  # real billed tokens for THIS debate (agents + moderator)
        try:
            hypotheses, stage1 = run_idea_debate(
                subset, self.client, self.model, config=self.config, usage=meter)
            seed_pack, stage2 = run_formula_debate(
                hypotheses, subset, self.client, self.model,
                config=self.config, diagnose_fn=self._diagnose_fn, usage=meter)
        except Exception:
            # A debate that died mid-way still PAID for every call it made — record them.
            self._record_usage(meter, completed=False)
            logger.warning("debate on %s crashed; its %d billed calls are still counted",
                           subset, meter.calls, exc_info=True)
            return None
        label = hypotheses[0].idea_label if hypotheses else ""
        self._save_artifacts(label, subset, idea_idx, stage1, stage2)
        # Cost accounting BEFORE the empty-pack early return: a completed debate paid its
        # calls whether or not any formula survived selection.
        pt, ct = self._record_usage(meter, completed=True)
        exprs = [f.expression for f in seed_pack.selected_formulas if f.expression]
        if not exprs:
            return None
        hypo = hypotheses[0].summary if hypotheses else ""
        return IdeaBatch(idea=label or f"unlabelled#{idea_idx}", expressions=exprs,
                         hypothesis=hypo, source="debate", model=self.model,
                         terminals=subset,
                         confidence=hypotheses[0].confidence if hypotheses else 0,
                         prompt_tokens=pt, completion_tokens=ct)

    def generate(self, n_ideas: int):
        for _ in range(n_ideas):
            batch = self._one(self.sample_subset())
            if batch is not None:
                yield batch


# ---------------------------------------------------------------------------
# GP generator (genetic programming as a factory source)
# ---------------------------------------------------------------------------

class GPGenerator:
    """Evolve alpha expressions with genetic programming, behind the factory interface.

    Wraps ``expr.engine.run_gp`` so GP runs through the SAME generate -> verify ->
    evaluate -> filter -> backtest path as the random / LLM / debate sources. That shared
    substrate is what makes the debate-vs-GP comparison apples-to-apples (identical eval,
    verify, store, and metrics — only the generator differs).

    Unlike the LLM/debate generators GP is not per-idea: it evolves ONE population over
    ``gp_generations`` and returns its best individuals. It therefore implements
    ``generate(n_ideas)`` directly (like :class:`RandomGenerator`) and is dispatched off
    the factory's non-pooled path. ``n_ideas`` controls how many top individuals to keep.
    GP makes no API calls, so the token/cost counters are zero. (This is the unseeded GP
    baseline; seeding GP with debate/LLM formulas is not currently wired into the factory.)
    """

    def __init__(self, pset, train, terminal_names: list[str], config, seed: int = 0):
        self.pset = pset
        self.train = train
        self.terminal_names = terminal_names
        self.config = config
        self.seed = seed
        self.model = None
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.n_calls = 0

    def generate(self, n_ideas: int):
        from alpha_gpt.expr.engine import run_gp
        results, _log = run_gp(
            seed_trees=None, data=self.train,
            population_size=self.config.gp_population,
            generations=self.config.gp_generations,
            crossover_prob=self.config.gp_crossover,
            mutation_prob=self.config.gp_mutation,
            tournament_size=self.config.gp_tournament_size,
            max_depth=self.config.gp_max_depth,
            random_seed=self.seed, terminal_names=self.terminal_names,
            hof_size=max(n_ideas, self.config.gp_population), verbose=False,
        )
        exprs = [r["expression"] for r in results if r.get("expression")]
        # One batch carrying all evolved expressions; the factory flattens + dedups them.
        yield IdeaBatch(idea="gp", expressions=exprs, hypothesis="genetic programming",
                        source="gp")
