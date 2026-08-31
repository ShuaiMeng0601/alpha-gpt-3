"""Configuration for Alpha-GPT 3.0 pipeline."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Data
    data_dir: str = "data/panels"
    aux_dir: str = "data/aux"  # non-terminal panels (e.g. sector) kept out of data_dir

    # Universe construction (point-in-time, survivorship-free)
    universe_size: int = 1500       # target names selected at each annual rebalance
    # NOTE: reserved — select_universe currently hardcodes a prior-calendar-year mean-cap
    # rank with an annual rebalance; these two knobs are NOT yet wired into selection.
    lookback_days: int = 252        # (reserved) trailing window for market-cap ranking
    rebalance_freq: str = "YS"      # (reserved) pandas offset alias; YS = year-start
    include_reits: bool = True      # REITs are common equity; keep unless excluded
    min_price: float = 5.0          # point-in-time price floor: drop sub-$5 (penny) stock-days

    # Terminal management.
    # terminal_screen controls how the candidate panels are narrowed into the per-run
    # terminal menu:
    #   "breadth_only" — keep EVERY field that clears the data-quality floor (coverage +
    #                    breadth); no IC ranking, no cap. Debate-native default: the model
    #                    reasons about which features to use, so we don't pre-screen by a
    #                    univariate heuristic or starve it with a cap.
    #   "ic_capped"    — legacy GP behaviour: stable core + top-|IC| extras up to
    #                    max_terminals (keeps the GP search space bounded).
    terminal_screen: str = "breadth_only"
    max_terminals: int = 40         # cap used only when terminal_screen == "ic_capped"
    min_coverage: float = 0.20      # min fraction of non-NaN cells for a field to qualify
    min_breadth: int = 100          # min stocks/day for a field to qualify as a terminal

    # Expression verification
    verify_max_depth: int = 20      # max parse-tree depth a stored alpha may have. Raised
                                    # from the old GP-era 8 so legitimately deep debate
                                    # formulas aren't rejected; still a sanity bound (it is
                                    # the only structural check applied to a stored alpha).

    # LLM provider. make_client routes by the model id: a provider-prefixed id (contains '/',
    # e.g. 'deepseek/deepseek-v3.2') goes to OpenRouter; a bare id (e.g. 'gpt-5.4-mini') goes to
    # OpenAI direct (OPENAI_API_KEY). Switch models without editing source via `LLM_MODEL` in .env.
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = os.getenv("LLM_MODEL", "deepseek/deepseek-v3.2")
    # NOTE: transient-error backoff (rate limits / 5xx) is handled in llm.call_json via its
    # api_retries param (default 5) — lets us push request concurrency without 429 storms.

    # GP search
    gp_population: int = 50
    gp_generations: int = 10
    gp_crossover: float = 0.7
    gp_mutation: float = 0.2
    gp_tournament_size: int = 3
    gp_max_depth: int = 6

    # Debate
    debate_rounds: int = 2
    idea_hypothesis_target: int = 3
    seed_formula_target: int = 8
    debate_json_retries: int = 2
    save_debate_artifacts: bool = True
    # How many terminals each idea is shown (None = ~sqrt(menu size), the point where two
    # independent draws share ~1 field on average). This is the diversity seed: every idea
    # seeing the whole menu is what let the model's prior concentrate on a few famous
    # fields and made independent ideas converge. See expr/terminal_subset.py.
    terminal_subset_size: int | None = None
    # Output-token budgets per call type. Sized so responses finish naturally
    # (finish_reason="stop") instead of being truncated mid-JSON (finish_reason="length"),
    # which truncated formula lists and forced wasted repair retries. You only pay for
    # tokens actually generated, so a generous cap costs nothing on calls that don't need it.
    # NOTE: on OpenAI reasoning models (gpt-5*/o*) this budget is max_completion_tokens and
    # is SHARED with invisible reasoning tokens — 4000 proved too small in production (every
    # moderator synthesis truncated, failed JSON parse, re-billed a repair retry that
    # truncated again, then fell back to dumb first-N selection). Hence the large defaults.
    agent_max_tokens: int = 12000     # one debate agent's JSON turn (draft/review/revise)
    moderator_max_tokens: int = 16000  # moderator synthesis / selection (3 full hypothesis
                                       # specs as JSON + reasoning overhead)

    # Evaluation. Two splits: ONE in-sample block (selection + orientation) and a held-out
    # test block. `is_end` is the last in-sample date; test is everything after it. The LLM
    # never fits on data, so a single long in-sample block gives a higher-power IC t-stat for
    # the selection gate than the old train/val split-and-sign-check.
    top_k: int = 5
    is_end: str = "2020-12-31"

    @property
    def openrouter_api_key(self) -> str:
        key = os.getenv("OPENROUTER_API_KEY", "")
        if not key:
            raise ValueError("OPENROUTER_API_KEY not set. Add it to .env file.")
        return key

    @property
    def openai_api_key(self) -> str:
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY not set. Add it to .env file.")
        return key
