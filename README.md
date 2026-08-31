# Alpha-GPT 3.0 (Work in Progress)

> **New here, or picking this up from someone else? Read [HANDOFF.md](HANDOFF.md) first.**
> It lists the committed run artifacts ($108 of API spend, 496 debate transcripts) so you can
> reproduce every result offline, and records where the work currently stands.

Multi-agent LLM debate framework for discovering quantitative trading alphas. The whole
thing runs from **one command** — `python run.py` — as a single staged pipeline you can
watch stream stage by stage into a self-contained run folder.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=sk-or-your-key-here
```

## Data

Panel data lives in `data/panels/` as parquet files (one per field, each a date × stock
DataFrame). To build them from raw CRSP/Compustat CSVs (and optional OSAP signals):

```bash
python scripts/prepare_data.py          # CRSP OHLCV panels + universe
python scripts/ingest_fundamentals.py   # Compustat -> cs_* characteristic panels
python scripts/ingest_osap.py           # Open Source Asset Pricing signals -> panels
python scripts/verify_data.py           # data-quality checks
```

## Running

One entry point, one pipeline. Pick a **generation source** and how many ideas to
generate; every run writes a timestamped, self-contained folder under `outputs/`.

```bash
# Full multi-agent debate run (uses your OpenRouter key)
python run.py --source debate --n 200

# Free, no-API end-to-end stress test (great for debugging the whole pipeline)
python run.py --source random --n 50

# Genetic-programming baseline (free, no LLM)
python run.py --source gp --n 100

# Single-shot LLM (one call per idea)
python run.py --source llm --n 200 --alphas-per-idea 5

# Tune the survivor filter / combination methods
python run.py --source debate --n 200 --min-ic 0.01 --methods equal ic_weighted
```

The pipeline runs five stages with a progress bar each, then prints an out-of-sample
comparison table:

**load → generate/verify/evaluate → filter → re-eval on test + backtest → report**

Each run folder (`outputs/run_<timestamp>/`) contains:

- `pipeline_config.json` — exact parameters of the run
- `pipeline.log` — full log (tee'd from the console)
- `alphas.db` — the run's own alpha database (SQLite)
- `factory_summary.json` — generation funnel + token cost
- `survivors.csv` — alphas that passed the validation filter
- `portfolio_report.md` / `portfolio_results.json` — out-of-sample comparison table
- `portfolio_equity_curves.png` — cumulative returns vs the VW market benchmark
- `pipeline_summary.json` — everything tied together

### Ablation across sources

The four sources (`random`, `gp`, `llm`, `debate`) are the ablation arms — is the
economic reasoning load-bearing, or do the mechanical baselines do just as well? Run each,
then tabulate the arms:

```bash
python run.py --source random --n 200
python run.py --source gp     --n 200
python run.py --source llm    --n 200
python run.py --source debate --n 200
python run.py compare                    # aggregate outputs/*/pipeline_summary.json by source
```

`compare` writes `outputs/comparison/summary_table.md` and `comparison.json` with the
mean ± std (across runs) of each source's headline equal-weight portfolio Sharpe/IC/return.

### Rebuild a portfolio from an existing DB

To try different survivor filters without regenerating alphas, run the portfolio stage in
isolation against a previous run's `alphas.db`:

```bash
python run.py portfolio --db outputs/run_<ts>/alphas.db --min-ic 0.01 --top-k 50
```

Selection uses validation metrics (stored in the DB); the backtest is on the held-out test
split, so the result stays out-of-sample.

## Tests

```bash
pytest            # config in pyproject.toml; `python -m pytest` also works
```

Fast, deterministic tests (no network, no real data files, no LLM/GP runs — all on tiny
in-memory synthetic panels via `tests/conftest.py`). Coverage spans the data pipeline
(universe selection, point-in-time ffill, train/val/test splits), metrics, the backtester
(an exhaustive correctness suite cross-checked against an independent reference), the
factory (DB, verifier, generators), the terminal screen, portfolio construction, and the
multi-agent debate orchestration. Every bug fixed in this codebase has a named
`*_regression` test guarding it — e.g. the CRSP `SecurityHdrFlg` data drop, forward-return
look-ahead, the fast-IC NaN-mask bias, and the earnings-growth sign.

## Architecture

One pipeline, one entry point. `run.py` parses settings and calls `run_pipeline`; every
domain lives in a focused subpackage under `alpha_gpt/`.

```
run.py                 # THE entry point: python run.py --source ... --n ...
alpha_gpt/
  # ── core ──────────────────────────────────────────────────────────
  config.py            # Config dataclass (.env loading) — the one place for knobs
  llm.py               # OpenRouter client + robust JSON calls (make_client, call_json)
  pipeline.py          # the pipeline: generate -> filter -> portfolio backtest -> report
  experiment.py        # run setup: load_and_split + compute_vw_benchmark
  compare.py           # cross-source aggregation over outputs/*/pipeline_summary.json

  # ── data ──────────────────────────────────────────────────────────
  data/loader.py       # load parquet panels, train/val/test split, PIT ffill helper
  data/universe.py     # point-in-time, survivorship-free annual universe selection
  data/coverage.py     # per-panel coverage/breadth report + terminal qualification
  data/fundamentals.py # Compustat -> cs_* characteristic panels
  data/osap.py         # Open Source Asset Pricing signals -> panels

  # ── alpha expression language + GP ────────────────────────────────
  expr/alpha_ops.py        # 35 curried operators — the single operator registry
  expr/primitives.py       # build a DEAP PrimitiveSet from that registry
  expr/seed_injector.py    # parse LLM expressions -> DEAP trees
  expr/engine.py           # eval_expr + DEAP GP evolution loop
  expr/terminal_selection.py  # screen candidate terminals (|IC| + breadth)

  # ── multi-agent debate ────────────────────────────────────────────
  debate/agents.py     # 3 style agents (Momentum, MeanReversion, Fundamental)
  debate/moderator.py  # two-stage (idea -> formula) debate orchestration
  debate/prompts.py    # prompt templates + the operator catalog shown to the model
  debate/models.py     # dataclasses + payload coercion helpers

  # ── generation engine ─────────────────────────────────────────────
  factory/generators.py        # random / GP / LLM / debate expression generators
  factory/verifier.py          # robust expression verification gate
  factory/run.py, parallel.py  # generate -> verify -> evaluate -> store loop
  factory/fast_metrics.py      # vectorized IC / Sharpe proxies for fast ranking
  factory/db.py                # SQLite store for generated alphas

  # ── evaluation + portfolio ────────────────────────────────────────
  evaluate/metrics.py      # IC, ICIR, turnover
  evaluate/backtester.py   # long-short quintile backtester
  evaluate/visualize.py    # equity-curve + GP-evolution plots
  evaluate/explainer.py    # LLM explains alphas in plain language
  evaluate/neutralize.py   # cross-sectional / sector neutralization helpers
  portfolio/construct.py   # combine survivors (equal / ic_weighted / orthogonalized)

scripts/                 # offline data build (run once): prepare_data, ingest_*, verify_data
```

**Adding an operator** is one entry in `expr/alpha_ops.py` — the primitive set and the LLM
parser both derive from it automatically. **Ablating the reasoning** is the `--source` knob
(`random` / `gp` / `llm` / `debate`); `run.py compare` reports the per-source portfolio
metrics side by side.
