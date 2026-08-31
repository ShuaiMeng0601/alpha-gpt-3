# Alpha-GPT — Architecture & Workflow

How the code is organized and how data flows through it, as of the current
(post-refactor) layout. For the deep data-engineering reference (look-ahead
controls, metric definitions, empirical results) see [`PIPELINE.md`](PIPELINE.md).

---

## 1. The big picture

Three phases. The first is offline and run rarely; the other two are the things you
run day to day, and they sit on a **shared core** so neither reaches into the other.

```
                         ┌───────────────── OFFLINE: dataset build ─────────────────┐
 CRSP / Compustat / OSAP │ scripts/prepare_data.py, ingest_*.py → data/panels/*.parquet │
 (raw CSVs)              │ PIT, survivorship-free; one date×PERMNO matrix per field     │
                         └──────────────────────────────┬───────────────────────────────┘
                                                         │  terminals
                          ┌──────────────────────────────┴──────────────────────────────┐
                          ▼                                                               ▼
        ┌──────── ABLATION pipeline ────────┐                     ┌──────── FACTORY pipeline ────────┐
        │ python -m alpha_gpt.main          │                     │ scripts/run_pipeline.py          │
        │ idea → seeds → (GP?) → validate   │                     │ generate → verify → score → store │
        │      → backtest top-k → report    │                     │  → filter → portfolio backtest    │
        └───────────────────────────────────┘                     └───────────────────────────────────┘
                          │                                                               │
                          ▼                                                               ▼
              outputs/<ts>_<mode>_run<i>/                                     outputs/run_<ts>/
              (alpha_report.md, results.json)                       (portfolio_report.md, alphas.db, …)
```

The **golden rule** (no leakage): selection happens on train/validation; the **test**
split is only ever touched by the final backtest.

---

## 2. Layering

Strict dependency direction — arrows point to dependencies; nothing below imports
anything above it, and there are no private cross-module imports.

```
 entry points     main.py (CLI)          pipeline.py        scripts/*.py
                       │                      │                  │
 orchestration    modes.py  compare.py    (pipeline stages)      │
                       │                      │                  │
 shared core      report.py   experiment.py   llm.py   ◄─────────┘
                       │            │            │
 domains          debate/  gp_search/  factory/  portfolio/  gate/  analysis/  backtest/
                       │            │            │
 foundations      operators/alpha_ops.py     data/      config.py
```

| Layer | Modules | Role |
|---|---|---|
| Entry points | `main.py`, `pipeline.py`, `scripts/*` | thin CLIs / arg parsing |
| Orchestration | `modes.py`, `compare.py` | sequence the steps of one pipeline |
| Shared core | `experiment.py`, `llm.py`, `report.py`, `gate/` | run setup, model I/O, eval+report, economic gate — used by both pipelines |
| Domains | `debate/`, `gp_search/`, `factory/`, `portfolio/`, `gate/`, `analysis/`, `backtest/` | the actual algorithms |
| Foundations | `operators/alpha_ops.py`, `data/`, `config.py` | the alpha language, panels, knobs |

---

## 3. Ablation pipeline (`python -m alpha_gpt.main`)

Purpose: compare seed-generation strategies (multi-agent debate vs. single-agent vs.
random) with optional GP amplification. A **mode is a row in `modes.MODES`**: a seed
source × a GP toggle.

```
main.main()
└─ for run_idx in range(num_runs):
   └─ modes.run_mode(idea, mode, config, out_dir, gp_seed, run_idx)
      1. experiment.load_and_split(config)        → train, val, test, terminals, pset
         experiment.compute_vw_benchmark(test)    → market benchmark curve
         llm.make_client(config)                  → (only if the mode needs an LLM)
      2. seeds = _generate_seeds(spec, …):
            none          → None                          (random-gp)
            single_agent  → llm.call_json(SINGLE_AGENT…)  (single-agent[-gp])
            debate        → moderator.run_idea_debate
                            moderator.run_formula_debate   (debate-only / full)
      3. candidates:
            GP   → seed_injector.inject_seeds → engine.run_gp → plot_gp_evolution
            else → report.parse_seed_formulas (parse + score on train)
      4. report.validate_alphas(candidates, pset, val)     → keep val-IC > 0
      5. gate.apply_gate(make_gate(gate), validated, pset) → keep cards ≥ threshold
                 → an alpha is used only if it has positive IC AND makes economic sense
                 → writes econ_scorecards.json
      6. report.evaluate_and_report(…)                     → backtest top-k on TEST
                 ├─ engine.eval_expr / metrics.* / backtester.backtest_alpha
                 ├─ explainer.explain_alpha (LLM, optional)
                 └─ writes alpha_report.md + results.json + equity_curves.png
```

`python -m alpha_gpt.main --compare` → `compare.run_compare()` scans every
`outputs/*/results.json`, groups by mode, and writes `outputs/comparison/`.

| mode | seed source | GP? | LLM cost |
|---|---|---|---|
| `random-gp` | none | yes | free |
| `single-agent` | 1 LLM call | no | ~1 call |
| `single-agent-gp` | 1 LLM call | yes | ~1 call |
| `debate-only` | multi-agent debate | no | ~20 calls |
| `full` | multi-agent debate | yes | ~20 calls |

---

## 4. Factory pipeline (`scripts/run_pipeline.py` → `pipeline.run_pipeline`)

Purpose: generate a large pool of alphas and build a backtested portfolio of the
survivors. Six stages, one self-contained run folder, a progress bar per stage.

```
pipeline.run_pipeline(config, source, n_ideas, …, gate, gate_threshold)
  Stage 1  load    experiment.load_and_split(config)
  Stage 2  factory generators.{LLM,Random}Generator
                   → factory.run.run_factory            (serial)
                     factory.parallel.run_factory_parallel (threaded gen + eval)
                       per expression:
                         verifier.verify_expression       (parse/structure/eval/sanity gate)
                         run.compute_alpha_metrics         (train+val IC/ICIR/Sharpe/…; fast_metrics)
                         db.insert_alpha                   (SQLite, dedup on expr hash)
  Stage 3  filter  db.query(consistency gate: train_ic·val_ic>0, |valIC| floor, …)
  Stage 4  gate    gate.make_gate(gate) + gate.apply_gate  (economic scorecard ≥ threshold)
                     → keeps only IC-survivors that ALSO make economic sense
                     → writes gate_scorecards.json + annotates survivors.csv
  Stage 5  backtest pipeline.backtest_portfolio:
                     portfolio.construct.build_signals     (re-eval survivors on TEST)
                     portfolio.construct.fit_combiners      (fit ortho betas on VAL — no look-ahead)
                     portfolio.construct.combine            (equal / ic_weighted / orthogonalized)
                     backtester.backtest_alpha              (long-short quintile, OOS)
  Stage 6  report  writes portfolio_report.md + .json + equity curves + pipeline_summary.json
```

The economic gate (Stage 4) is a separate shared-core package, `alpha_gpt/gate/`, with
swappable backends (`none` / `heuristic` / `llm` / `debate`) behind one `apply_gate` path
shared with the ablation pipeline — so the "use this alpha" decision depends on positive
IC **and** economic reasoning, and the backend is a single ablation knob.

`scripts/build_portfolio.py` runs **the gate + Stage 5–6 only** against an existing
`alphas.db` (re-filter / re-gate / re-combine without regenerating). `scripts/run_factory.py`
runs **Stage 1–2 only** (fill a DB).

Selection metrics are computed once in `run.compute_alpha_metrics` (the single source
of truth for what the DB stores) and reused by both the serial and parallel runners.

---

## 5. The alpha language (single source of truth)

```
operators/alpha_ops.py     # defines every operator; exposes UNARY_OPS, BINARY_OPS, ALL_OPS
        │  (imported, never re-listed)
        ├─► gp_search/primitives.py      # build the DEAP PrimitiveSet from UNARY/BINARY_OPS
        └─► gp_search/seed_injector.py   # CURRIED_MAP derived from ALL_OPS; LLM-string → DEAP tree
gp_search/engine.py        # eval_expr(tree, pset, panels) → signal; run_gp(...) evolution
```

An alpha is a symbolic expression over **terminals** (data panels) and **operators**
(time-series, cross-sectional, element-wise), evaluated by `engine.eval_expr` into a
`date × PERMNO` signal panel.

---

## 6. Artifacts

| Pipeline | Folder | Key files |
|---|---|---|
| Ablation | `outputs/<ts>_<mode>_run<i>/` | `alpha_report.md`, `results.json`, `equity_curves.png`, `gp_evolution.png`, `debate/` |
| Compare | `outputs/comparison/` | `summary_table.md`, `comparison.json` |
| Factory | `outputs/run_<ts>/` | `portfolio_report.md`, `portfolio_results.json`, `portfolio_equity_curves.png`, `survivors.csv`, `alphas.db`, `factory_summary.json`, `pipeline.log` |

---

## 7. Where to plug in new things

| To add… | Edit | Notes |
|---|---|---|
| an **ablation mode** | one row in `modes.MODES` | (seed source, GP toggle) — runner is shared |
| an **operator** | one entry in `operators/alpha_ops.py` | the primitive set and the LLM parser derive from it |
| a **seed source** | a branch in `modes._generate_seeds` | return a `list[str]` of expressions |
| a **combination method** | a branch in `portfolio.construct.combine` | fit any learned params on val in `fit_combiners` |
| a **data field / terminal** | a `data/` builder + `scripts/ingest_*` | drop a `date×PERMNO` parquet into `data/panels/` |
| a **generator** | a class in `factory/generators.py` | yield `IdeaBatch`es |
```
