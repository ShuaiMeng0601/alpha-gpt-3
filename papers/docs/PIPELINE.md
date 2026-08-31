# Alpha-GPT 3.0 — Engineering Reference

A detailed, implementation-accurate description of the whole system: from raw
market data to a backtested portfolio of machine-generated alphas. Written as an
internal reference (not a conference paper). Every stage names the real files,
functions, parameters, and the anti-look-ahead controls.

---

## 0. What the system is

Two macro-phases:

1. **Dataset build (offline, run occasionally).** Turn raw CRSP/Compustat/OSAP into
   a clean, point-in-time, survivorship-free panel library — one `date × PERMNO`
   float matrix per data field. These fields are the *terminals* of the alpha
   language.
2. **Alpha factory (run repeatedly).** Autonomously generate alpha *expressions*,
   verify them, evaluate them on train/validation, store them in a database; then
   filter, combine many of them into a single portfolio signal, and backtest that
   portfolio on a held-out test period.

```
                          ┌─────────────────────── DATASET BUILD ───────────────────────┐
 CRSP CIZ (26GB) ─┐       │ PIT survivorship-free universe (annual rebalance)            │
 Compustat (2.3GB)├──────▶│ price/vol panels + cs_* fundamentals + osap_* signals        │──▶ data/panels/*.parquet
 OSAP signals ────┘       │ + sector aux + coverage report + terminal discovery/IC-cap   │     data/universe/membership.parquet
                          └──────────────────────────────────────────────────────────────┘
                                                     │  (terminals)
                                                     ▼
   ┌──────────────────────────────── ALPHA FACTORY (loop N) ────────────────────────────────┐
   │  idea (LLM) ─▶ formula(s) ─▶ VERIFY ─▶ EVALUATE (train+val) ─▶ STORE (SQLite, dedup)     │
   │     ▲ optional: multi-agent Idea Debate + Formula Debate;  optional: GP amplification     │
   └──────────────────────────────────────────────────────────────────────────────────────────┘
                                                     │  (DB of expressions + metrics)
                                                     ▼
   ┌────────────────────────────── PORTFOLIO (offline) ──────────────────────────────────────┐
   │  filter survivors (val metrics) ─▶ ECON GATE (scorecard ≥ 7/10) ─▶ re-eval on TEST ─▶     │
   │  sign-flip+zscore ─▶ combine (1/N | IC-weighted | orthogonalized) ─▶ backtest on TEST     │
   └──────────────────────────────────────────────────────────────────────────────────────────┘
```

**Golden rule (leakage):** selection happens on **train/validation**; the **test**
split (2021–2023) is only ever touched by the final portfolio backtest.

---

## 1. Repository map by stage

| Stage | Files |
|---|---|
| Dataset: universe | `scripts/prepare_data.py`, `alpha_gpt/data/universe.py` |
| Dataset: fundamentals | `scripts/ingest_fundamentals.py`, `alpha_gpt/data/fundamentals.py` |
| Dataset: OSAP | `scripts/ingest_osap.py`, `alpha_gpt/data/osap.py` |
| Dataset: quality/aux | `alpha_gpt/data/coverage.py`, `alpha_gpt/analysis/neutralize.py`, `scripts/verify_data.py` |
| Loading/splits | `alpha_gpt/data/loader.py`, `alpha_gpt/config.py` |
| Alpha language | `alpha_gpt/operators/alpha_ops.py`, `alpha_gpt/gp_search/primitives.py`, `seed_injector.py` |
| Terminal selection | `alpha_gpt/gp_search/terminal_selection.py` |
| Idea/formula debate | `alpha_gpt/debate/{prompts,agents,moderator,models}.py` |
| GP search | `alpha_gpt/gp_search/engine.py` |
| Factory | `alpha_gpt/factory/{generators,verifier,run,parallel,fast_metrics,db}.py`, `scripts/run_factory.py` |
| Portfolio | `alpha_gpt/portfolio/construct.py`, `scripts/build_portfolio.py` |
| Eval primitives | `alpha_gpt/analysis/metrics.py`, `alpha_gpt/backtest/backtester.py`, `analysis/visualize.py` |

---

## Stage 0 — Dataset construction

### 0.1 Sources
- **CRSP CIZ daily** (`data/crsp_daily.csv`, ~26 GB, ~50M rows, 25,306 PERMNOs, 2000–2025):
  OHLC, returns (`DlyRet` already includes delisting returns in CIZ), volume,
  shares, market cap, bid/ask, share-class/exchange/delisting metadata.
- **Compustat quarterly fundamentals** (`data/compustat_fundamentals.csv`, ~700 line
  items keyed by `gvkey`, with `rdq` earnings-announcement date) and the **WRDS
  Financial Ratios** file (`compustat_ratios.csv`, keyed by both `gvkey` and
  `permno`, with `public_date`).
- **OSAP** (Open Source Asset Pricing, Chen–Zimmermann): ~200 published firm-level
  signals, monthly, keyed by `permno, yyyymm` (user-downloaded → `data/osap_signals.csv`).

### 0.2 Point-in-time, survivorship-free universe (`data/universe.py`, `prepare_data.py`)
The original code selected a static 500-stock universe using *full-sample* filters
(look-ahead + survivorship). Replaced with a **two-pass streaming build**:

- **Pass 1** streams CRSP in chunks, drops the ~23% `SecurityHdrFlg=='Y'` header
  duplicate rows, and reduces to a per-`(PERMNO, year)` summary (mean market cap +
  an eligibility flag). Eligibility = US common stock on a major exchange:
  `ShareType=NS`, `SecurityType=EQTY`, `IssuerType∈{CORP,REIT}`, `PrimaryExch∈{N,A,Q}`,
  `USIncFlg=Y` (≈ classic CRSP shrcd 10/11).
- **Annual selection** (`select_universe`): at each calendar year, take the
  top-`universe_size` (default 1,500) eligible names by **prior-year** mean market
  cap — strictly trailing, so no future information enters selection.
- **Pass 2** streams CRSP again, keeps only member rows, dedups `(PERMNO, date)`,
  and pivots each field to a `date × PERMNO` panel.
- **Membership mask** = in-universe-that-year **AND** actually trading
  (`close.notna()`), so a name is dropped at delisting and outside its membership
  years. Every panel is `.where(membership)`-masked. `forward_returns =
  returns.shift(-1).where(membership)` (computed from the *masked* returns, so it is
  self-consistent and a name's last forward return doesn't leak an out-of-universe day).

Result: **2,752 names** across 2010–2023 (≈1,500/yr with turnover), with **2,183
mid-sample exits and 1,146 entries** — i.e. genuinely survivorship-free.
`scripts/verify_data.py` asserts all of this (10 checks).

### 0.3 Price/volume panels
`close, open, high, low, volume, returns, price, shrout, market_cap, bid, ask,
num_trades, dollar_volume` (13 panels). Forward returns: next-day, masked.

### 0.4 Compustat → `cs_*` characteristics (`fundamentals.py`)
- `gvkey→permno` mapping is taken from the `(gvkey, permno, public_date)` triples
  already in the ratios file (no WRDS CCM download); many-to-one resolved by
  date-overlap + in-universe + largest-cap tie-break.
- **PIT date** = `rdq` (earnings announcement; present 86.6%) else `datadate + 90d`.
- Derives ~17 standard characteristics (gross profitability, ROA/ROE, margins,
  leverage, B/M, E/P, S/P, asset growth, sales growth, accruals, …), forward-filled
  to daily and masked. Namespaced `cs_*` to avoid collisions with WRDS ratios.

### 0.5 OSAP ingester (`osap.py`)
Wide monthly CSV → per-signal `date × PERMNO` panel. **Critical PIT control:** a
month-`M` signal is stamped active on `month_end(M) + 1 day`, then forward-filled to
daily (so a January signal only affects February+). Coverage-filtered, namespaced
`osap_*`. (Unit-tested: a Jan signal is NaN in January, active in February.)

### 0.6 Sector aux panel
GICS `gsector` → `data/aux/sector.parquet` (kept **out** of `data/panels/` so it is
never treated as a numeric terminal). For future cross-sectional neutralization
(`analysis/neutralize.py`, not yet wired into the eval path).

### 0.7 Terminal discovery + IC cap (`primitives.discover_terminals`, `terminal_selection.py`)
With ~50–250 candidate panels, feeding all of them to the search explodes it.
`discover_terminals` lists candidate fields (excludes `forward_returns`,
`vw_market_return`, `sector`, underscore files). `select_terminals` keeps a stable
core (`DEFAULT_TERMINALS`) plus the highest-|IC| extras up to `config.max_terminals`
(default 40), where |IC| is the fast vectorized rank-correlation on the train split.
The full library stays on disk; each run uses the most predictive subset.

### 0.8 Coverage report (`coverage.py`)
Per-panel % non-NaN over the universe, median per-day breadth (stocks/day), and a
first-valid-date look-ahead canary → `data/panels/_coverage_report.csv`. A field
qualifies as a terminal only if coverage ≥ 0.20 **and** breadth ≥ 100/day (the
IC/backtest code drops NaN pairwise per day, so sparse fields silently shrink the
effective universe).

### 0.9 Anti-look-ahead controls (summary)
| Risk | Control |
|---|---|
| Survivorship | trailing-only selection; names retained through delisting |
| Universe selection look-ahead | prior-year cap; no full-sample filters |
| Fundamentals timing | `rdq` / `public_date` (already release-lagged) |
| Monthly→daily fills | OSAP shifted +1 month; mask after ffill |
| Delisting returns | embedded in CIZ `DlyRet`; captured as prior-day forward return |
| Test leakage | selection on train/val only; test reserved for final backtest |

---

## Stage 1 — Idea generation

Three interchangeable strategies feed the same downstream pipeline.

### 1.1 Autonomous LLM idea generator (factory default, `factory/generators.py::LLMGenerator`)
Each call: sample a **theme** from a fixed list (momentum, value, quality, accruals,
low-vol, liquidity, size, growth, volume-price divergence, seasonality, leverage,
reversion, trend, payout) → one chat completion asking the model to *invent* one
original cross-sectional idea in that theme and emit `k` distinct formulas using
**only** the run's resolved terminals → parse strict JSON `{idea, hypothesis,
expressions[]}`. Diversity comes from theme sampling + temperature (0.9). Token usage
is recorded on every call (incl. parse failures) for honest cost accounting.

### 1.2 Multi-agent Idea Debate (optional, higher quality, `debate/`)
The original Alpha-GPT 3.0 contribution. Three agents with distinct priors
(Momentum, Mean-Reversion, Fundamental) run a 3-phase debate:
**draft** (each proposes a structured hypothesis: mechanism, signal type,
directionality, proxies, filters, normalization — *no formulas yet*) →
**cross-review** (each scores the others on 8 dimensions, accept/revise/reject) →
**revision** (each revises, explicitly accepting/rejecting feedback) →
**moderator** synthesizes 2–3 final hypothesis specs. ~10–12 LLM calls. Used by
`main.py --mode debate-only|full`; too expensive for million-scale loops.

### 1.3 Random generator (no API, `RandomGenerator`)
Samples valid random expression trees from the primitive set. Lets you stress-test
the entire loop/DB/portfolio for **$0** and provides a noise baseline.

---

## Stage 2 — Formula generation & the alpha language

### 2.1 The DSL (`operators/alpha_ops.py`)
An alpha is a symbolic expression over **terminals** (data panels) and **operators**:
- **Time-series** (per stock, curried windows): `ts_mean/std/delta/rank/min/max`
  (w∈{5,10,20,60}), `ts_returns` (1,5,20), `ts_corr` (10,20).
- **Cross-sectional** (per day, across stocks): `cs_rank`, `cs_zscore`.
- **Element-wise**: `add, sub, mul, safe_div, log_abs, abs_val, sign, neg`.
All operate on `date × PERMNO` DataFrames and are NaN-safe.

### 2.2 Direct formula emission (factory)
The LLM returns expressions directly (Stage 1.1). `seed_injector.normalize_expression`
maps LLM-style `ts_delta(close, 5)` → curried `ts_delta_5(close)` and aliases
(`rank→cs_rank`, etc.); `parse_expression` builds a DEAP `PrimitiveTree`.

### 2.3 Formula Debate (optional, `debate/`)
Mirror of the idea debate at the formula level: each agent drafts 1–2 formulas per
hypothesis (tagged with a role: main/directional/filter/composite), peers review for
faithfulness/implementability/robustness/novelty/simplicity, authors revise, a
hard **parse-gate** drops anything that doesn't compile against the live
operator/terminal registry, and a moderator selects a diverse seed set.

---

## Stage 3 — Verification (`factory/verifier.py`)

The gate that keeps the DB clean at scale. `verify_expression` runs:
1. **Parse** (`parse_expression`) → reject `parse_error`.
2. **Structure**: ≥1 terminal, depth ≤ 8 → `no_terminals` / `too_deep`.
3. **Safe eval** on the train panels (`eval_expr`, exceptions caught) → `eval_error`.
4. **Output sanity**: replace ±inf→NaN; reject `all_nan`; require coverage ≥ 5% and
   median breadth ≥ 20 → `sparse`; require cross-sectional variation on ≥30% of days
   → `constant` (a constant signal has undefined IC).
Returns the evaluated signal so the evaluator doesn't recompute it. On the 1,000-run,
rejects were: 74 all-NaN, 47 parse, 24 sparse, 5 constant, 3 too-deep.

---

## Stage 4 — Evaluation (train/val) & metrics

### 4.1 Splits (`data/loader.py`, `config.py`)
Strict calendar split: **train 2010–2017**, **validation 2018–2020**, **test 2021–2023**.

### 4.2 Metric definitions
For a daily cross-section of alpha scores `α_t` and next-day returns `r_{t+1}`:
- **IC** (Information Coefficient): `IC_t = Spearman(α_t, r_{t+1})`; report `mean_t IC_t`.
  Economic signal if consistently > ~0.02.
- **ICIR**: `mean_t(IC_t) / std_t(IC_t)`. Stability / signal-to-noise; the IC
  t-stat is `ICIR · √T`.
- **Turnover**: mean daily |Δ cross-sectional rank| (rebalancing cost proxy).
- **Coverage**: fraction of non-NaN cells over the universe.

### 4.3 Two evaluators (deliberately different speeds)
- **Fast/vectorized** (`factory/fast_metrics.py`) — used during generation for
  *selection* metrics. IC via rank-then-Pearson over the whole panel at once (no
  per-date Python loop); Sharpe on a **rank-weighted, dollar-neutral factor return**
  `w_t = (rank(α_t) − mean) / Σ|·|`, `ret_t = Σ_i w_{it} r_{i,t+1}`. ~50× faster than
  the looped version; close enough to *rank* alphas.
- **Exact** (`analysis/metrics.py` + `backtest/backtester.py`) — used for the final
  portfolio. Spearman per day; quintile long-short backtest.

### 4.4 What is stored per alpha
train/val IC, train/val ICIR, val Sharpe / annual return / max drawdown, turnover,
coverage, n_terminals, depth, generation tokens. **Test metrics are not computed
here** (no leakage).

---

## Stage 5 — GP amplification (optional, `gp_search/engine.py`)

To get many alphas per LLM call, seed a DEAP genetic-programming search with the
LLM's parsed expressions and evolve them: tournament selection, subtree crossover,
point mutation, depth-limited, fitness = mean train IC (subsampled dates for speed).
Each idea can thus spawn dozens–hundreds of verified candidates at $0 marginal LLM
cost. Not used in the 1,000-run above, but the cheapest path to million-scale.

---

## Stage 6 — Storage (`factory/db.py`)

SQLite, one row per alpha: `hash` (unique, dedup on normalized expression),
`expression`, `idea`, `hypothesis`, `source`, `model`, the metrics above,
`gen_prompt_tokens`/`gen_completion_tokens`, `status`, `created_at`. Indices on
status / val_ic / val_sharpe. **Only expressions + scalars are stored, never
signals** — 1M signals would be terabytes; signals are recomputed on demand for the
filtered survivors. `query()` filters by **|IC|/|ICIR|/|Sharpe|** (a strongly
negative-IC alpha is one sign-flip from a strong signal), turnover, coverage.

---

## Stage 7 — Repeat N times (the loop & parallelism)

`factory/run.py` (sequential) and `factory/parallel.py` (parallel). The parallel
runner has three phases:
1. **Generation** — thread-pool the LLM calls (`gen_workers`, default 8/10). LLM
   calls are I/O-bound, so threads give near-linear speedup. (Random = instant.)
2. **Evaluation** — thread-pool verify+eval (`eval_workers`). **Threads, not
   processes**, because the panels are GBs; processes would copy them per worker and
   exhaust RAM. NumPy/scipy release the GIL during the heavy ops, so threads still
   help, and the vectorized metrics keep each task short.
3. **Store** — single-writer dedup + insert (SQLite).

**Throughput / cost model** (DeepSeek-V3.2 @ OpenRouter, observed):
- ~218 tokens-in + ~70 tokens-out per *alpha* (5 alphas/idea amortizes the ~800-token
  prompt) → **~$0.000076 / alpha**.
- 1,000 alphas ≈ **$0.06, ~13 min** (gen 4 min @8 threads, eval 9 min @10 threads).
- Money scales trivially (1M ≈ ~$75); **wall-clock is the real constraint** → GP
  amplification and/or more eval workers are the levers.

---

## Stage 8 — Portfolio construction (`portfolio/construct.py`)

1. **Filter survivors** from the DB. Default/recommended: a **train∧val consistency
   gate** (`train_ic·val_ic > 0`) + a mild |val IC| floor + **no top-k cap**.
   Empirically this beat "top-k by val IC" by a wide margin (equal-weight OOS Sharpe
   0.39 → ~1.2 on the first run) because (a) breadth helps (IR ≈ IC·√breadth, so
   don't cap) and (b) the highest-val-IC alphas are often the most *overfit* — a
   stricter val-IC bar alone is the wrong lever; requiring the signal to hold on
   *both* train and val removes the lucky ones. Optional extra gates: |train IC|
   floor, |ICIR|, max turnover. (Caveat: tuning thresholds by looking at test is
   test-peeking; the consistency gate is principled because it's a validation-side
   rule, but specific threshold values chosen on test are optimistic.)
2. **Economic-reasoning gate** (`gate/`). Keep a survivor only if it ALSO clears a
   10-criterion economic scorecard (mechanism, direction, parsimony, robustness,
   interpretability, tradability, risk-orthogonality, novelty, data-integrity, breadth),
   total ≥ threshold (default 7/10). Backend is a swappable knob — `none` (off) /
   `heuristic` (deterministic, offline; the default) / `llm` / `debate` (the 3 debate
   personas score as a panel). So an alpha is used only if it has positive IC AND makes
   economic sense; writes `gate_scorecards.json`.
3. **Re-evaluate** each survivor's expression on the **test** panels → signal.
4. **Sign-flip** by `sign(val_ic)` (so all alphas point the same way) and
   **cross-sectionally z-score** each (comparable scale).
5. **Combine** into one composite signal, three methods:
   - **equal (1/N)**: NaN-aware mean of the z-scored signals.
   - **ic_weighted**: weighted mean, weights ∝ |val IC|.
   - **orthogonalized**: sequentially residualize each signal against the running
     composite, `r_i = s_i − β·comp`, `β = ⟨s_i,comp⟩/⟨comp,comp⟩`, then average the
     residuals (removes redundancy among similar ideas).

---

## Stage 9 — Portfolio backtest & metrics (`scripts/build_portfolio.py`)

The composite signal is backtested **once** on the test split with the exact
**long-short quintile** backtester (`backtest_alpha`): each day rank stocks, equal-
weight long the top quintile and short the bottom, daily LS return, then annualized
Sharpe, annual return, max drawdown, plus IC/ICIR/turnover. Baselines reported
alongside: the **best single alpha** and the **value-weighted market** benchmark.
Outputs a comparison table + equity-curve plot + `portfolio_results.json`.

---

## 10. End-to-end runbook

```bash
# Dataset (occasional)
python scripts/prepare_data.py --universe-size 1500
python scripts/ingest_fundamentals.py
python scripts/ingest_osap.py            # after downloading data/osap_signals.csv
python scripts/verify_data.py
python -m alpha_gpt.data.coverage

# Factory + portfolio in ONE command (recommended) — one self-contained run folder
python scripts/run_pipeline.py --source llm --n 200 --alphas-per-idea 5

# ...or run the two stages separately (e.g. to re-filter an existing DB):
python scripts/run_factory.py --n 200 --alphas-per-idea 5 --source llm \
       --parallel --gen-workers 8 --eval-workers 10 --db data/alphas.db
python scripts/build_portfolio.py --db data/alphas.db --min-ic 0.005 --top-k 60
```

---

## 11. Empirical result (first unpolished 1,000-alpha run)

- Generation: 200 LLM calls → 194 ideas → 970 expressions; 811 stored, 6 dup, 153
  rejected by the verifier. 218k tokens, **$0.062**, **13.2 min**.
- Per-alpha validation signal is weak (median |val IC| 0.004) — expected for
  zero-shot LLM ideas. The single highest-val-IC alpha **lost 16% OOS** (Sharpe −0.79).
- Portfolio of top-60, out-of-sample (test 2021–2023):

| Method | # | IC | ICIR | Sharpe | Ann.Ret | MaxDD |
|---|--:|--:|--:|--:|--:|--:|
| best single | 1 | −0.005 | −0.03 | −0.79 | −15.9% | −47.5% |
| equal (1/N) | 60 | 0.012 | 0.07 | 0.39 | 5.4% | −22.3% |
| ic_weighted | 60 | 0.010 | 0.06 | 0.23 | 2.6% | −23.4% |
| **orthogonalized** | 60 | **0.016** | **0.11** | **0.88** | **13.9%** | −19.1% |

Takeaways: (1) the single "best" alpha overfits val→test; (2) combining weak,
uncorrelated signals produces a real factor with a *shallower* drawdown; (3)
orthogonalization beats naive 1/N because the LLM emits many correlated ideas.

---

## 12. Reading the results — is a Sharpe of 0.88 good?

Reference points (annualized, **gross** unless noted):
- S&P 500 long-only, long run: ~0.4–0.6.
- A typical *single* published anomaly (value, momentum, …): ~0.3–0.6 gross.
- A well-built *multi-factor* tangency portfolio (FF5 / q-factors): ~1.0–1.5 gross.
- Top quant funds: ~1–2 **net of costs**, after enormous effort and capacity limits.

So **0.88 gross, out-of-sample, from a completely unpolished automated pipeline is a
genuinely encouraging number** — it sits in the "this is real signal, not noise"
range and beats both the best single alpha and the market benchmark with a smaller
drawdown. But it is **not** a validated edge yet, because:
- **Gross, not net.** Turnover ≈ 0.10/day is high; at ~10 bps/round-trip this could
  shave ~2–3%/yr and pull net Sharpe toward ~0.5–0.7. Net is what matters.
- **One test window, one run.** No multi-period / multi-seed validation → high
  variance; some of 0.88 could be luck.
- **Selection effects.** 60 chosen from 811 on validation; the val→test map is noisy.
- **No costs/borrow/capacity/neutralization** modeled.

Honest verdict: treat 0.88 as a **promising prototype signal**, strong evidence the
*system* produces something, not a deployable strategy. The robust finding is the
*shape* (portfolio ≫ best single, shallower drawdown, orthogonalized wins), which is
exactly what good alpha-combination should look like.

---

## 13. Limitations & roadmap

- **Net-of-cost backtest** (turnover × spread) — the single most important missing
  piece for credibility. Refs: Novy-Marx–Velikov, Chen–Velikov (in `papers/library/`).
- **Validation rigor** — multiple test windows / rolling OOS / multiple seeds;
  deflated Sharpe & multiple-testing control (Harvey-Liu-Zhu) given 811 candidates.
- **Sector/size neutralization** — `analysis/neutralize.py` exists but isn't wired
  into eval/backtest.
- **Incremental selection** — score each new alpha by marginal contribution to the
  existing portfolio, not standalone IC.
- **Scale** — GP amplification (Stage 5) + more eval workers to push toward 1M; the
  cost is wall-clock, not dollars.
- **Idea-aware terminals** — currently the GP terminal cap is IC-screened and
  idea-agnostic; could condition the menu on the idea/theme.
- **Richer data axes** — IBES (analyst), OptionMetrics (IV), 13F, short interest
  (the WRDS "Phase 2" ingesters), each a new orthogonal terminal family.
```
