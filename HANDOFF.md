# Handoff — state of the work, and what you don't need to re-run

Everything expensive is committed. **$108 of API spend and 496 multi-agent debates are in
this repo**, so you can reproduce every number in the report without an API key.

## What's committed (and what it cost)

| Run | What it is | Ideas | Alphas | Cost |
|---|---|---|---|---|
| `outputs/run_20260722_095043/` | first debate run | 180 | 1,037 | $39.16 |
| `outputs/run_20260722_110816/` | second debate run | 276 | 1,453 | $60.04 |
| `outputs/combined_100_20260722_123326/` | the two above merged, deduped — **the paper's book** | 456 | 2,352 | — |
| `outputs/probe_subset_v1/` | decorrelation experiment (see below) | 40 | 172 | $9.06 |

Each run folder has `alphas.db` (every alpha with its in-sample metrics), `survivors.csv`,
`generated.jsonl`, the portfolio/gate JSON, the figures, and `debate.tar.gz`.

`debate.tar.gz` is the full transcript of every debate — drafts, cross-reviews, revisions,
moderator synthesis, per-formula diagnostics. 496 traces, 83 MB of JSON compressed to 20 MB.
Unpack one with:

```bash
tar -xzf outputs/probe_subset_v1/debate.tar.gz -C outputs/probe_subset_v1/
```

## What is NOT committed, and why

- **`data/panels/` (533 MB)** — rebuild with `python scripts/prepare_data.py`. Excluded
  because single files hit GitHub's 50 MB warning, and because CRSP/Compustat is licensed:
  pull it under your own WRDS credentials (`scripts/download_wrds.py`). `data/aux/sector.parquet`
  and `data/universe/membership.parquet` ARE committed — they're small and everything needs them.
- **`papers/library/` (63 MB)** — other authors' PDFs. Reading list with links is in
  `papers/llm_alpha_mining_literature.md`.
- **`.env`** — put `OPENAI_API_KEY` and/or `OPENROUTER_API_KEY` there. Only needed to generate
  NEW alphas; all analysis below runs offline.

## Reproducing the headline numbers

You need `data/panels/` for anything that re-evaluates signals. No API key required.

```bash
python scripts/measure_diversity.py outputs/combined_100_20260722_123326/alphas.db
```

Should print **rho-bar 0.151, effective breadth 6.6** on 1,485 survivors. That is the paper's
central result: the book sits at 99.6% of its own `1/rho-bar` ceiling, so generating more
alphas at the same correlation cannot raise the Sharpe.

```bash
python run.py portfolio --db outputs/combined_100_20260722_123326/alphas.db
```

Rebuilds the portfolio from stored alphas (test Sharpe ~1.10 equal-weight) without regenerating.

## Where the work actually stands

The report (`papers/MFE27_Term2_Thibaut_AlphaGPT3_Liu_FinalReport.pdf`, source in
`papers/docs/`) is **finished and submitted**. Everything below is after that.

**The finding.** Effective breadth is `N_eff = N / (1 + (N-1)*rho_bar) -> 1/rho_bar`. The
456-idea book has rho-bar = 0.151, so it caps at ~6.6 independent bets and is already at
99.6% of that. The $60 second run bought +0.5 effective bets. More alphas is the wrong lever;
lowering rho-bar is the only one.

**The experiment that failed (`probe_subset_v1`).** Three changes at once: dropped the 15
hardcoded themes for a per-idea random subset of ~7 of 54 terminals; replaced the
Momentum/MeanReversion/Fundamental agents with Economist/Statistician/Trader lenses; kept
low-confidence ideas instead of discarding them. Result: **rho-bar got worse, 0.151 -> 0.213**.

Diagnosis from the traces (`scripts/measure_diversity.py` reports the last two lines):

- **Zero of 172 alphas used any time-series operator**, and only 6 distinct operators
  appeared vs 32 in the baseline. Everything collapsed to `cs_rank(a) + cs_rank(b) - cs_rank(c)`.
- The three agents converged completely — in a sampled debate all three picked the same
  fields with the same interpretations, and 239 of 240 Stage-1 reviews returned the identical
  decision. The 3-agent ensemble became a 1-agent ensemble.
- The themes had been secretly carrying *computational form* ("price momentum" forces
  `ts_returns`). Terminal subsets vary WHICH fields an idea sees, not WHAT SHAPE of
  computation is applied — and form matters more than fields for correlation.

The three changes were confounded, so this cannot separate "subsets don't work" from
"removing the strategy priors broke form diversity."

**Known bug, not yet fixed.** `DebateGenerator` builds its diagnostics from the *full*
54-field pset while `_annotate_formula_parse_status` checks against the *7-field subset*
(`alpha_gpt/factory/generators.py:309` vs `alpha_gpt/debate/moderator.py:199`). Off-menu formulas
get measured as healthy — 177 of 210 carry an `OK` diagnostic — reviewers endorse them on that
evidence, then they're silently dropped at the end. ~29% of Stage-2 output is wasted this way.
Fix: build the diagnose pset per debate from the subset.

## Suggested next steps, in order

1. **Fix the pset mismatch above**, and parse-check drafts against the subset too, so the
   existing revise loop can act on `REJECTED: uses off-menu field roe` instead of losing 29%
   silently.
2. **Restore draft-time agent differentiation** on an axis orthogonal to strategy family —
   sample a *construction form* (time-series / cross-sectional / interaction) per agent. This
   addresses both the agent collapse and the zero-`ts_`-operator result.
3. **Ingest OSAP** (`scripts/ingest_osap.py`, nothing ingested yet). The 54 current fields are
   almost entirely quarterly accounting data and contain only a few dozen independent
   directions; OSAP adds ~100-200 momentum/reversal/volatility/liquidity signals. This raises
   the `1/rho-bar` ceiling itself, which no amount of prompt engineering can do.

Probe cost is ~$0.23/idea, so a 40-idea diversity probe is ~$9 and ~10 minutes. `rho-bar` is
label-free, so tune on it freely without touching the test split. Track it together with
mean |IC| — rho-bar alone is gameable, since pure noise drives it to zero.
