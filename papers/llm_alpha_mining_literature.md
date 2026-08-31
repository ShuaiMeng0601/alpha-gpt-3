# LLM-Based Alpha Mining — Literature Map

> Working reference for the Alpha-GPT project. Created 2026-06-28.
>
> **Legend:** ✓ = citation verified directly from CogAlpha's reference list (the paper exists, details checked). Unmarked entries are from general knowledge — title/authors/year are reliable, but **verify the exact arXiv ID before citing**.
>
> This is a fast-moving area (most of section A is 2024–2026); expect new entries.

---

## TL;DR — how the field is organized

There are three clusters worth knowing, plus one adjacent line that's easy to confuse:

- **A. LLM-based alpha generation** — the direct competitors. LLMs propose/refine alpha *expressions or code*, usually combined with evolution or search.
- **B. LLM + evolution methods** — the general machinery (FunSearch etc.) that the alpha papers borrow.
- **C. Pre-LLM automated alpha mining** — GP and RL over formulas; the prior paradigm the LLM papers are beating.
- **Adjacent: LLM trading agents** — make *trading decisions*, not alpha formulas. Different problem; don't conflate.

**Where this project sits:** the autonomous descendant of the **Alpha-GPT** line (human-in-the-loop → autonomous **multi-agent debate**), with a **two-stage** idea→formula decomposition, a **symbolic DSL** representation, **GP amplification**, and richer **US + survivorship-free** data. The closest concurrent work is **CogAlpha** (code-based evolution).

---

## A. LLM-based alpha / factor generation (direct competitors)

| Paper | Authors, year | Venue / ID | What it does | Relevance to us |
|---|---|---|---|---|
| **Alpha-GPT** | Wang, Yuan, Zhou, Ni, Shum, Guo, 2023 | arXiv (verify) | Human-in-the-loop: NL idea → LLM → alpha → backtest → iterate | **Our lineage (1.0)** |
| **Alpha-GPT 2.0** | Yuan, Wang, Guo, 2024 | arXiv (verify) | Adds human–AI alignment / fuller framework | **Our lineage (2.0)** |
| **CogAlpha** | Liu et al., 2026 | arXiv:2511.18850 | Code-based evolution; 7-level/21-agent hierarchy; multi-agent quality checker | Closest concurrent work; "code > formula" claim |
| **AutoAlpha** (LLM) | Kou et al., 2024 ✓ | arXiv:2409.06289 | LLM evaluates/selects candidates + agentic framework | Agentic selection variant |
| **AlphaAgent** | Tang et al., 2025 ✓ | arXiv:2502.16789 | LLM mining with regularized exploration to counteract **alpha decay** | Decay-resistance angle |
| **AlphaJungle** ("Navigating the Alpha Jungle") | Shi, Duan, Li, 2025 ✓ | arXiv:2505.11122 | LLM-powered **MCTS** for formulaic factor mining; multi-step refinement | The tree-search route (cf. our "why debate over MCTS") |
| **RD-Agent(Q)** / R&D-Agent-Quant | Li et al., 2025 ✓ | arXiv:2505.15155 | Multi-agent, data-centric **feedback loop**; factor–model co-optimization (Microsoft) | Closest to our "feedback loop" future work |
| **QuantAgent** | Wang et al., ~2024 | verify | Self-improving LLM agent for trading/alpha | Verify scope before citing |

---

## B. LLM + evolution methods (the machinery these build on)

| Paper | Authors, year | Venue / ID | What it does |
|---|---|---|---|
| **FunSearch** | Romera-Paredes et al., 2024 ✓ | *Nature* 625 | LLM as mutation operator in evolutionary program search — the template for "LLM + evolution" |
| **AlphaEvolve** (DeepMind) | Novikov et al., 2025 ✓ | arXiv:2506.13131 | Coding agent for algorithmic discovery via evolution |
| **Mind Evolution** ("Evolving Deeper LLM Thinking") | Lee et al., 2025 ✓ | arXiv:2501.09891 | Evolutionary search as inference-time scaling |
| **EvoPrompt** | Guo et al., 2024 ✓ | ICLR 2024 | Evolutionary algorithms + LLMs (prompt optimization; same machinery) |

---

## C. Pre-LLM automated alpha mining (the prior paradigm)

| Paper | Authors, year | Venue / ID | What it does |
|---|---|---|---|
| **AlphaGen** | Yu et al., 2023 ✓ | KDD 2023 | "Generating Synergistic Formulaic Alpha Collections via **RL**" — the RL-over-formulas SOTA |
| **AutoAlpha** (evolutionary) | Zhang et al., 2020 ✓ | arXiv:2002.08245 | Hierarchical **evolutionary** algorithm for mining alphas |
| **AlphaEvolve** (alpha mining) | Cui et al., 2021 ✓ | SIGMOD 2021 | Learning framework to discover novel alphas (GP-style) |
| **AlphaForge** | Shi et al., 2025 ✓ | AAAI 2025 | Mine **and dynamically combine** formulaic factors |
| **101 Formulaic Alphas** | Kakushadze, 2016 | arXiv/journal | Canonical hand-built library (the "WorldQuant 101" reference) |
| GP / symbolic-regression roots | Lin et al. 2019; Schmidt & Lipson 2010 | — | GA-based alpha mining; symbolic regression |

---

## Adjacent: LLM trading / portfolio agents (different problem)

These make **trading decisions**, not alpha formulas — related but a separate task. Worth a one-line acknowledgment so a reviewer doesn't think they were missed: **FinMem, TradingGPT, FinAgent, FinGPT**, etc.

---

## ⚠️ Naming traps

1. **Two different "AutoAlpha"** — Zhang 2020 (evolutionary) vs. Kou 2024 (LLM). Same name, unrelated papers.
2. **Two different "AlphaEvolve"** — Cui 2021 (alpha mining, GP) vs. Novikov 2025 (DeepMind coding agent). Totally unrelated.

---

## CogAlpha at a glance (the paper that triggered this)

- **Core contribution:** *code-based evolution* (Python functions, not symbolic formulas) — argues formula/GP methods are confined to "shallow regions" of the search space.
- **Machinery:** 7-level / 21-agent hierarchy (coverage) + 5 paraphrasing modes (diversity) + multi-agent quality checker (Code Quality / Repair / Judge / Logic) + adaptive generation (feedback) + "thinking evolution" (LLM-driven mutation/crossover).
- **Scale:** small/quality-first — initial pool 80, ~96 children/generation, **~20 alphas in the final portfolio**. No "million-scale" claim.
- **Data:** OHLCV only; CSI300 primary (+CSI500/S&P500/HSI/HSCI); train 2011–19 / val 2020 / test 2021–24; Qlib; gpt-oss-120b; LightGBM combination.
- **Results:** beats 21 baselines on CSI300 (IC 0.059, RankIC 0.081, IR 1.90). The reasoning model o3 was the *worst* LLM.
- **Notable omission:** **does not cite Alpha-GPT (1.0 or 2.0)** — only the same group's "Quant 4.0" survey. So it doesn't occupy the Alpha-GPT lineage.

---

## How our project is positioned

| Axis | CogAlpha | This project |
|---|---|---|
| Multi-agent shape | 7-level hierarchy + paraphrasing (coverage) | 3-persona adversarial **debate**: draft→review→revise (diversity + verification) |
| Decomposition | by market topic | by **stage**: idea → formula |
| Representation | code | **symbolic DSL** (fast, parseable, no-look-ahead, GP-amenable) — *candidate ablation axis vs. code* |
| Local search | LLM-driven mutation | classic **DEAP GP** (free, cheap breadth amplification) |
| Feedback loop | yes (adaptive generation) | future work |
| Data | OHLCV only | richer: fundamentals + ~200 OSAP signals |
| Market / rigor | China-centric, Qlib | US, **survivorship-free PIT**, careful look-ahead controls |
| Scale | quality-first (~20 final) | breadth-first (1k+, toward 1M) |

**Our differentiation in one line:** debate (vs. hierarchy), two-stage decomposition, richer/survivorship-free data, breadth-at-scale — and we can turn CogAlpha's *asserted* "code > formula" into a **controlled** ablation (debate/data/eval fixed, swap only representation), which the literature currently lacks.

---

## Reading priority

If reading only a few to position against:

1. **Alpha-GPT** (1.0/2.0) — our lineage.
2. **CogAlpha** — closest concurrent (code-evolution).
3. **AlphaAgent** — decay/regularization angle.
4. **AlphaGen** — the RL baseline everyone benchmarks.
5. *(optional)* **RD-Agent(Q)** and **AlphaJungle** — feedback-loop and MCTS angles our doc already references.
