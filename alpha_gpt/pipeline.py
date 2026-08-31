"""Single staged entry point for the whole alpha pipeline.

Runs every stage end-to-end with a progress bar per stage and writes ALL artifacts
into one timestamped run folder (outputs/run_<ts>/) so a run is self-contained and
easy to debug:

  outputs/run_<ts>/
    pipeline_config.json        # exact parameters of this run
    pipeline.log                # full log (tee'd from the console)
    alphas.db                   # the run's own alpha database
    factory_summary.json        # generate/verify/evaluate funnel + cost
    survivors.csv               # alphas that passed the validation filter
    portfolio_report.md         # out-of-sample comparison table
    portfolio_results.json
    portfolio_equity_curves.png
    pipeline_summary.json        # everything tied together

Stages:
  1. load     — load panels, split train/val/test, screen GP terminals
  2. factory  — generate (LLM/random) -> verify -> train/val metrics -> dedup/store
  3. filter   — select survivors by the in-sample IC |t-stat| gate
  4. backtest — re-evaluate survivors on the HELD-OUT test split, combine + backtest OOS
  5. report   — write the report, equity curves, and run summary

Driven by ``run.py`` (the single CLI), e.g.:
    python run.py --source llm --n 200 --alphas-per-idea 5
    python run.py --source random --n 50        # free stress test
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import random
import time

import pandas as pd
from tqdm import tqdm

from alpha_gpt.evaluate.metrics import compute_ic, compute_icir, compute_turnover
from alpha_gpt.evaluate.neutralize import characteristic_neutralize
from alpha_gpt.evaluate.visualize import plot_equity_curves, plot_gate_ablation
from alpha_gpt.evaluate.backtester import backtest_alpha
from alpha_gpt.config import Config
from alpha_gpt.experiment import compute_vw_benchmark, load_and_split
from alpha_gpt.factory import db
from alpha_gpt.factory.generators import (
    DebateGenerator, GPGenerator, LLMGenerator, RandomGenerator, RandomPriceGenerator)
from alpha_gpt.factory.parallel import run_factory_parallel
from alpha_gpt.factory.run import run_factory
from alpha_gpt.llm import make_client
from alpha_gpt.portfolio.construct import build_signals, combine, fit_combiners

logger = logging.getLogger(__name__)

# `orthogonalized` is intentionally NOT a default: with positively-correlated oriented
# alphas its averaged (un-renormalized) residuals collapse toward the single highest-IC
# alpha, so it inherits that one alpha's high variance rather than diversifying. It remains
# available via --methods for anyone who wants it. (No look-ahead — see portfolio tests.)
DEFAULT_METHODS = ["equal"]
N_STAGES = 5


def _stage(i: int, title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n[STAGE {i}/{N_STAGES}] {title}\n{bar}", flush=True)


# ---------------------------------------------------------------------------
# Portfolio stage (also reused by the run.py `portfolio` command — single source of truth)
# ---------------------------------------------------------------------------

def _portfolio_metrics(sig, fwd):
    ic = compute_ic(sig, fwd)
    bt = backtest_alpha(sig, fwd)
    return {
        "ic": float(ic.mean()) if not ic.empty else 0.0,
        "icir": compute_icir(ic),
        "sharpe": bt.sharpe,
        "annual_return": bt.annual_return,
        "max_drawdown": bt.max_drawdown,
        "turnover": compute_turnover(sig),
    }, bt


def _method_scores(survivors, pset, fit_split, weights, methods, betas,
                   show_progress: bool = False) -> dict:
    """Sharpe of each combination method evaluated ON THE IN-SAMPLE split.

    Used ONLY to decide which method to highlight as the headline — never to compute the
    numbers reported for it (those stay on test). Choosing the headline in-sample instead of
    by the max test Sharpe avoids a multiple-comparisons optimization bias on the single test
    window. Returns {method: in_sample_sharpe}; empty if no fit split is available.
    """
    if fit_split is None:
        return {}
    prog = (lambda it: tqdm(it, desc="  score methods (in-sample)", unit="alpha")) if show_progress else None
    fit_signals = build_signals(survivors, pset, fit_split.panels, progress=prog)
    if not fit_signals:
        return {}
    fwd = fit_split.forward_returns
    scores = {}
    for method in methods:
        if method == "orthogonalized" and betas is None:
            continue
        comp, _ = combine(fit_signals, method=method, weights=weights, betas=betas)
        if comp is None:
            continue
        m, _bt = _portfolio_metrics(comp, fwd)
        scores[method] = m["sharpe"]
    return scores


# Trading-data panels — excluded from the neutralization basis (everything else is a
# characteristic the generator can consume as a terminal).
_PRICE_PANELS = {"open", "high", "low", "close", "volume", "returns", "bid", "ask",
                 "dollar_volume", "num_trades", "price", "forward_returns"}


def _neutralization_chars(panels: dict) -> dict:
    """Neutralization basis for the factor-neutral diagnostic row: EVERY fundamental
    characteristic panel the generator could consume, plus low-vol and short-term reversal
    derived from prices. Measured on the random baseline: a small 5-factor basis does NOT
    span a random fundamental book (its neutralized Sharpe even rises), while this full
    input basis collapses it (0.89 -> 0.29 on 2021-25) — so the residual row measures
    signal BEYOND static tilts on the book's own inputs. (Characteristics enter as per-day
    cross-sectional ranks, so monotone transforms like log(mcap) are equivalent.)"""
    chars = {k: v for k, v in panels.items() if k not in _PRICE_PANELS}
    if "returns" in panels:
        chars["vol60"] = panels["returns"].rolling(60, min_periods=20).std()
    if "close" in panels:
        chars["ret21"] = panels["close"].pct_change(21)
    return chars


def _select_headline(rows: list[dict], fit_scores: dict) -> dict | None:
    """Pick the headline row. Prefer the method that scored best IN-SAMPLE so the highlighted
    method is not selected by peeking at test (avoids optimization bias). Falls back to the
    max test Sharpe only when no in-sample scores exist.
    """
    if not rows:
        return None
    if fit_scores:
        eligible = [r for r in rows if r["method"] in fit_scores]
        if eligible:
            return max(eligible, key=lambda r: fit_scores[r["method"]])
    return max(rows, key=lambda r: r["sharpe"])


def print_portfolio_table(rows: list[dict]) -> None:
    print("\n" + "=" * 92)
    print("PORTFOLIO COMPARISON — out-of-sample (test)")
    print("=" * 92)
    print(f"{'method':<22}{'#':>5}{'IC':>9}{'ICIR':>9}{'Sharpe':>9}{'AnnRet':>10}{'MaxDD':>9}{'Turnover':>10}")
    print("-" * 92)
    for r in rows:
        print(f"{r['method']:<22}{r['n_alphas']:>5}{r['ic']:>9.4f}{r['icir']:>9.3f}"
              f"{r['sharpe']:>9.2f}{r['annual_return']:>9.1%}{r['max_drawdown']:>9.1%}"
              f"{r['turnover']:>10.3f}")


def _write_report(path: str, rows: list[dict], best: dict | None, n_survivors: int, filters: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Alpha Portfolio Report", "",
        f"- **Generated:** {ts}",
        f"- **Survivors:** {n_survivors}  ({filters})",
        "- **Backtest:** long-short quintile, out-of-sample (test split), **gross of costs**",
        "", "## Out-of-sample comparison", "",
        "| Method | # | IC | ICIR (daily) | Sharpe (ann.) | Ann. Return | Max DD | Turnover |",
        "|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['n_alphas']} | {r['ic']:.4f} | {r['icir']:.3f} | "
            f"{r['sharpe']:.2f} | {r['annual_return']:.1%} | {r['max_drawdown']:.1%} | "
            f"{r['turnover']:.3f} |"
        )
    if best:
        lines += ["",
                  f"**Headline method** (chosen on validation) — `{best['method']}`: "
                  f"test Sharpe {best['sharpe']:.2f}, IC {best['ic']:.4f}, "
                  f"Ann. Return {best['annual_return']:.1%}."]
    lines += ["", "![equity curves](portfolio_equity_curves.png)", "",
              "> Caveats: gross of transaction costs; single test window. Daily ICIR annualizes "
              "by ×√252. Survivor selection is on validation; orthogonalization parameters are "
              "also fit on validation and applied to the held-out test split, and the headline "
              "method is chosen on validation — so no test information enters construction or "
              "method selection (no look-ahead, no test-set cherry-picking)."]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def backtest_portfolio(
    survivors: list[dict], pset, test, fit_split=None, *, methods=None,
    out_dir: str, benchmark=None, filters_desc: str = "", show_progress: bool = True,
    include_best_single: bool = True,
) -> dict:
    """Re-evaluate survivors on `test`, combine + backtest each method, write artifacts.

    `fit_split` (the in-sample split) is used to FIT the learned combiner (orthogonalization
    betas) which is then applied to test — no look-ahead. Returns a summary dict. Shared by
    the full pipeline and the run.py `portfolio` command.
    """
    methods = methods or DEFAULT_METHODS
    os.makedirs(out_dir, exist_ok=True)

    prog = (lambda it: tqdm(it, desc="  re-eval on test", unit="alpha")) if show_progress else None
    signals = build_signals(survivors, pset, test.panels, progress=prog)
    logger.info(f"Re-evaluated {len(signals)} survivor signals on the TEST split.")
    if not signals:
        return {"rows": [], "best": None, "n_survivors": len(survivors), "n_signals": 0}

    fwd = test.forward_returns
    weights = {r["id"]: r.get("is_ic", 0.0) for r in survivors}

    # Fit the LEARNED combiner (orthogonalization betas) on the IN-SAMPLE split, then apply it
    # fixed to test — so no test information enters the construction. (equal uses no fitted
    # parameters; signals keep their native orientation — no sign-flipping. `weights` is
    # |in-sample IC|, used only to ORDER the orthogonalization and pick the best-single row.)
    betas = None
    if fit_split is not None and "orthogonalized" in methods:
        fprog = (lambda it: tqdm(it, desc="  fit combiners (in-sample)", unit="alpha")) if show_progress else None
        betas = fit_combiners(survivors, pset, fit_split.panels, weights, progress=fprog)

    rows, backtests = [], {}

    # Baseline: the single best alpha, as a NUMERIC reference row only. It is deliberately
    # NOT added to `backtests`, so it stays off the equity curve — one alpha is too
    # high-variance for its curve to be a meaningful comparison (it just adds noise).
    # Skipped when `include_best_single=False` (e.g. a debate-focused run that wants the
    # table to show only the equal-weight portfolio vs the market benchmark).
    if include_best_single:
        best_id = max(signals, key=lambda i: abs(weights.get(i, 0.0)))
        m, _ = _portfolio_metrics(signals[best_id], fwd)
        m.update(method="best_single_alpha", n_alphas=1)
        rows.append(m)

    equal_comp, equal_n = None, 0
    for method in tqdm(methods, desc="  backtest", unit="method", disable=not show_progress):
        if method == "orthogonalized" and betas is None:
            logger.warning(f"skipping {method}: no validation split provided to fit it")
            continue
        comp, sel = combine(signals, method=method, weights=weights, betas=betas)
        if comp is None:
            continue
        if method == "equal":
            equal_comp, equal_n = comp, len(sel)
        m, bt = _portfolio_metrics(comp, fwd)
        m.update(method=method, n_alphas=len(sel))
        rows.append(m); backtests[method] = bt

    # Choose the headline method IN-SAMPLE (not by the max test Sharpe) — no peeking.
    fit_scores = _method_scores(survivors, pset, fit_split, weights, methods, betas,
                                show_progress=show_progress)
    best = _select_headline(rows, fit_scores)

    # Factor-neutral DIAGNOSTIC row: the equal book with generic characteristic tilts
    # (size/value/quality/low-vol/reversal) regressed out of the composite each day.
    # Shows how much Sharpe is factor exposure vs residual signal — a random fundamental
    # book collapses here, a genuinely novel one doesn't. Appended AFTER headline
    # selection, so it can never be picked as the headline.
    chars = _neutralization_chars(test.panels)
    if equal_comp is not None and chars:
        resid = characteristic_neutralize(equal_comp, chars)
        m, bt = _portfolio_metrics(resid, fwd)
        m.update(method="equal_factor_neutral", n_alphas=equal_n)
        rows.append(m); backtests["equal_factor_neutral"] = bt

    plot_equity_curves(backtests, os.path.join(out_dir, "portfolio_equity_curves.png"), benchmark=benchmark)
    with open(os.path.join(out_dir, "portfolio_results.json"), "w") as f:
        json.dump({"n_survivors": len(survivors), "n_signals": len(signals),
                   "filters": filters_desc, "rows": rows}, f, indent=2)
    _write_report(os.path.join(out_dir, "portfolio_report.md"), rows, best, len(survivors), filters_desc)
    return {"rows": rows, "best": best, "n_survivors": len(survivors), "n_signals": len(signals)}


# ---------------------------------------------------------------------------
# Full pipeline orchestrator
# ---------------------------------------------------------------------------

def _save_survivors_csv(path: str, survivors: list[dict]) -> None:
    cols = ["id", "expression", "source", "idea", "confidence", "offered_terminals",
            "is_ic", "is_icir", "is_tstat",
            "is_sharpe", "turnover", "coverage", "n_terminals", "depth"]
    df = pd.DataFrame(survivors)
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(path, index=False)


def backtest_gate_ablation(conn, pset, test, fit_split, out_dir, *, thresholds=None,
                           base_filters=None, benchmark=None, show_progress=False) -> list[dict]:
    """Ablate the t-stat selection gate: for a sweep of |t-stat| thresholds, build the
    equal-weight portfolio of the alphas that clear each bar and backtest it on the held-out
    test split. Writes `gate_ablation.png` (one equity curve per threshold) and returns the
    per-threshold summary rows. This directly answers "does raising the t-stat bar improve
    out-of-sample Sharpe?" — a knob-vs-OOS-Sharpe curve, no extra API calls.
    """
    thresholds = thresholds if thresholds is not None else [0.0, 1.0, 2.0, 3.0, 4.0]
    base = dict(base_filters or {})
    base.pop("min_abs_tstat", None)  # we sweep this axis ourselves

    # Evaluate every candidate that clears the LOOSEST bar once on test, then subset by
    # threshold (an alpha's signal is identical across bars — no need to re-evaluate).
    loosest = min(thresholds)
    pool = db.query(conn, status="ok", min_abs_tstat=loosest, order_by="abs(is_tstat)", **base)
    if not pool:
        return []
    prog = (lambda it: tqdm(it, desc="  gate ablation (test)", unit="alpha")) if show_progress else None
    sig_by_id = build_signals(pool, pset, test.panels, progress=prog)
    fwd = test.forward_returns

    rows, curves = [], {}
    for tau in sorted(thresholds):
        ids = [r["id"] for r in pool if abs(r.get("is_tstat") or 0.0) >= tau and r["id"] in sig_by_id]
        if not ids:
            continue
        comp, sel = combine({i: sig_by_id[i] for i in ids}, method="equal")
        if comp is None:
            continue
        m, bt = _portfolio_metrics(comp, fwd)
        rows.append({"min_tstat": tau, "n_alphas": len(sel), "sharpe": m["sharpe"],
                     "ic": m["ic"], "annual_return": m["annual_return"]})
        curves[f"|t|>={tau:g}  (n={len(sel)}, Sharpe {m['sharpe']:.2f})"] = bt

    if curves:
        plot_gate_ablation(curves, os.path.join(out_dir, "gate_ablation.png"), benchmark=benchmark)
    with open(os.path.join(out_dir, "gate_ablation.json"), "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def run_pipeline(
    config: Config, *, source: str = "llm", n_ideas: int = 200, alphas_per_idea: int = 5,
    seed: int | None = None, parallel: bool = True, gen_workers: int = 32, eval_workers: int = 8,
    subsample: int = 3, min_ic: float = 0.0, min_icir: float = 0.0, min_sharpe: float = 0.0,
    max_turnover=None, min_tstat: float = 2.0, top_k: int = 0,
    methods=None, out_dir: str | None = None, include_best_single: bool = True,
    max_cost: float | None = None,
) -> dict:
    """Run generate -> verify -> evaluate -> filter -> backtest -> report.

    Selection is on IN-SAMPLE metrics; the primary gate is ``min_tstat`` (|t-stat| of the mean
    in-sample IC). Test is touched once, by the final backtest + the gate-ablation sweep."""
    methods = methods or DEFAULT_METHODS
    # seed=None -> draw a fresh seed so each run is a NEW random draw; the resolved int is
    # recorded in pipeline_config.json below, so any run stays reproducible via --seed <that>.
    if seed is None:
        seed = random.SystemRandom().randrange(2**31)
    t0 = time.time()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_dir or os.path.join("outputs", f"run_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    # Tee logs into the run folder for debugging.
    fh = logging.FileHandler(os.path.join(out_dir, "pipeline.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    params = dict(
        source=source, n_ideas=n_ideas, alphas_per_idea=alphas_per_idea, seed=seed,
        parallel=parallel, gen_workers=gen_workers, eval_workers=eval_workers, subsample=subsample,
        min_ic=min_ic, min_icir=min_icir, min_sharpe=min_sharpe, max_turnover=max_turnover,
        min_tstat=min_tstat, top_k=top_k, methods=methods,
        model=config.llm_model if source in ("llm", "debate") else None,
        max_cost=max_cost,
        started=stamp,
    )
    with open(os.path.join(out_dir, "pipeline_config.json"), "w") as f:
        json.dump(params, f, indent=2)

    print(f"\n🏭  Alpha pipeline → {out_dir}/   (source={source}, ideas={n_ideas}×{alphas_per_idea}, seed={seed})")
    summary: dict = {"out_dir": out_dir, "params": params}

    try:
        # --- Stage 1: data ---
        _stage(1, "Load panels + split in-sample/test + screen terminals")
        in_sample, test, available_terminals, pset = load_and_split(config)
        summary["n_terminals"] = len(available_terminals)

        # --- Stage 2: factory (generate + verify + evaluate + store) ---
        _stage(2, f"Generate + verify + evaluate ({source}: {n_ideas} ideas × {alphas_per_idea})")
        db_path = os.path.join(out_dir, "alphas.db")
        conn = db.connect(db_path)
        if source == "random":
            gen = RandomGenerator(pset, alphas_per_idea=alphas_per_idea, seed=seed)
        elif source == "random_price":
            gen = RandomPriceGenerator(alphas_per_idea=alphas_per_idea, seed=seed)
        elif source == "gp":
            gen = GPGenerator(pset, in_sample, available_terminals, config, seed=seed)
        elif source == "debate":
            gen = DebateGenerator(make_client(config), config.llm_model, available_terminals,
                                  config, seed=seed, pset=pset, panels=in_sample.panels,
                                  out_dir=out_dir, diag_subsample=subsample,
                                  max_cost_usd=max_cost)
        else:
            gen = LLMGenerator(make_client(config), config.llm_model, available_terminals,
                               alphas_per_idea=alphas_per_idea, seed=seed,
                               max_cost_usd=max_cost, config=config)
        verify_kwargs = {"max_depth": config.verify_max_depth}
        if parallel:
            factory = run_factory_parallel(
                gen, pset, in_sample, conn, n_ideas, source=source,
                gen_workers=gen_workers, eval_workers=eval_workers,
                subsample=subsample, verify_kwargs=verify_kwargs, show_progress=True,
                out_dir=out_dir)
        else:
            factory = run_factory(gen, pset, in_sample, conn, n_ideas,
                                  verify_kwargs=verify_kwargs, show_progress=True)
        with open(os.path.join(out_dir, "factory_summary.json"), "w") as f:
            json.dump(factory, f, indent=2)
        summary["factory"] = factory
        print(f"  stored {factory['stored']} alphas  |  est. cost ${factory.get('est_cost_usd', 0):.4f}")

        # --- Stage 3: filter ---
        _stage(3, "Filter survivors (in-sample |t-stat| gate)")
        base_filters = dict(min_abs_ic=min_ic, min_abs_icir=min_icir, min_abs_sharpe=min_sharpe,
                            max_turnover=max_turnover)
        survivors = db.query(
            conn, status="ok", min_abs_tstat=min_tstat, **base_filters,
            order_by="abs(is_tstat)", desc=True, limit=(top_k or None))
        _save_survivors_csv(os.path.join(out_dir, "survivors.csv"), survivors)
        summary["n_survivors"] = len(survivors)
        print(f"  survivors: {len(survivors)} / {factory['stored']} stored "
              f"(min|t-stat|={min_tstat}, min|IC|={min_ic}, cap={top_k or 'none'})")
        if not survivors:
            print("  no survivors — loosen filters; stopping after the factory stage.")
            return _finish(summary, out_dir, fh, t0)

        # --- Stage 4: re-evaluate on test + combine + backtest ---
        _stage(4, "Re-evaluate survivors on held-out TEST + combine + backtest")
        benchmark = compute_vw_benchmark(test)
        filters_desc = f"min|t-stat|={min_tstat}, min|IC|={min_ic}, cap={top_k or 'none'}"
        port = backtest_portfolio(
            survivors, pset, test, in_sample, methods=methods,
            out_dir=out_dir, benchmark=benchmark, filters_desc=filters_desc, show_progress=True,
            include_best_single=include_best_single)
        summary["portfolio"] = port
        if port["rows"]:
            print_portfolio_table(port["rows"])
            b = port["best"]
            print(f"\nHeadline (chosen in-sample): {b['method']} (test Sharpe {b['sharpe']:.2f}, "
                  f"IC {b['ic']:.4f}, AnnRet {b['annual_return']:.1%})")

        # --- Stage 4b: does the t-stat gate actually help? sweep it on the held-out test ---
        gate = backtest_gate_ablation(conn, pset, test, in_sample, out_dir,
                                      base_filters=base_filters, benchmark=benchmark,
                                      show_progress=True)
        summary["gate_ablation"] = gate
        if gate:
            print("\n  t-stat gate → test Sharpe (equal-weight):")
            for g in gate:
                print(f"    |t|>={g['min_tstat']:>3g}: Sharpe {g['sharpe']:+.2f}  (n={g['n_alphas']})")

        # --- Stage 5: report / summary ---
        _stage(5, "Write report + run summary")
        return _finish(summary, out_dir, fh, t0)
    finally:
        # Always detach the file handler even if a stage raised.
        logging.getLogger().removeHandler(fh)
        fh.close()


def _finish(summary: dict, out_dir: str, fh, t0: float) -> dict:
    summary["elapsed_seconds"] = round(time.time() - t0)
    with open(os.path.join(out_dir, "pipeline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n✓ Done in {summary['elapsed_seconds']}s. All artifacts in {out_dir}/")
    for name in ("pipeline_config.json", "factory_summary.json", "survivors.csv",
                 "portfolio_report.md", "portfolio_results.json",
                 "portfolio_equity_curves.png", "gate_ablation.png", "gate_ablation.json",
                 "alphas.db", "pipeline.log"):
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            print(f"    {name}")
    return summary
