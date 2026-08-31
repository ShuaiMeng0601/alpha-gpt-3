"""Alpha-GPT — the single entry point.

Run the whole pipeline end-to-end (generate -> verify -> evaluate -> filter -> backtest ->
report) into one timestamped run folder, streaming stage by stage so you can watch it run:

    python run.py --source debate --n 200            # full multi-agent debate run
    python run.py --source random --n 50             # free, no-API stress test
    python run.py --source random_price --n 200      # price/volume-only control (no fundamentals)
    python run.py --source gp --n 100                # genetic-programming baseline
    python run.py --source llm --n 200 --alphas-per-idea 5

Ablate across generation sources (random / gp / llm / debate), then tabulate the arms:

    python run.py compare                            # aggregate outputs/*/pipeline_summary.json

Rebuild a portfolio from an existing run's alpha DB without regenerating (to try new
filters):

    python run.py portfolio --db outputs/run_<ts>/alphas.db --min-ic 0.01
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os

from alpha_gpt.config import Config
from alpha_gpt.pipeline import DEFAULT_METHODS, run_pipeline


def _add_filter_args(p: argparse.ArgumentParser) -> None:
    """Survivor-filter knobs shared by the run and portfolio commands."""
    p.add_argument("--min-tstat", type=float, default=2.0,
                   help="min |t-stat| of the in-sample IC (primary overfitting gate)")
    p.add_argument("--min-ic", type=float, default=0.0, help="min |in-sample IC| floor")
    p.add_argument("--min-icir", type=float, default=0.0, help="min |in-sample ICIR|")
    p.add_argument("--min-sharpe", type=float, default=0.0, help="min |in-sample Sharpe|")
    p.add_argument("--max-turnover", type=float, default=None)
    p.add_argument("--top-k", type=int, default=0, help="cap # survivors (0 = no cap)")
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    p.add_argument("--out", default=None, help="output folder (default outputs/run_<timestamp>)")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Alpha-GPT pipeline — one command, one run folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Default command (bare `run.py --source ...`): the full pipeline.
    p.add_argument("--source", choices=["random", "random_price", "gp", "llm", "debate"], default="debate",
                   help="random (free) | gp (genetic programming, free) | llm (1 call/idea) | "
                        "debate (full multi-agent debate — economically grounded alphas)")
    p.add_argument("--n", type=int, default=200, help="number of ideas to generate")
    p.add_argument("--alphas-per-idea", type=int, default=5)
    p.add_argument("--model", default=None,
                   help="LLM model id, overriding LLM_MODEL/.env for this run. A bare id "
                        "(e.g. gpt-5.4-mini) uses OpenAI direct via OPENAI_API_KEY; a "
                        "provider-prefixed id (e.g. deepseek/deepseek-v3.2) uses OpenRouter")
    p.add_argument("--max-cost", type=float, default=None,
                   help="stop generating new ideas once estimated API spend (USD) reaches "
                        "this cap (in-flight ideas still finish; eval/backtest still run)")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed; omit for a fresh random draw each run (the resolved "
                        "seed is saved to pipeline_config.json for reproducibility)")
    p.add_argument("--no-parallel", action="store_false", dest="parallel",
                   help="disable thread-pooled generation/eval")
    p.add_argument("--gen-workers", type=int, default=32)
    p.add_argument("--eval-workers", type=int, default=8)
    p.add_argument("--subsample", type=int, default=3, help="date subsample for eval speed")
    _add_filter_args(p)

    sub = p.add_subparsers(dest="command")

    cmp_p = sub.add_parser("compare", help="aggregate outputs/*/pipeline_summary.json by source")
    cmp_p.add_argument("--outputs", default="outputs", help="directory of run folders")

    port_p = sub.add_parser("portfolio", help="rebuild a portfolio from an existing alpha DB")
    port_p.add_argument("--db", required=True, help="path to a run's alphas.db")
    _add_filter_args(port_p)
    return p


def _cmd_run(args: argparse.Namespace) -> None:
    config = Config()
    if args.model:
        config.llm_model = args.model
    run_pipeline(
        config, source=args.source, n_ideas=args.n, alphas_per_idea=args.alphas_per_idea,
        seed=args.seed, parallel=args.parallel, gen_workers=args.gen_workers,
        eval_workers=args.eval_workers, subsample=args.subsample,
        min_ic=args.min_ic, min_icir=args.min_icir, min_sharpe=args.min_sharpe,
        max_turnover=args.max_turnover, min_tstat=args.min_tstat,
        top_k=args.top_k, methods=args.methods, out_dir=args.out,
        max_cost=args.max_cost,
    )


def _cmd_compare(args: argparse.Namespace) -> None:
    from alpha_gpt.compare import run_compare
    run_compare(outputs_dir=args.outputs)


def _cmd_portfolio(args: argparse.Namespace) -> None:
    from alpha_gpt.experiment import compute_vw_benchmark, load_and_split
    from alpha_gpt.factory import db
    from alpha_gpt.pipeline import backtest_portfolio, print_portfolio_table

    config = Config()
    conn = db.connect(args.db)
    survivors = db.query(
        conn, status="ok", min_abs_tstat=args.min_tstat, min_abs_ic=args.min_ic,
        min_abs_icir=args.min_icir, min_abs_sharpe=args.min_sharpe,
        max_turnover=args.max_turnover, order_by="abs(is_tstat)", desc=True,
        limit=(args.top_k or None))
    print(f"DB stats: {db.stats(conn)}")
    print(f"Survivors after filter: {len(survivors)} "
          f"(min|t-stat|={args.min_tstat}, cap={args.top_k or 'none'})")
    if not survivors:
        print("No survivors — loosen filters.")
        return

    in_sample, test, _, pset = load_and_split(config)
    benchmark = compute_vw_benchmark(test)
    out = args.out or os.path.join(
        "outputs", f"portfolio_{datetime.datetime.now():%Y%m%d_%H%M%S}")
    filters_desc = f"min|t-stat|={args.min_tstat}, min|IC|={args.min_ic}, cap={args.top_k or 'none'}"
    port = backtest_portfolio(survivors, pset, test, in_sample, methods=args.methods,
                              out_dir=out, benchmark=benchmark, filters_desc=filters_desc)
    if port["rows"]:
        print_portfolio_table(port["rows"])
        b = port["best"]
        print(f"\nHeadline (chosen in-sample): {b['method']} (test Sharpe {b['sharpe']:.2f}, "
              f"IC {b['ic']:.4f}, AnnRet {b['annual_return']:.1%})")
    print(f"\nSaved report (portfolio_report.md + .json) + equity curves to {out}/")


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for noisy in ("httpx", "httpcore", "openai"):  # keep HTTP chatter out of the progress bars
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.command == "compare":
        _cmd_compare(args)
    elif args.command == "portfolio":
        _cmd_portfolio(args)
    else:
        _cmd_run(args)


if __name__ == "__main__":
    main()
