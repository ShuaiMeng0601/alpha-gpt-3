"""The generation loop: generate -> verify -> evaluate (train+val) -> store.

Selection metrics are TRAIN/VAL only; the test split is never touched here so the
later portfolio backtest stays honest.
"""

import datetime
import logging

from tqdm import tqdm

from alpha_gpt.evaluate.metrics import compute_turnover
from alpha_gpt.factory import db
from alpha_gpt.factory.fast_metrics import summarize_full
from alpha_gpt.factory.verifier import verify_expression

logger = logging.getLogger(__name__)

# Prices (USD per 1M tokens) keyed by the exact model id used. VERIFY current pricing before a
# paid run (openrouter.ai/models or platform.openai.com/pricing). An unknown model falls back to
# DEFAULT_PRICE, which is set to the MOST EXPENSIVE known tier on purpose: cost telemetry should
# fail SAFE (over-report), never silently under-report and let a run blow past budget.
PRICES = {
    "deepseek/deepseek-v3.2": {"prompt": 0.252, "completion": 0.378},
    "gpt-5.4-mini": {"prompt": 0.75, "completion": 4.50},          # OpenAI direct
    "openai/gpt-5.4-mini": {"prompt": 0.75, "completion": 4.50},   # via OpenRouter
}
DEFAULT_PRICE = {"prompt": 5.0, "completion": 30.0}  # conservative (gpt-5.5-tier) fail-safe


def estimate_cost(prompt_tokens: int, completion_tokens: int, price=None, model=None) -> float:
    if price is None:
        price = PRICES.get(model)
        if price is None:
            if model:
                logger.warning(f"No price for model {model!r}; using DEFAULT_PRICE (cost may be off).")
            price = DEFAULT_PRICE
    return prompt_tokens / 1e6 * price["prompt"] + completion_tokens / 1e6 * price["completion"]


def _subsample(df, k):
    """Take every k-th date (k>1) to speed up selection metrics; identity otherwise."""
    return df.iloc[::k] if k and k > 1 else df


def compute_alpha_metrics(vr, pset, in_sample, subsample: int = 1) -> dict:
    """In-sample selection metrics for a verified alpha (the single source of truth for
    what the factory stores per alpha). Test is never touched here. ``vr.signal`` was
    already evaluated on the in-sample panels by the verifier, so no re-evaluation is needed.

    ``subsample`` thins the date axis for speed; the t-stat corrects for it via
    ``ic_days_scale`` so the stored value reflects the full in-sample length. Turnover is
    always measured on the full (un-thinned) in-sample signal.
    """
    ic, icir, tstat, sharpe, ann, dd = summarize_full(
        _subsample(vr.signal, subsample),
        _subsample(in_sample.forward_returns, subsample),
        ic_days_scale=max(1, subsample),
    )
    return {
        "is_ic": ic, "is_icir": icir, "is_tstat": tstat, "is_sharpe": sharpe,
        "is_annual_return": ann, "is_max_drawdown": dd,
        "turnover": compute_turnover(vr.signal), "coverage": vr.coverage,
    }


def run_factory(generator, pset, in_sample, conn, n_ideas, verify_kwargs=None,
                log_every: int = 10, show_progress: bool = False) -> dict:
    verify_kwargs = verify_kwargs or {}
    seen: set[str] = set()
    n_seen = n_ok = n_dup = 0
    rejects: dict[str, int] = {}

    for bi, batch in enumerate(tqdm(generator.generate(n_ideas), total=n_ideas,
                                    desc="  ideas", unit="idea", disable=not show_progress)):
        n_expr = max(1, len(batch.expressions))
        per_pt = round((batch.prompt_tokens or 0) / n_expr)
        per_ct = round((batch.completion_tokens or 0) / n_expr)
        for raw in batch.expressions:
            n_seen += 1
            vr = verify_expression(raw, pset, in_sample.panels, **verify_kwargs)

            # In-run dedup (skip re-evaluating repeats, including rejects).
            h = db.expr_hash(vr.normalized) if vr.normalized else None
            if h is not None:
                if h in seen:
                    n_dup += 1
                    continue
                seen.add(h)

            def _reject_rec(reason):
                # Persist the reject (status=reason, no metrics) so failure modes stay
                # diagnosable after the run; every reader filters status="ok". The hash
                # lives in a "reject:" namespace so a stale reject row can never shadow
                # a later ok row for the same expression (hash is UNIQUE + OR IGNORE).
                return {
                    "hash": db.expr_hash("reject:" + (vr.normalized or raw)),
                    "expression": vr.normalized or raw,
                    "raw_expression": raw,
                    "idea": batch.idea, "hypothesis": batch.hypothesis,
                    "source": batch.source, "model": batch.model,
                    "n_terminals": vr.n_terminals, "depth": vr.depth,
                    "coverage": vr.coverage,
                    "gen_prompt_tokens": per_pt, "gen_completion_tokens": per_ct,
                    "status": reason,
                    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                }

            if not vr.ok:
                rejects[vr.reason] = rejects.get(vr.reason, 0) + 1
                db.insert_alpha(conn, _reject_rec(vr.reason))
                continue

            try:
                metrics = compute_alpha_metrics(vr, pset, in_sample)
            except Exception:
                rejects["metrics_error"] = rejects.get("metrics_error", 0) + 1
                db.insert_alpha(conn, _reject_rec("metrics_error"))
                continue

            rec = {
                "hash": h,
                "expression": vr.normalized,
                "raw_expression": raw,
                "idea": batch.idea,
                "hypothesis": batch.hypothesis,
                "source": batch.source,
                "model": batch.model,
                "n_terminals": vr.n_terminals,
                "depth": vr.depth,
                "gen_prompt_tokens": per_pt,
                "gen_completion_tokens": per_ct,
                "status": "ok",
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                **metrics,
            }
            if db.insert_alpha(conn, rec):
                n_ok += 1
            else:
                n_dup += 1

        if (bi + 1) % log_every == 0:
            conn.commit()  # periodic durability without an fsync per row
            if not show_progress:
                logger.info(f"ideas={bi + 1} seen={n_seen} stored={n_ok} dup={n_dup}")

    conn.commit()  # flush the final batch
    total_pt = getattr(generator, "total_prompt_tokens", 0)
    total_ct = getattr(generator, "total_completion_tokens", 0)
    total_tok = total_pt + total_ct
    cost = estimate_cost(total_pt, total_ct, model=getattr(generator, "model", None))
    return {
        "ideas": n_ideas, "seen": n_seen, "stored": n_ok, "dup": n_dup, "rejects": rejects,
        "llm_calls": getattr(generator, "n_calls", 0),
        "prompt_tokens": total_pt, "completion_tokens": total_ct, "total_tokens": total_tok,
        "est_cost_usd": round(cost, 4),
        "tokens_per_stored_alpha": round(total_tok / n_ok, 1) if n_ok else 0,
        "cost_per_stored_alpha_usd": round(cost / n_ok, 6) if n_ok else 0.0,
    }
