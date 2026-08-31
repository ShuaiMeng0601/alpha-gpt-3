"""Parallel factory run: thread-pool the LLM calls AND the evaluation.

Why threads (not processes): the panels are large (~GBs); processes would copy
them per worker and exhaust memory. Threads share one read-only copy. The LLM
calls are I/O-bound (huge win from threading) and numpy/scipy release the GIL
during the heavy eval ops, so threads still help there. Date-subsampling makes
each eval cheaper (filter ranking only needs approximate metrics).
"""

import concurrent.futures as cf
import datetime
import json
import logging
import os
import time

from tqdm import tqdm

from alpha_gpt.factory import db
from alpha_gpt.factory.run import compute_alpha_metrics, estimate_cost
from alpha_gpt.factory.verifier import verify_expression

logger = logging.getLogger(__name__)


def _eval_task(raw, meta, pset, in_sample, subsample, verify_kwargs):
    vr = verify_expression(raw, pset, in_sample.panels, **verify_kwargs)
    h = db.expr_hash(vr.normalized) if vr.normalized else None
    # Rejects carry enough context to be persisted (status=reason) for later diagnosis.
    reject_base = {
        "hash": h, "expression": vr.normalized or raw, "raw_expression": raw,
        "n_terminals": vr.n_terminals, "depth": vr.depth, "coverage": vr.coverage, **meta,
    }
    if not vr.ok:
        return {"status": vr.reason, **reject_base}
    try:
        metrics = compute_alpha_metrics(vr, pset, in_sample, subsample)
        return {
            "status": "ok", "hash": h, "expression": vr.normalized, "raw_expression": raw,
            "n_terminals": vr.n_terminals, "depth": vr.depth, **metrics, **meta,
        }
    except Exception:
        return {"status": "metrics_error", **reject_base}


def run_factory_parallel(generator, pset, in_sample, conn, n_ideas, *, source,
                         gen_workers=10, eval_workers=8, subsample=3, verify_kwargs=None,
                         show_progress=False, out_dir=None):
    verify_kwargs = verify_kwargs or {}
    t0 = time.time()

    # --- Phase 1: generation (threads for LLM/debate; instant for random) ---
    # Both LLM and debate generators expose `_one(terminals)`; the debate just does far more
    # work per call (a full multi-agent debate). They thread-pool identically here — ideas
    # are independent, so `gen_workers` debates run concurrently.
    #
    # Generation is the ONLY paid phase; verify/eval/store below are free. So we persist each
    # generated batch to `generated.jsonl` the moment it lands — a crash / Ctrl-C / OOM during
    # the (possibly hours-long) generation or eval then preserves the paid expressions instead
    # of discarding the whole run. Best-effort: a logging failure never interrupts generation.
    gen_log = None
    if out_dir and source in ("llm", "debate"):
        try:
            gen_log = open(os.path.join(out_dir, "generated.jsonl"), "a")
        except Exception:
            gen_log = None
    batches = []
    if source in ("llm", "debate"):
        # One independent terminal subset per idea — drawn up front (single-threaded) so a
        # run's draws depend only on the generator's seed, not on thread scheduling.
        subsets = [generator.sample_subset() for _ in range(n_ideas)]
        with cf.ThreadPoolExecutor(max_workers=gen_workers) as ex:
            for b in tqdm(ex.map(generator._one, subsets), total=n_ideas,
                          desc="  generate", unit="idea", disable=not show_progress):
                if b is not None:
                    batches.append(b)
                    if gen_log is not None:
                        try:
                            gen_log.write(json.dumps({
                                "idea": b.idea, "hypothesis": b.hypothesis, "source": b.source,
                                "model": b.model, "terminals": b.terminals,
                                "confidence": b.confidence, "expressions": b.expressions,
                                "prompt_tokens": b.prompt_tokens,
                                "completion_tokens": b.completion_tokens}) + "\n")
                            gen_log.flush()  # durable per-batch, so a crash keeps paid work
                        except Exception:
                            pass
        if gen_log is not None:
            gen_log.close()
        logger.info(f"generation: {len(batches)}/{n_ideas} ideas in {time.time() - t0:.0f}s")
    else:
        batches = list(generator.generate(n_ideas))
    t_gen = time.time()

    # --- Phase 2: flatten to per-expression tasks (with token attribution) ---
    tasks = []
    for b in batches:
        n = max(1, len(b.expressions))
        meta_base = {
            "idea": b.idea, "hypothesis": b.hypothesis, "source": b.source, "model": b.model,
            # The subset the idea was SHOWN and its self-rated confidence, carried onto every
            # alpha it produced: that is what makes "did subsetting flatten terminal usage?"
            # and "does confidence predict out-of-sample IC?" answerable from the DB alone.
            "offered_terminals": ",".join(b.terminals),
            "confidence": b.confidence,
            "gen_prompt_tokens": round((b.prompt_tokens or 0) / n),
            "gen_completion_tokens": round((b.completion_tokens or 0) / n),
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        }
        for raw in b.expressions:
            tasks.append((raw, dict(meta_base)))

    # --- Phase 3: parallel evaluation (threads share read-only panels) ---
    results = []
    with cf.ThreadPoolExecutor(max_workers=eval_workers) as ex:
        futs = [ex.submit(_eval_task, raw, meta, pset, in_sample, subsample, verify_kwargs)
                for raw, meta in tasks]
        for i, f in enumerate(tqdm(cf.as_completed(futs), total=len(tasks),
                                   desc="  evaluate", unit="expr", disable=not show_progress)):
            results.append(f.result())
            if not show_progress and (i + 1) % 250 == 0:
                logger.info(f"evaluated {i + 1}/{len(tasks)}")
    t_eval = time.time()

    # --- Phase 4: store (single-writer, dedup) ---
    seen, n_ok, n_dup, rejects = set(), 0, 0, {}
    for r in tqdm(results, desc="  store", unit="rec", disable=not show_progress):
        h = r.get("hash")
        if h and h in seen:
            n_dup += 1; continue
        if h:
            seen.add(h)
        if r["status"] != "ok":
            rejects[r["status"]] = rejects.get(r["status"], 0) + 1
            # Persist the reject (status=reason, no metrics) so failure modes stay
            # diagnosable after the run; every reader filters status="ok". The hash lives
            # in a "reject:" namespace so a stale reject row can never shadow a later ok
            # row for the same expression (hash is UNIQUE + OR IGNORE).
            reject_key = r.get("expression") or r.get("raw_expression") or ""
            db.insert_alpha(conn, {**r, "hash": db.expr_hash("reject:" + reject_key)})
            continue
        if db.insert_alpha(conn, r):
            n_ok += 1
        else:
            n_dup += 1
    conn.commit()  # single commit for the whole store phase (no per-row fsync)

    pt = getattr(generator, "total_prompt_tokens", 0)
    ct = getattr(generator, "total_completion_tokens", 0)
    cost = estimate_cost(pt, ct, model=getattr(generator, "model", None))
    return {
        "ideas": n_ideas, "batches": len(batches), "evaluated": len(tasks),
        "stored": n_ok, "dup": n_dup, "rejects": rejects,
        "llm_calls": getattr(generator, "n_calls", 0),
        "prompt_tokens": pt, "completion_tokens": ct, "est_cost_usd": round(cost, 4),
        "cost_per_stored_alpha_usd": round(cost / n_ok, 6) if n_ok else 0.0,
        "gen_seconds": round(t_gen - t0), "eval_seconds": round(t_eval - t_gen),
        "total_seconds": round(time.time() - t0),
    }
