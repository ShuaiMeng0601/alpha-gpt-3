"""Ingest OSAP (Open Source Asset Pricing) firm-level signals into daily panels.

Input: the wide monthly CSV (columns: permno, yyyymm, <~200 signal columns>) from
openassetpricing.com. Each signal is an END-OF-MONTH value usable to predict the
NEXT month, so the point-in-time control is: stamp each observation active on the
day AFTER its month-end, forward-fill to daily, then mask to universe membership.
This guarantees no look-ahead (a month-M signal only affects month-(M+1) onward).

Output: data/panels/osap_<signal>.parquet (+ _osap_manifest.json).
"""

import json
import os
import re

import pandas as pd

from alpha_gpt.data.coverage import panel_coverage, qualifies
from alpha_gpt.data.loader import ffill_to_membership

CHUNK = 500_000
_NON_SIGNAL = {"permno", "yyyymm", "date", "eom", "id", "ret", "me", "mve_c"}


def _sanitize(name: str) -> str:
    return "osap_" + re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


def _detect_cols(columns) -> tuple[str, str]:
    permno = next((c for c in ("permno", "PERMNO") if c in columns), None)
    date = next((c for c in ("yyyymm", "date", "DATE", "eom") if c in columns), None)
    if permno is None or date is None:
        raise ValueError(f"Could not find permno/date columns in OSAP file: {list(columns)[:8]}")
    return permno, date


def _month_end_plus1(series: pd.Series) -> pd.Series:
    """Parse a yyyymm (or date) column to (month-end + 1 day) = active date."""
    s = series.astype("string").str.replace("-", "", regex=False).str.slice(0, 6)
    dt = pd.to_datetime(s, format="%Y%m", errors="coerce") + pd.offsets.MonthEnd(0)
    return dt + pd.Timedelta(days=1)


def ingest_osap(
    osap_csv: str,
    out_dir: str,
    membership: pd.DataFrame,
    min_coverage: float = 0.20,
    min_breadth: int = 100,
    max_signals: int | None = None,
) -> dict:
    """Read OSAP wide CSV → daily masked panels for qualifying signals."""
    universe = set(int(p) for p in membership.columns)
    dmin, dmax = membership.index.min(), membership.index.max()

    # Pass A: stream + filter to universe and date span (keeps memory bounded).
    head = pd.read_csv(osap_csv, nrows=5)
    permno_col, date_col = _detect_cols(head.columns)
    signal_cols = [c for c in head.columns if c.lower() not in _NON_SIGNAL]

    kept = []
    for ch in pd.read_csv(osap_csv, chunksize=CHUNK, low_memory=False):
        ch = ch[ch[permno_col].isin(universe)].copy()
        if ch.empty:
            continue
        ch["active"] = _month_end_plus1(ch[date_col])
        ch = ch[(ch["active"] >= dmin) & (ch["active"] <= dmax + pd.Timedelta(days=40))]
        if not ch.empty:
            kept.append(ch[[permno_col, "active"] + signal_cols])
    if not kept:
        raise ValueError("No OSAP rows overlap the universe/date range.")
    df = pd.concat(kept, ignore_index=True)
    df = df.rename(columns={permno_col: "PERMNO"})
    df = df.drop_duplicates(["PERMNO", "active"], keep="last")
    print(f"  OSAP rows in universe: {len(df):,}; candidate signals: {len(signal_cols)}")

    # Pass B: per signal → monthly pivot → daily ffill (active-dated) → mask.
    manifest = []
    for i, sig in enumerate(signal_cols):
        if max_signals is not None and len([m for m in manifest if m["kept"]]) >= max_signals:
            break
        sub = df[["PERMNO", "active", sig]].dropna(subset=[sig])
        if sub.empty:
            continue
        sub[sig] = pd.to_numeric(sub[sig], errors="coerce")
        monthly = sub.pivot_table(index="active", columns="PERMNO", values=sig, aggfunc="last")
        daily = ffill_to_membership(monthly, membership)
        rec = {"field": _sanitize(sig)}
        rec.update(panel_coverage(daily, membership))
        ok = qualifies(rec, min_coverage, min_breadth)
        rec["kept"] = bool(ok)
        manifest.append(rec)
        if ok:
            daily.to_parquet(os.path.join(out_dir, f"{_sanitize(sig)}.parquet"))
        if (i + 1) % 25 == 0:
            print(f"  processed {i + 1}/{len(signal_cols)} signals "
                  f"({sum(m['kept'] for m in manifest)} kept)")

    with open(os.path.join(out_dir, "_osap_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    n_kept = sum(m["kept"] for m in manifest)
    print(f"  OSAP: kept {n_kept}/{len(manifest)} signals (coverage>={min_coverage}, "
          f"breadth>={min_breadth}/day)")
    return {"kept": n_kept, "evaluated": len(manifest)}
