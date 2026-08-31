"""Build point-in-time, survivorship-free panel data from CRSP + Compustat.

Replaces the old static 500-stock sampler. The universe is selected annually
from TRAILING data (no full-sample look-ahead), names are retained through
delisting, and every panel is masked to point-in-time membership.

Two streaming passes over the large CRSP CSV keep memory bounded:
  Pass 1: per-(permno, year) cap + eligibility summary -> annual universe
  Pass 2: keep only member rows -> pivot to date x PERMNO panels -> mask

Outputs (data/panels/, date x PERMNO float64 parquet, one per field):
  close, open, high, low, volume, returns, price, shrout, market_cap,
  bid, ask, num_trades, dollar_volume, forward_returns, <compustat ratios...>
Plus data/universe/membership.parquet (bool date x PERMNO).

Usage:
    python scripts/prepare_data.py --universe-size 1500
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from alpha_gpt.config import Config
from alpha_gpt.data import universe as U
from alpha_gpt.data.loader import ffill_to_membership, parse_yyyymmdd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(RAW_DIR, "panels")
UNIV_DIR = os.path.join(RAW_DIR, "universe")

DATE_MIN = 20100101
DATE_MAX = 20251231   # track the raw CRSP extent (crsp_daily currently runs through 2025-12-31)
CHUNK = 3_000_000

# Pass 1: only what universe selection needs.
PASS1_COLS = ["PERMNO", "YYYYMMDD", "DlyCap", "SecurityHdrFlg"] + U.ELIG_FIELDS
# Pass 2: price/volume fields to turn into panels.
PASS2_COLS = [
    "PERMNO", "YYYYMMDD", "SecurityHdrFlg",
    "DlyClose", "DlyOpen", "DlyHigh", "DlyLow", "DlyVol", "DlyRet", "DlyPrc",
    "ShrOut", "DlyCap", "DlyBid", "DlyAsk", "DlyNumTrd", "DlyPrcVol",
]
FIELD_MAP = {
    "DlyClose": "close", "DlyOpen": "open", "DlyHigh": "high", "DlyLow": "low",
    "DlyVol": "volume", "DlyRet": "returns", "DlyPrc": "price",
    "ShrOut": "shrout", "DlyCap": "market_cap", "DlyBid": "bid", "DlyAsk": "ask",
    "DlyNumTrd": "num_trades", "DlyPrcVol": "dollar_volume",
}
NUMERIC_RAW = [c for c in PASS2_COLS if c not in ("PERMNO", "YYYYMMDD", "SecurityHdrFlg")]

RATIO_COLS = [
    "permno", "public_date", "bm", "pe_op_dil", "pe_exi", "ps", "pcf",
    "npm", "opmbd", "gpm", "roa", "roe", "roce", "debt_at", "curr_ratio",
    "quick_ratio", "cash_ratio", "de_ratio", "ptb", "accrual", "divyield",
    "at_turn", "inv_turn", "rect_turn", "GProf", "CAPEI",
]


def _drop_header_rows(ch: pd.DataFrame) -> pd.DataFrame:
    """Keep all daily rows — do NOT filter on SecurityHdrFlg.

    SecurityHdrFlg=='Y' does NOT mark metadata/duplicate rows. In CRSP CIZ it flags the
    rows that fall in a security's CURRENT (most-recent) info segment, and those are REAL
    daily observations (valid price/volume/cap). The previous version dropped them, which
    silently deleted every security's data from its latest info-change onward (e.g. AAPL
    from Oct-2023; all mega-caps gone by 2024-25), collapsing recent-year coverage to
    near zero. True duplicate (PERMNO, date) rows, if any, are removed downstream by
    drop_duplicates(keep='last'). Kept as a named hook in case real metadata rows ever
    need filtering on a different, verified signal.
    """
    return ch


# ---------------------------------------------------------------------------
# Pass 1: annual universe selection
# ---------------------------------------------------------------------------

def pass1_select_universe(crsp_path, config, date_min, date_max):
    """Stream CRSP, build per-(permno, year) summary, select annual members."""
    print("Pass 1/2: scanning CRSP for universe selection...")
    lo, hi = pd.Timestamp(str(date_min)), pd.Timestamp(str(date_max))
    frames, dates = [], set()
    for ch in pd.read_csv(crsp_path, usecols=PASS1_COLS, chunksize=CHUNK, low_memory=False):
        ch = _drop_header_rows(ch)
        ch["date"] = parse_yyyymmdd(ch["YYYYMMDD"])
        ch = ch[(ch["date"] >= lo) & (ch["date"] <= hi)]
        if ch.empty:
            continue
        for c in U.ELIG_FIELDS:
            ch[c] = ch[c].astype("string").str.strip()
        ch["DlyCap"] = pd.to_numeric(ch["DlyCap"], errors="coerce")
        dates.update(ch["date"].unique().tolist())
        frames.append(U.chunk_year_aggregates(ch, config.include_reits))
        print(f"  scanned chunk -> {len(frames)} summaries")

    year_agg = U.accumulate_year_aggregates(frames)
    members = U.select_universe(year_agg, config.universe_size)
    daily_index = pd.DatetimeIndex(sorted(dates))
    year_mask = U.build_year_mask(members, daily_index)
    sizes = {y: len(p) for y, p in sorted(members.items())}
    print(f"  selected universe per year: {sizes}")
    print(f"  union of names across years: {year_mask.shape[1]}; trading days: {len(daily_index)}")
    return members, daily_index, year_mask


# ---------------------------------------------------------------------------
# Pass 2: build price/volume panels for member names
# ---------------------------------------------------------------------------

def pass2_build_panels(crsp_path, member_permnos, daily_index, date_min, date_max):
    """Stream CRSP again, keep only member rows, pivot fields to panels."""
    print("Pass 2/2: extracting member rows and pivoting to panels...")
    member_set = set(int(p) for p in member_permnos)
    lo, hi = pd.Timestamp(str(date_min)), pd.Timestamp(str(date_max))
    frames = []
    for ch in pd.read_csv(crsp_path, usecols=PASS2_COLS, chunksize=CHUNK, low_memory=False):
        ch = _drop_header_rows(ch)
        ch["date"] = parse_yyyymmdd(ch["YYYYMMDD"])
        ch = ch[(ch["date"] >= lo) & (ch["date"] <= hi)]
        ch = ch[ch["PERMNO"].isin(member_set)]
        if ch.empty:
            continue
        frames.append(ch)
    df = pd.concat(frames, ignore_index=True)
    for col in NUMERIC_RAW:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["DlyPrc"] = df["DlyPrc"].abs()  # CRSP: negative = bid/ask midpoint
    df = df.drop_duplicates(["PERMNO", "date"], keep="last")
    print(f"  member rows: {len(df):,} across {df['PERMNO'].nunique()} names")

    panels = {}
    for raw, name in FIELD_MAP.items():
        panels[name] = df.pivot(index="date", columns="PERMNO", values=raw).reindex(daily_index)
    return panels


# ---------------------------------------------------------------------------
# Compustat ratios (forward-filled to daily, masked to membership)
# ---------------------------------------------------------------------------

def prepare_ratio_panels(out_dir, membership):
    """Forward-fill quarterly WRDS ratios to daily, restricted+masked to universe."""
    ratios_path = os.path.join(RAW_DIR, "compustat_ratios.csv")
    if not os.path.exists(ratios_path):
        print("  compustat_ratios.csv not found, skipping ratios.")
        return
    print("Building Compustat ratio panels...")
    df = pd.read_csv(ratios_path, usecols=RATIO_COLS, low_memory=False)
    df["date"] = pd.to_datetime(df["public_date"])  # already release-lagged (PIT)
    df = df.rename(columns={"permno": "PERMNO"}).drop(columns=["public_date"])
    df = df[df["PERMNO"].isin(membership.columns)]
    df = df.drop_duplicates(["PERMNO", "date"], keep="last")

    fields = [c for c in df.columns if c not in ("PERMNO", "date")]
    for field in fields:
        df[field] = pd.to_numeric(df[field], errors="coerce")
        # Union the public_date index with the trading calendar BEFORE ffill so
        # ratios dated on a non-trading day (month-end weekends/holidays) are not
        # dropped — same PIT builder as fundamentals/osap (see ffill_to_membership).
        pivoted = df.pivot(index="date", columns="PERMNO", values=field)
        panel = ffill_to_membership(pivoted, membership)
        panel.to_parquet(os.path.join(out_dir, f"{field.lower()}.parquet"))
    print(f"  saved {len(fields)} ratio panels")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build PIT survivorship-free panels")
    parser.add_argument("--universe-size", type=int, default=None)
    parser.add_argument("--start", type=int, default=DATE_MIN)
    parser.add_argument("--end", type=int, default=DATE_MAX)
    parser.add_argument("--no-reits", action="store_true")
    args = parser.parse_args()

    config = Config()
    if args.universe_size:
        config.universe_size = args.universe_size
    if args.no_reits:
        config.include_reits = False

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(UNIV_DIR, exist_ok=True)
    crsp_path = os.path.join(RAW_DIR, "crsp_daily.csv")

    # 1) Universe + 2) panels
    members, daily_index, year_mask = pass1_select_universe(crsp_path, config, args.start, args.end)
    member_permnos = sorted({p for perms in members.values() for p in perms})
    panels = pass2_build_panels(crsp_path, member_permnos, daily_index, args.start, args.end)

    # 3) Finalize membership (in-universe-year AND trading) and mask everything.
    #    Apply a point-in-time price floor: penny / sub-$5 stock-days are excluded
    #    (their explosive % moves otherwise dominate the equal-weight portfolio).
    membership = U.finalize_membership(year_mask, panels["close"])
    if config.min_price and config.min_price > 0:
        px = panels["close"].reindex(index=membership.index, columns=membership.columns)
        membership = membership & (px >= config.min_price)
    membership.to_parquet(os.path.join(UNIV_DIR, "membership.parquet"))
    print(f"Membership: {membership.shape}, in-universe cells: {int(membership.values.sum()):,}")

    # Raw next-day returns captured BEFORE masking: a position entered on an
    # in-universe day T earns the ACTUAL day-(T+1) return even if the name leaves
    # the universe on T+1 (delisting, or crossing the sub-$5 price floor). Deriving
    # forward returns from the masked returns instead would drop that return based on
    # next-day membership — an optimistic look-ahead / survivorship bias.
    raw_returns = panels["returns"].reindex(index=membership.index, columns=membership.columns)

    for name in list(panels):
        panels[name] = panels[name].reindex(
            index=membership.index, columns=membership.columns
        ).where(membership)
        panels[name].to_parquet(os.path.join(OUT_DIR, f"{name}.parquet"))
    print(f"Saved {len(panels)} CRSP panels: {sorted(panels)}")

    # 4) Forward returns: next-day ACTUAL return, counted whenever the position day T
    #    is in-universe (masked by membership[T] only, not membership[T+1]).
    forward = raw_returns.shift(-1).where(membership)
    forward.to_parquet(os.path.join(OUT_DIR, "forward_returns.parquet"))
    print(f"Saved forward_returns: {forward.shape}")

    # 5) Compustat ratios
    prepare_ratio_panels(OUT_DIR, membership)

    # 6) Invalidate stale value-weighted benchmark cache (recomputed by the pipeline)
    vw = os.path.join(OUT_DIR, "vw_market_return.parquet")
    if os.path.exists(vw):
        os.remove(vw)

    print(f"\nDone. Panels in {OUT_DIR}/  Membership in {UNIV_DIR}/")


if __name__ == "__main__":
    main()
