"""Point-in-time, survivorship-free universe construction for CRSP CIZ data.

The universe is rebalanced annually: at the start of each calendar year we take
the names that are *eligible* (US common stock on a major exchange) and pick the
top-N by TRAILING (prior-year) market cap. A name stays in the panel from its
entry until it delists; nothing is selected using full-sample information, so the
panel is free of the survivorship/look-ahead bias in the original static sample.

Memory note: designed for a two-pass stream over the ~26GB CRSP CSV. Pass 1 feeds
chunks to `chunk_year_aggregates` (tiny per-(permno, year) summary); after
selection, Pass 2 keeps only member rows. Nothing here holds the full daily panel.

The CIZ code sets below are confirmed against the actual value distributions in
data/crsp_daily.csv (see scripts probe). Adjust here if the extract differs.
"""

from __future__ import annotations

import pandas as pd

# --- CIZ eligibility code sets (CONFIRM against the probe value_counts) ---
COMMON_SHARE_TYPES = {"NS"}          # normal/common shares (NS=44.6M; AD/SB/UG/CE excluded)
EQUITY_SECURITY_TYPES = {"EQTY"}     # equity (excludes FUND=ETF/CEF)
CORP_ISSUER_TYPES = {"CORP"}         # operating corporations (ACOR=fund-associated excluded)
REIT_ISSUER_TYPES = {"REIT"}         # added when include_reits=True
MAJOR_EXCHANGES = {"N", "A", "Q"}    # NYSE, NYSE American (AMEX), NASDAQ (R/B/X excluded)
US_INCORPORATED = {"Y"}              # USIncFlg; matches classic CRSP shrcd 10/11 common

ELIG_FIELDS = ["ShareType", "SecurityType", "IssuerType", "PrimaryExch", "USIncFlg"]
HEADER_FLAG = "SecurityHdrFlg"       # CIZ current-segment flag; 'Y' rows are REAL daily obs — do NOT drop


def row_eligible(df: pd.DataFrame, include_reits: bool = True) -> pd.Series:
    """Boolean row mask: US common stock on a major exchange (CIZ codes)."""
    issuers = set(CORP_ISSUER_TYPES) | (set(REIT_ISSUER_TYPES) if include_reits else set())
    return (
        df["ShareType"].isin(COMMON_SHARE_TYPES)
        & df["SecurityType"].isin(EQUITY_SECURITY_TYPES)
        & df["IssuerType"].isin(issuers)
        & df["PrimaryExch"].isin(MAJOR_EXCHANGES)
        & df["USIncFlg"].isin(US_INCORPORATED)
    )


def chunk_year_aggregates(chunk: pd.DataFrame, include_reits: bool = True) -> pd.DataFrame:
    """Reduce a raw CRSP chunk to a per-(PERMNO, year) summary frame.

    Expects columns: PERMNO, date (datetime), DlyCap, ELIG_FIELDS.
    Returns a frame indexed by (PERMNO, year) with cap_sum, cap_cnt, elig_any.
    Header rows must already be dropped by the caller.
    """
    c = chunk[["PERMNO", "date", "DlyCap"]].copy()
    c["year"] = chunk["date"].dt.year
    c["elig"] = row_eligible(chunk, include_reits)
    return c.groupby(["PERMNO", "year"]).agg(
        cap_sum=("DlyCap", "sum"),
        cap_cnt=("DlyCap", "count"),
        elig_any=("elig", "max"),
    )


def accumulate_year_aggregates(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine per-chunk summaries into one per-(PERMNO, year) summary."""
    allg = pd.concat(frames)
    return allg.groupby(level=[0, 1]).agg(
        {"cap_sum": "sum", "cap_cnt": "sum", "elig_any": "max"}
    )


def select_universe(year_agg: pd.DataFrame, n: int) -> dict[int, list[int]]:
    """Pick top-N eligible PERMNOs per year using TRAILING (prior-year) cap.

    year_agg: index (PERMNO, year), columns cap_sum/cap_cnt/elig_any.
    Returns {year: [permno, ...]}. For the earliest year (no prior year) the
    current year's own aggregates are used (documented first-year exception).
    """
    mean_cap = (year_agg["cap_sum"] / year_agg["cap_cnt"]).rename("mean_cap")
    elig = year_agg["elig_any"].astype(bool)
    years = sorted({int(y) for _, y in year_agg.index})
    members: dict[int, list[int]] = {}
    for y in years:
        src = y - 1 if (y - 1) in years else y  # trailing; fall back for first year
        try:
            cap_src = mean_cap.xs(src, level=1)
            elig_src = elig.xs(src, level=1)
        except KeyError:
            continue
        cand = cap_src[elig_src.reindex(cap_src.index).fillna(False)].dropna()
        if not cand.empty:
            members[y] = [int(p) for p in cand.nlargest(n).index]
    return members


def build_year_mask(
    members_by_year: dict[int, list[int]], daily_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Boolean date x PERMNO frame: True where a name is a member that year.

    This is the calendar-year membership; intersect with `close.notna()`
    downstream so a name is dropped at delisting (no values past its last trade).
    """
    all_permnos = sorted({p for perms in members_by_year.values() for p in perms})
    mask = pd.DataFrame(False, index=daily_index, columns=all_permnos)
    years = daily_index.year
    for y, perms in members_by_year.items():
        rows = years == y
        cols = [p for p in perms if p in mask.columns]
        if cols:
            mask.loc[rows, cols] = True
    return mask


def finalize_membership(year_mask: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """Effective membership = in-universe-year AND actually trading that day.

    Drops a name at delisting and removes trading days outside its membership
    years. `close` and `year_mask` are aligned to a common date x PERMNO grid.
    """
    cols = year_mask.columns.union(close.columns)
    ym = year_mask.reindex(index=close.index, columns=cols, fill_value=False)
    cl = close.reindex(index=close.index, columns=cols)
    return ym & cl.notna()
