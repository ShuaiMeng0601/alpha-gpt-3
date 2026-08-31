"""Mine Compustat quarterly fundamentals into point-in-time cs_* characteristics.

gvkey->permno mapping uses the (gvkey, permno, public_date) triples already in
compustat_ratios.csv (no WRDS CCM download needed). Point-in-time date = rdq
(earnings announcement) when present, else datadate + 90 days. Characteristics
are forward-filled to daily and masked to universe membership. Also builds a
sector panel (GICS) into data/aux/ for future neutralization (not a terminal).
"""

import os

import numpy as np
import pandas as pd

from alpha_gpt.data.loader import ffill_to_membership, parse_yyyymmdd

FUND_ITEMS = [
    "gvkey", "datadate", "rdq", "gsector",
    "atq", "ltq", "ceqq", "saleq", "revtq", "cogsq", "niq", "ibq", "oiadpq",
    "dlttq", "actq", "lctq", "cheq", "cshoq", "prccq", "xrdq",
]


def build_gvkey_permno_link(ratios_path: str) -> pd.DataFrame:
    """gvkey->permno validity spans from the ratios file."""
    r = pd.read_csv(ratios_path, usecols=["gvkey", "permno", "public_date"], low_memory=False)
    r = r.dropna(subset=["gvkey", "permno"])
    r["public_date"] = pd.to_datetime(r["public_date"], errors="coerce")
    g = (
        r.groupby([r["gvkey"].astype("int64"), r["permno"].astype("int64")])["public_date"]
        .agg(start="min", end="max")
        .reset_index()
    )
    g.columns = ["gvkey", "permno", "start", "end"]
    return g


def load_and_map(fund_path: str, link: pd.DataFrame, universe: set[int]) -> pd.DataFrame:
    """Load fundamentals, set PIT effective date, attach permno by date-overlap."""
    df = pd.read_csv(fund_path, usecols=lambda c: c in FUND_ITEMS, low_memory=False)
    df["gvkey"] = pd.to_numeric(df["gvkey"], errors="coerce")
    df = df.dropna(subset=["gvkey"])
    df["gvkey"] = df["gvkey"].astype("int64")
    df["eff_date"] = parse_yyyymmdd(df["rdq"]).fillna(parse_yyyymmdd(df["datadate"]) + pd.Timedelta(days=90))
    df = df.dropna(subset=["eff_date"])

    m = df.merge(link, on="gvkey", how="inner")
    tol = pd.Timedelta(days=400)
    m = m[(m["eff_date"] >= m["start"] - tol) & (m["eff_date"] <= m["end"] + tol)]
    # Disambiguate many-to-one (a gvkey that re-mapped permnos over time): prefer a
    # permno whose validity span actually CONTAINS eff_date, then in-universe, then the
    # closest span midpoint — so a quarter isn't bound to a permno it matches only via
    # the ±tol slack (which could be a different, already-delisted security).
    m["in_span"] = (m["eff_date"] >= m["start"]) & (m["eff_date"] <= m["end"])
    m["in_univ"] = m["permno"].isin(universe)
    m["mid"] = m["start"] + (m["end"] - m["start"]) / 2
    m["dist"] = (m["eff_date"] - m["mid"]).abs()
    m = m.sort_values(["in_span", "in_univ", "dist"], ascending=[False, False, True])
    m = m.drop_duplicates(["gvkey", "datadate"], keep="first")
    return m


def compute_characteristics(m: pd.DataFrame) -> pd.DataFrame:
    """Derive ~18 standard characteristics. Growth/accruals use 4Q/1Q lags."""
    f = m.sort_values(["permno", "datadate"]).reset_index(drop=True)
    div = lambda a, b: a / b.replace(0, np.nan)
    mktcap = f["prccq"].abs() * f["cshoq"]

    out = pd.DataFrame({"PERMNO": f["permno"].astype("int64"), "date": f["eff_date"]})
    out["cs_gp_at"] = div(f["revtq"] - f["cogsq"], f["atq"])         # gross profitability
    out["cs_roa"] = div(f["niq"], f["atq"])
    out["cs_roe"] = div(f["niq"], f["ceqq"])
    out["cs_opmargin"] = div(f["oiadpq"], f["saleq"])
    out["cs_gpmargin"] = div(f["revtq"] - f["cogsq"], f["revtq"])
    out["cs_leverage"] = div(f["ltq"], f["atq"])
    out["cs_debt_at"] = div(f["dlttq"], f["atq"])
    out["cs_current_ratio"] = div(f["actq"], f["lctq"])
    out["cs_cash_at"] = div(f["cheq"], f["atq"])
    out["cs_bm"] = div(f["ceqq"], mktcap)                            # book/market
    out["cs_ep"] = div(f["ibq"], mktcap)                            # earnings/price
    out["cs_sp"] = div(f["saleq"], mktcap)                          # sales/price
    out["cs_rd_sale"] = div(f["xrdq"], f["saleq"])
    out["cs_asset_growth"] = div(f["atq"], f.groupby("permno")["atq"].shift(4)) - 1
    out["cs_sales_growth"] = div(f["saleq"], f.groupby("permno")["saleq"].shift(4)) - 1
    # Earnings can be negative, so scale the change by |prior| (a one-sided abs on the
    # denominator alone mis-signs loss->profit turnarounds).
    ibq_lag4 = f.groupby("permno")["ibq"].shift(4)
    out["cs_earnings_growth"] = div(f["ibq"] - ibq_lag4, ibq_lag4.abs())
    nwc = (f["actq"] - f["cheq"]) - f["lctq"]
    out["cs_accruals"] = div(nwc - nwc.groupby(f["permno"]).shift(1), f["atq"])
    return out


def _to_daily_panel(long_df: pd.DataFrame, value: str, membership: pd.DataFrame) -> pd.DataFrame:
    """Pivot a [PERMNO, date, value] long frame to a daily masked panel."""
    sub = long_df[["PERMNO", "date", value]].dropna(subset=[value])
    sub = sub[sub["PERMNO"].isin(membership.columns)].drop_duplicates(["PERMNO", "date"], keep="last")
    if sub.empty:
        return None
    panel = sub.pivot_table(index="date", columns="PERMNO", values=value, aggfunc="last")
    return ffill_to_membership(panel, membership)


def fundamentals_to_panels(chars: pd.DataFrame, membership: pd.DataFrame, out_dir: str) -> list[str]:
    saved = []
    for c in [c for c in chars.columns if c.startswith("cs_")]:
        panel = _to_daily_panel(chars.rename(columns={c: "v"}).assign(v=chars[c]), "v", membership)
        if panel is not None:
            panel.to_parquet(os.path.join(out_dir, f"{c}.parquet"))
            saved.append(c)
    return saved


def build_sector_panel(m: pd.DataFrame, membership: pd.DataFrame, aux_dir: str) -> None:
    """GICS sector code as a date x PERMNO panel in aux_dir (NOT a terminal)."""
    sec = m[["permno", "eff_date", "gsector"]].rename(columns={"permno": "PERMNO", "eff_date": "date"})
    sec["gsector"] = pd.to_numeric(sec["gsector"], errors="coerce")
    panel = _to_daily_panel(sec, "gsector", membership)
    if panel is not None:
        os.makedirs(aux_dir, exist_ok=True)
        panel.to_parquet(os.path.join(aux_dir, "sector.parquet"))
