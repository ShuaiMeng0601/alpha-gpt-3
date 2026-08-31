"""Download the three raw WRDS extracts this repo builds its panels from, via the WRDS
Python API (``pip install wrds``) — no manual web-query CSV exports needed.

Produces (column-compatible with scripts/prepare_data.py + scripts/ingest_fundamentals.py):

  data/crsp_daily.csv             CRSP Daily Stock File v2 (CIZ), all stocks, one row per
                                  (PERMNO, day): prices/volume/cap + share-eligibility flags.
  data/compustat_fundamentals.csv Compustat quarterly (comp.fundq) + GICS sector, the raw
                                  items behind the cs_* characteristic panels.
  data/compustat_ratios.csv       WRDS Financial Ratios Suite (point-in-time: public_date
                                  is the release-lagged availability date).

Usage:
    python scripts/download_wrds.py                     # everything, 2005..today
    python scripts/download_wrds.py --only crsp         # one extract
    python scripts/download_wrds.py --username you      # else wrds prompts / ~/.pgpass

Then rebuild panels exactly as before:
    python scripts/prepare_data.py --universe-size 1500
    python scripts/ingest_fundamentals.py

Notes:
  * Credentials: the wrds package prompts on first connect and can store a ~/.pgpass.
  * CRSP daily is the big one (~35M rows); it streams year-by-year into the CSV so memory
    stays bounded. Expect tens of minutes depending on connection. Its start is separate
    (--crsp-start, default 2010-01-01 = prepare_data's DATE_MIN) because prepare_data
    discards earlier daily rows anyway; fundamentals start at --start (default 2005: 5y
    of headroom so 4-quarter lags in growth/accruals have history at the 2010 panel start).
  * If the download extends past 2025-12-31, pass a matching --end to prepare_data.py
    (its DATE_MAX default currently caps panels at 2025-12-31).
  * divyield compat: the old manual web export stored divyield as percent STRINGS
    ("1.92%"), which prepare_data's to_numeric coerced to all-NaN — the current
    divyield panel is empty. This download writes numeric fractions (0.0192), so the
    divyield terminal becomes populated for the first time; results touching it will
    (legitimately) change.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# --- CRSP daily (CIZ / v2). crsp.wrds_dsfv2_query is WRDS's flat view backing the web
# "Daily Stock File v2" query: dsf_v2 daily rows merged with the point-in-time security
# info segment (share type / exchange / incorporation flags) valid on each day.
CRSP_VIEW = "crsp.wrds_dsfv2_query"
# postgres column -> CSV column expected by prepare_data.py (CIZ CamelCase)
CRSP_RENAME = {
    "permno": "PERMNO", "dlycaldt": "YYYYMMDD",
    "sharetype": "ShareType", "securitytype": "SecurityType", "issuertype": "IssuerType",
    "primaryexch": "PrimaryExch", "usincflg": "USIncFlg", "securityhdrflg": "SecurityHdrFlg",
    "dlyclose": "DlyClose", "dlyopen": "DlyOpen", "dlyhigh": "DlyHigh", "dlylow": "DlyLow",
    "dlyvol": "DlyVol", "dlyret": "DlyRet", "dlyprc": "DlyPrc", "shrout": "ShrOut",
    "dlycap": "DlyCap", "dlybid": "DlyBid", "dlyask": "DlyAsk",
    "dlynumtrd": "DlyNumTrd", "dlyprcvol": "DlyPrcVol",
}

# --- Compustat quarterly: the FUND_ITEMS alpha_gpt.data.fundamentals reads. gsector lives
# on the company header (comp.company), not fundq.
FUNDQ_ITEMS = ["gvkey", "datadate", "rdq", "atq", "ltq", "ceqq", "saleq", "revtq", "cogsq",
               "niq", "ibq", "oiadpq", "dlttq", "actq", "lctq", "cheq", "cshoq", "prccq", "xrdq"]

# --- WRDS Financial Ratios Suite (firm level). public_date = release-lagged PIT date.
RATIO_TABLE = "wrdsapps_finratio.firm_ratio"
RATIO_ITEMS = ["gvkey", "permno", "adate", "qdate", "public_date",
               "capei", "bm", "pe_op_dil", "pe_exi", "ps", "pcf", "npm", "opmbd", "gpm",
               "roa", "roe", "roce", "gprof", "debt_at", "de_ratio", "curr_ratio",
               "quick_ratio", "cash_ratio", "at_turn", "inv_turn", "rect_turn",
               "accrual", "ptb", "divyield"]
# prepare_data.py reads these two by exact (non-lowercase) name
RATIO_RENAME = {"gprof": "GProf", "capei": "CAPEI"}


def _connect(username: str | None):
    try:
        import wrds
    except ImportError:
        raise SystemExit("The 'wrds' package is required: ./venv/bin/pip install wrds")
    kwargs = {"wrds_username": username} if username else {}
    return wrds.Connection(**kwargs)


def _available_columns(db, view: str) -> set[str]:
    schema, table = view.split(".", 1)
    desc = db.describe_table(library=schema, table=table)
    return set(desc["name"].astype(str).str.lower())


# Columns the universe/panel build cannot proceed without: identity, date, the return/
# price/cap fields, and every eligibility flag (a missing flag would make row_eligible()
# all-False -> an EMPTY universe discovered only after a multi-GB download).
CRSP_REQUIRED = {"permno", "dlycaldt", "dlyret", "dlyclose", "dlycap", "shrout",
                 "sharetype", "securitytype", "issuertype", "primaryexch", "usincflg"}


def download_crsp(db, out_path: str, start: str, end: str) -> None:
    cols = _available_columns(db, CRSP_VIEW)
    need = CRSP_REQUIRED - cols
    if need:
        raise SystemExit(f"{CRSP_VIEW} lacks required columns {sorted(need)}; "
                         "aborting before a multi-GB download of unusable data")
    want = [c for c in CRSP_RENAME if c in cols]
    missing = sorted(set(CRSP_RENAME) - cols)
    if missing:
        print(f"  note: {CRSP_VIEW} lacks optional {missing}; those CSV columns will be empty")
    years = range(int(start[:4]), int(end[:4]) + 1)
    first = True
    total, max_date = 0, None
    for y in years:
        lo, hi = max(f"{y}-01-01", start), min(f"{y}-12-31", end)
        if lo > hi:
            continue
        q = (f"select {', '.join(want)} from {CRSP_VIEW} "
             f"where dlycaldt between '{lo}' and '{hi}'")
        df = db.raw_sql(q, date_cols=["dlycaldt"])
        for c in set(CRSP_RENAME) - set(df.columns):
            df[c] = pd.NA  # keep the CSV contract for the optional columns
        if len(df):
            # postgres returns SAS numerics as float; pin the integer PERMNO contract so
            # rebuilt panels keep int64 column labels (matching all existing artifacts)
            df["permno"] = df["permno"].astype("int64")
            max_date = max(max_date or df["dlycaldt"].max(), df["dlycaldt"].max())
        df = df.rename(columns=CRSP_RENAME)[list(CRSP_RENAME.values())]
        df.to_csv(out_path, index=False, mode="w" if first else "a", header=first)
        first = False
        total += len(df)
        print(f"  crsp {y}: {len(df):,} rows (total {total:,})", flush=True)
    print(f"  wrote {out_path} ({total:,} rows, through {max_date})")
    if max_date is not None and str(max_date)[:10] > "2025-12-31":
        print("  NOTE: raw CRSP now extends past 2025-12-31 — run prepare_data.py with a "
              f"matching --end (e.g. {str(max_date)[:10].replace('-', '')}) or bump its DATE_MAX")


def download_fundq(db, out_path: str, start: str) -> None:
    q = (f"select f.{', f.'.join(FUNDQ_ITEMS)}, c.gsector "
         f"from comp.fundq f left join comp.company c on f.gvkey = c.gvkey "
         f"where f.indfmt = 'INDL' and f.datafmt = 'STD' and f.popsrc = 'D' "
         f"and f.consol = 'C' and f.datadate >= '{start}'")
    df = db.raw_sql(q, date_cols=["datadate", "rdq"])
    df.to_csv(out_path, index=False)
    print(f"  wrote {out_path} ({len(df):,} rows, {df['gvkey'].nunique():,} gvkeys)")


def download_ratios(db, out_path: str, start: str) -> None:
    cols = _available_columns(db, RATIO_TABLE)
    need = {"gvkey", "permno", "public_date"} - cols
    if need:
        raise SystemExit(f"{RATIO_TABLE} lacks required columns {sorted(need)}; "
                         "the gvkey->permno link cannot be built without them")
    want = [c for c in RATIO_ITEMS if c in cols]
    missing = sorted(set(RATIO_ITEMS) - cols)
    if missing:
        # Backfilled as empty below: prepare_data reads with a strict usecols list, so a
        # DROPPED column would crash it — an empty column just yields a NaN panel.
        print(f"  note: {RATIO_TABLE} lacks {missing}; writing them as empty columns")
    q = (f"select {', '.join(want)} from {RATIO_TABLE} "
         f"where public_date >= '{start}' and permno is not null")
    df = db.raw_sql(q, date_cols=["adate", "qdate", "public_date"])
    df["permno"] = df["permno"].astype("int64")  # pin the integer PERMNO contract
    for c in set(RATIO_ITEMS) - set(df.columns):
        df[c] = pd.NA
    df = df.rename(columns=RATIO_RENAME)[[RATIO_RENAME.get(c, c) for c in RATIO_ITEMS]]
    df.to_csv(out_path, index=False)
    print(f"  wrote {out_path} ({len(df):,} rows, {df['permno'].nunique():,} permnos)")


def main():
    p = argparse.ArgumentParser(description="Download raw WRDS extracts for this repo")
    p.add_argument("--username", default=None, help="WRDS username (else prompt/~/.pgpass)")
    p.add_argument("--start", default="2005-01-01",
                   help="earliest FUNDAMENTALS date (default 2005: 5y of 4Q-lag headroom "
                        "before the 2010 panel start)")
    p.add_argument("--crsp-start", default="2010-01-01",
                   help="earliest CRSP daily date (default 2010 = prepare_data's DATE_MIN; "
                        "earlier daily rows would be downloaded only to be discarded)")
    p.add_argument("--end", default=str(pd.Timestamp.today().date()))
    p.add_argument("--only", choices=["crsp", "fundq", "ratios"], default=None,
                   help="download a single extract instead of all three")
    args = p.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    db = _connect(args.username)
    try:
        if args.only in (None, "crsp"):
            print("CRSP daily (v2/CIZ)...")
            download_crsp(db, os.path.join(RAW_DIR, "crsp_daily.csv"), args.crsp_start, args.end)
        if args.only in (None, "fundq"):
            print("Compustat quarterly fundamentals...")
            download_fundq(db, os.path.join(RAW_DIR, "compustat_fundamentals.csv"), args.start)
        if args.only in (None, "ratios"):
            print("WRDS financial ratios (point-in-time)...")
            download_ratios(db, os.path.join(RAW_DIR, "compustat_ratios.csv"), args.start)
    finally:
        db.close()
    print("\nDone. Next: python scripts/prepare_data.py && python scripts/ingest_fundamentals.py")


if __name__ == "__main__":
    main()
