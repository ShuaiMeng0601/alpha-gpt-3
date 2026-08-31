"""CLI: mine Compustat fundamentals into cs_* characteristic panels + sector aux.

Uses data already on disk (data/compustat_fundamentals.csv + compustat_ratios.csv)
and data/universe/membership.parquet. Run after scripts/prepare_data.py:

    python scripts/ingest_fundamentals.py
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from alpha_gpt.config import Config
from alpha_gpt.data.fundamentals import (
    build_gvkey_permno_link,
    build_sector_panel,
    compute_characteristics,
    fundamentals_to_panels,
    load_and_map,
)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    parser = argparse.ArgumentParser(description="Mine Compustat fundamentals -> cs_* panels")
    parser.add_argument("--data-dir", default="data/panels")
    parser.add_argument("--aux-dir", default=None)
    parser.add_argument("--membership", default="data/universe/membership.parquet")
    args = parser.parse_args()
    config = Config()
    aux_dir = args.aux_dir or config.aux_dir

    fund = os.path.join(RAW_DIR, "compustat_fundamentals.csv")
    ratios = os.path.join(RAW_DIR, "compustat_ratios.csv")
    if not os.path.exists(fund):
        raise SystemExit(f"{fund} not found.")
    if not os.path.exists(args.membership):
        raise SystemExit(f"{args.membership} not found. Run scripts/prepare_data.py first.")

    membership = pd.read_parquet(args.membership)
    universe = set(int(x) for x in membership.columns)

    link = build_gvkey_permno_link(ratios)
    print(f"gvkey->permno link pairs: {len(link):,}")
    m = load_and_map(fund, link, universe)
    print(f"mapped fundamentals rows: {len(m):,} ({m['permno'].nunique()} permnos)")

    chars = compute_characteristics(m)
    saved = fundamentals_to_panels(chars, membership, args.data_dir)
    print(f"saved {len(saved)} cs_ panels: {saved}")

    build_sector_panel(m, membership, aux_dir)
    print(f"saved sector aux panel -> {aux_dir}/sector.parquet")


if __name__ == "__main__":
    main()
