"""CLI: ingest OSAP firm-level signals into data/panels/.

Download the wide firm-level signals CSV from openassetpricing.com (or via the
`openassetpricing` package) and save it to data/osap_signals.csv, then run:

    python scripts/ingest_osap.py --osap-csv data/osap_signals.csv
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from alpha_gpt.config import Config
from alpha_gpt.data.osap import ingest_osap


def main():
    parser = argparse.ArgumentParser(description="Ingest OSAP signals to panels")
    parser.add_argument("--osap-csv", default="data/osap_signals.csv")
    parser.add_argument("--data-dir", default="data/panels")
    parser.add_argument("--membership", default="data/universe/membership.parquet")
    parser.add_argument("--max-signals", type=int, default=None)
    args = parser.parse_args()
    config = Config()

    if not os.path.exists(args.osap_csv):
        raise SystemExit(
            f"OSAP CSV not found at {args.osap_csv}.\n"
            "Download the wide firm-level signals from openassetpricing.com "
            "(or `pip install openassetpricing`) and save it there."
        )
    if not os.path.exists(args.membership):
        raise SystemExit(f"{args.membership} not found. Run scripts/prepare_data.py first.")

    membership = pd.read_parquet(args.membership)
    ingest_osap(
        args.osap_csv, args.data_dir, membership,
        config.min_coverage, config.min_breadth, args.max_signals,
    )


if __name__ == "__main__":
    main()
