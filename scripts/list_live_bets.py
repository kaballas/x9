#!/usr/bin/env python3
"""List current live bet candidates from unresolved races."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from betting.config import CONFIG
from betting.live_candidates import run_live_candidates
from betting.reporting import export_csv


DISPLAY_COLS = [
    "start_time_iso",
    "competition_name",
    "race_number",
    "runner_name",
    "live_price",
    "price_quality",
    "open_price",
    "fluc1",
    "fluc2",
    "model_score",
    "model_rank",
    "market_rank",
    "model_prob",
    "raw_market_prob",
    "fair_market_prob",
    "market_implied_prob",
    "raw_edge",
    "fair_edge",
    "edge",
    "ev",
    "stake",
]


def parse_args():
    p = argparse.ArgumentParser(description="List live bet candidates")
    p.add_argument("--db", default=None)
    p.add_argument("--output", default=None, help="Optional CSV output path")
    p.add_argument("--min-edge", type=float, default=None)
    return p.parse_args()


def build_runtime_config(args):
    cfg = dict(CONFIG)
    if args.db:
        cfg["database_path"] = args.db
    if args.min_edge is not None:
        cfg["min_edge"] = args.min_edge
    return cfg


def main():
    args = parse_args()
    cfg = build_runtime_config(args)
    candidates = run_live_candidates(cfg)
    if candidates.empty:
        print("No live candidates found matching current filters.")
        return

    show = [c for c in DISPLAY_COLS if c in candidates.columns]
    display = candidates[show].copy()
    for pct_col in ["raw_market_prob", "fair_market_prob", "market_implied_prob", "raw_edge", "fair_edge", "edge", "ev"]:
        if pct_col in display.columns:
            display[pct_col] = display[pct_col].map(lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "")

    try:
        print(display.to_string(index=False))
    except Exception:
        print(display)

    print(f"\nTotal candidates: {len(candidates)}")
    if args.output:
        export_csv(candidates, args.output)
        print(f"Exported to: {args.output}")


if __name__ == "__main__":
    main()
