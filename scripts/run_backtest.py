#!/usr/bin/env python3
"""Run historical backtest and print/export report."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from betting.backtest import run_backtest
from betting.config import CONFIG
from betting.reporting import build_report, export_all, print_report


def parse_args():
    p = argparse.ArgumentParser(description="Run win-only backtest")
    p.add_argument("--db", default=None, help="Path to race_reports.sqlite")
    p.add_argument("--output-dir", default="outputs/backtests", help="Output directory")
    p.add_argument("--min-edge", type=float, default=None)
    p.add_argument("--min-price", type=float, default=None)
    p.add_argument("--max-price", type=float, default=None)
    p.add_argument("--min-field-size", type=int, default=None)
    p.add_argument("--max-field-size", type=int, default=None)
    p.add_argument(
        "--staking",
        choices=["flat", "fibonacci"],
        default="flat",
        help="Staking mode for the backtest report",
    )
    return p.parse_args()


def build_runtime_config(args):
    cfg = dict(CONFIG)
    if args.db:
        cfg["database_path"] = args.db
    if args.min_edge is not None:
        cfg["min_edge"] = args.min_edge
    if args.min_price is not None:
        cfg["min_price"] = args.min_price
    if args.max_price is not None:
        cfg["max_price"] = args.max_price
    if args.min_field_size is not None:
        cfg["min_field_size"] = args.min_field_size
    if args.max_field_size is not None:
        cfg["max_field_size"] = args.max_field_size
    cfg["staking_mode"] = args.staking
    return cfg


def main():
    args = parse_args()
    cfg = build_runtime_config(args)
    print("Running backtest...")
    result_df, scored_df = run_backtest(cfg, return_scored=True)
    report = build_report(result_df, cfg, calibration_source_df=scored_df)
    print_report(report)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = export_all(result_df, report, args.output_dir, ts)
    print(f"\nOutputs written:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
