#!/usr/bin/env python3
"""
Fit an isotonic-regression calibration model from historical backtest data.

Runs the scoring pipeline (features → scoring → probabilities) on all resulted
races WITHOUT applying filters, then fits a monotone mapping from
raw_model_prob → actual win rate using sklearn IsotonicRegression.

The saved model is loaded automatically by betting/calibration.py for both
live candidate selection and backtest runs.

Usage
-----
# Fit and save (default output: betting/calibration_model.pkl)
python3 scripts/fit_calibration.py

# Custom database or output path
python3 scripts/fit_calibration.py --db database/race_reports.sqlite --output betting/calibration_model.pkl

# Dry-run: print stats but do not save
python3 scripts/fit_calibration.py --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from betting.backtest import run_pipeline
from betting.config import CONFIG
from betting.db import load_draw_bias_table, load_jockey_stats_table, load_race_runners, load_trainer_stats_table
from betting.features import build_features
from betting.meta_model import add_meta_model_signal
from betting.probabilities import assign_probabilities
from betting.scoring import score_runners
from betting.validation import validate_input


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit probability calibration model")
    p.add_argument("--db", default=None, help="Path to SQLite database")
    p.add_argument(
        "--output",
        default=None,
        help="Where to save the fitted model (.pkl). Default: betting/calibration_model.pkl",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print calibration stats but do not save the model",
    )
    return p.parse_args()


def score_all_backtest(config: dict) -> pd.DataFrame:
    """Run pipeline up to assign_probabilities on all backtest rows (no filters)."""
    df = load_race_runners(config["database_path"], "backtest", config)
    with sqlite3.connect(config["database_path"]) as conn:
        draw_bias_df = load_draw_bias_table(conn)
        jockey_stats_df = load_jockey_stats_table(conn)
        trainer_stats_df = load_trainer_stats_table(conn)
    df = validate_input(df, config)
    df = build_features(df, config, draw_bias_df, jockey_stats_df, trainer_stats_df)
    df = add_meta_model_signal(df, config, draw_bias_df, jockey_stats_df, trainer_stats_df)
    df = score_runners(df, config)
    df = assign_probabilities(df, config)
    return df


def fit_isotonic(df: pd.DataFrame) -> IsotonicRegression:
    """Fit isotonic regression mapping raw_model_prob → is_winner."""
    clean = df[df["is_winner"].notna() & df["raw_model_prob"].notna()].copy()
    if len(clean) < 100:
        raise ValueError(
            f"Only {len(clean)} rows with known outcomes — too few to calibrate reliably."
        )
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(clean["raw_model_prob"].values, clean["is_winner"].astype(float).values)
    return ir


def calibration_table(df: pd.DataFrame, ir: IsotonicRegression) -> pd.DataFrame:
    """Print before/after calibration comparison by probability bucket."""
    clean = df[df["is_winner"].notna() & df["raw_model_prob"].notna()].copy()
    clean["calibrated_prob"] = ir.predict(clean["raw_model_prob"].values)
    clean["prob_bucket"] = pd.cut(
        clean["raw_model_prob"],
        bins=[0, 0.05, 0.07, 0.10, 0.15, 0.25, 1.0],
        labels=["0-5%", "5-7%", "7-10%", "10-15%", "15-25%", "25%+"],
    )
    table = (
        clean.groupby("prob_bucket", observed=False)
        .agg(
            runners=("is_winner", "count"),
            wins=("is_winner", "sum"),
            raw_model_prob=("raw_model_prob", "mean"),
            calibrated_prob=("calibrated_prob", "mean"),
            avg_price=("live_price", "mean"),
        )
        .assign(
            actual_win_pct=lambda t: (t["wins"] / t["runners"] * 100).round(1),
            raw_prob_pct=lambda t: (t["raw_model_prob"] * 100).round(1),
            cal_prob_pct=lambda t: (t["calibrated_prob"] * 100).round(1),
            raw_ratio=lambda t: (
                t["wins"] / t["runners"] / t["raw_model_prob"].replace(0, np.nan)
            ).round(2),
            cal_ratio=lambda t: (
                t["wins"] / t["runners"] / t["calibrated_prob"].replace(0, np.nan)
            ).round(2),
        )
    )
    return table[
        [
            "runners",
            "wins",
            "actual_win_pct",
            "raw_prob_pct",
            "cal_prob_pct",
            "raw_ratio",
            "cal_ratio",
            "avg_price",
        ]
    ]


def main() -> None:
    args = parse_args()

    config = dict(CONFIG)
    if args.db:
        config["database_path"] = args.db

    default_output = Path(__file__).resolve().parent.parent / "betting" / "calibration_model.pkl"
    output_path = Path(args.output) if args.output else default_output

    print("Loading and scoring all backtest rows (no filters)…")
    df = score_all_backtest(config)
    total = len(df)
    with_result = df["is_winner"].notna().sum()
    print(f"  Total rows: {total:,}  |  With known outcome: {with_result:,}")

    print("\nFitting isotonic regression…")
    ir = fit_isotonic(df)
    print("  Done.")

    print("\nCALIBRATION TABLE  (ratio ≈ 1.00 means perfectly calibrated)")
    print("-" * 75)
    table = calibration_table(df, ir)
    print(table.to_string())
    print("-" * 75)

    if args.dry_run:
        print("\n[dry-run] Model NOT saved.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ir, output_path)
    print(f"\nCalibration model saved → {output_path}")
    print("It will be loaded automatically on the next backtest or live run.")


if __name__ == "__main__":
    main()
