#!/usr/bin/env python3
"""
Predict within-race win probabilities using an XGBoost binary model, then
normalise per race with a softmax over raw margins and compare vs market.

Source rows: runner_rankings (typically created by scripts/rank_all_races.py).

Usage
-----
.venv/bin/python scripts/predict_xgb_value_bets.py
.venv/bin/python scripts/predict_xgb_value_bets.py --min-edge 0.02
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


EXCLUDE_FEATURE_COLS = {
    "race_id",
    "is_winner",
    "finish_place",
    "ranked_at",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict value bets from runner_rankings using XGBoost")
    p.add_argument("--db", default="database/race_reports.sqlite", help="SQLite DB path")
    p.add_argument("--model", default="outputs/xgb_runner_rankings.pkl", help="Model bundle .pkl")
    p.add_argument("--min-edge", type=float, default=0.0, help="Minimum model_prob - market_prob")
    p.add_argument("--max-rows", type=int, default=None, help="Optional cap for display")
    return p.parse_args()


def softmax_within_race(margins: pd.Series) -> pd.Series:
    m = pd.to_numeric(margins, errors="coerce")
    if m.notna().sum() == 0:
        if len(m) == 0:
            return pd.Series(dtype=float, index=margins.index)
        return pd.Series(1.0 / len(m), index=margins.index)
    m = m.fillna(0.0)
    shifted = m - m.max()
    exps = np.exp(shifted)
    total = exps.sum()
    if total <= 0:
        return pd.Series(1.0 / len(m), index=margins.index)
    return pd.Series(exps / total, index=margins.index)


def load_rows(db_path: str) -> pd.DataFrame:
    # "Live" rows: no known outcome. Unresulted rows keep finish_place NULL.
    q = "SELECT * FROM runner_rankings WHERE finish_place IS NULL"
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(q, conn)


def build_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    work = df.copy()
    if "start_time_iso" in work.columns:
        dt = pd.to_datetime(work["start_time_iso"], errors="coerce", utc=True)
        work["start_time_ts"] = dt.map(lambda x: x.timestamp() if pd.notna(x) else np.nan)
        work = work.drop(columns=["start_time_iso"])

    cols = [c for c in work.columns if c not in EXCLUDE_FEATURE_COLS]
    feats = work[cols].copy()

    obj_cols = [c for c in feats.columns if feats[c].dtype == "object"]
    for c in feats.columns:
        if c in obj_cols:
            continue
        feats[c] = pd.to_numeric(feats[c], errors="coerce")

    if obj_cols:
        feats[obj_cols] = feats[obj_cols].fillna("")
        feats = pd.get_dummies(feats, columns=obj_cols, dummy_na=False)

    feats = feats.replace([np.inf, -np.inf], np.nan)

    # Align to training feature columns.
    aligned = feats.reindex(columns=feature_columns, fill_value=0.0)
    return aligned


def compute_fair_market_prob(df: pd.DataFrame) -> pd.Series:
    if "fair_market_prob" in df.columns and df["fair_market_prob"].notna().any():
        return pd.to_numeric(df["fair_market_prob"], errors="coerce").fillna(0.0)

    prices = pd.to_numeric(df.get("live_price"), errors="coerce")
    inv = (1.0 / prices.where(prices > 0)).fillna(0.0)
    return inv.groupby(df["race_id"]).transform(lambda s: s / s.sum() if s.sum() > 0 else 0.0)


def main() -> None:
    args = parse_args()

    db_path = Path(args.db)
    model_path = Path(args.model)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")
    if not model_path.exists():
        raise SystemExit(f"Model bundle not found: {model_path} (train it first)")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    df = load_rows(str(db_path))
    if df.empty:
        print("No live/unresulted rows found in runner_rankings (is_winner IS NULL).")
        return

    X = build_features(df, feature_columns)
    margins = model.predict(X, output_margin=True)
    df = df.copy()
    df["xgb_margin"] = margins

    df["model_prob"] = df.groupby("race_id")["xgb_margin"].transform(softmax_within_race)
    df["market_prob"] = compute_fair_market_prob(df)
    df["edge"] = df["model_prob"] - df["market_prob"]

    picks = df[df["edge"] >= float(args.min_edge)].copy()
    if picks.empty:
        print(f"No candidates with edge >= {args.min_edge:.4f}.")
        return

    picks = picks.sort_values(["start_time_iso", "race_id", "edge"], ascending=[True, True, False])

    show_cols = [
        "start_time_iso",
        "competition_name",
        "race_number",
        "race_id",
        "runner_number",
        "runner_name",
        "live_price",
        "model_prob",
        "market_prob",
        "edge",
        "model_score",
        "model_rank",
    ]
    show_cols = [c for c in show_cols if c in picks.columns]
    out = picks[show_cols].copy()

    for c in ["model_prob", "market_prob", "edge"]:
        if c in out.columns:
            out[c] = out[c].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "")

    if args.max_rows is not None:
        out = out.head(int(args.max_rows))

    try:
        print(out.to_string(index=False))
    except Exception:
        print(out)

    print(f"\nCandidates: {len(picks)} (edge >= {args.min_edge:.4f})")


if __name__ == "__main__":
    main()
