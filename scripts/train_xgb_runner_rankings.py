#!/usr/bin/env python3
"""
Train an XGBoost binary classifier from runner_rankings.

Model: objective='binary:logistic' (runner-level), then at prediction-time we
normalise per-race with a softmax over raw margins to get within-field win
probabilities.

Usage
-----
.venv/bin/python scripts/train_xgb_runner_rankings.py
.venv/bin/python scripts/train_xgb_runner_rankings.py --model-out outputs/xgb_model.pkl
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss


EXCLUDE_FEATURE_COLS = {
    # identifiers / grouping
    "race_id",
    # leaky labels / outcomes
    "is_winner",
    "finish_place",
    # non-features / bookkeeping
    "ranked_at",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost model from runner_rankings")
    p.add_argument("--db", default="database/race_reports.sqlite", help="SQLite DB path")
    p.add_argument("--model-out", default="outputs/xgb_runner_rankings.pkl", help="Output .pkl path")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--valid-race-frac", type=float, default=0.2)
    p.add_argument("--n-estimators", type=int, default=800)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument(
        "--drop-object-cols",
        action="store_true",
        help="Drop object/text columns instead of one-hot encoding them (reduces memorization via names)",
    )
    return p.parse_args()


def load_runner_rankings(db_path: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            # Use only resulted rows for training. Unresulted rows typically have
            # finish_place NULL but is_winner=0, which would be label noise.
            "SELECT * FROM runner_rankings WHERE finish_place IS NOT NULL",
            conn,
        )
    return df


@dataclass(frozen=True)
class DesignMatrix:
    X: pd.DataFrame
    y: pd.Series
    race_id: pd.Series
    feature_columns: list[str]


def build_design_matrix(df: pd.DataFrame, drop_object_cols: bool) -> DesignMatrix:
    if df.empty:
        raise ValueError("No training rows found in runner_rankings (is_winner IS NOT NULL).")
    if "race_id" not in df.columns or "is_winner" not in df.columns:
        raise ValueError("runner_rankings must include race_id and is_winner.")

    work = df.copy()

    # Convert start_time_iso (if present) into a numeric timestamp feature.
    if "start_time_iso" in work.columns:
        dt = pd.to_datetime(work["start_time_iso"], errors="coerce", utc=True)
        # pandas 3.x: avoid Series.view; use datetime accessor.
        work["start_time_ts"] = dt.map(lambda x: x.timestamp() if pd.notna(x) else np.nan)
        work = work.drop(columns=["start_time_iso"])

    y = pd.to_numeric(work["is_winner"], errors="coerce").fillna(0).astype(int)
    race_id = work["race_id"]

    feature_cols = [c for c in work.columns if c not in EXCLUDE_FEATURE_COLS]
    feats = work[feature_cols].copy()

    # Cast likely-numeric columns; objects become categoricals via one-hot.
    obj_cols = [c for c in feats.columns if feats[c].dtype == "object"]
    for c in feats.columns:
        if c in obj_cols:
            continue
        feats[c] = pd.to_numeric(feats[c], errors="coerce")

    if obj_cols and drop_object_cols:
        feats = feats.drop(columns=obj_cols)
        obj_cols = []

    if obj_cols:
        feats[obj_cols] = feats[obj_cols].fillna("")
        feats = pd.get_dummies(feats, columns=obj_cols, dummy_na=False)

    # XGBoost doesn't like inf; keep NaN (handled internally) and clamp inf.
    feats = feats.replace([np.inf, -np.inf], np.nan)

    return DesignMatrix(X=feats, y=y, race_id=race_id, feature_columns=list(feats.columns))


def split_by_race(dm: DesignMatrix, valid_race_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    races = dm.race_id.dropna().unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(races)
    n_valid = int(len(races) * valid_race_frac)
    valid_races = set(races[:n_valid])
    is_valid = dm.race_id.isin(valid_races).values
    train_idx = np.where(~is_valid)[0]
    valid_idx = np.where(is_valid)[0]
    return train_idx, valid_idx


def train_model(dm: DesignMatrix, train_idx: np.ndarray, valid_idx: np.ndarray, args: argparse.Namespace):
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        tree_method="hist",
        random_state=args.seed,
        n_jobs=-1,
    )

    X_train = dm.X.iloc[train_idx]
    y_train = dm.y.iloc[train_idx]
    X_valid = dm.X.iloc[valid_idx]
    y_valid = dm.y.iloc[valid_idx]

    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    # Basic runner-level metrics (race-level normalisation happens at predict time).
    p_valid = model.predict_proba(X_valid)[:, 1]
    ll = log_loss(y_valid, p_valid, labels=[0, 1])

    return model, ll, int(len(train_idx)), int(len(valid_idx))


def main() -> None:
    args = parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    print(f"Loading training data from {db_path} ...")
    df = load_runner_rankings(str(db_path))
    print(f"  rows: {len(df):,}  races: {df['race_id'].nunique():,}")

    dm = build_design_matrix(df, drop_object_cols=bool(args.drop_object_cols))
    train_idx, valid_idx = split_by_race(dm, args.valid_race_frac, args.seed)
    print(f"Split by race_id: train rows={len(train_idx):,} valid rows={len(valid_idx):,}")

    model, valid_logloss, n_train, n_valid = train_model(dm, train_idx, valid_idx, args)
    print(f"Valid logloss (runner-level, before per-race softmax): {valid_logloss:.5f}")
    print(f"Features: {len(dm.feature_columns):,}")

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "feature_columns": dm.feature_columns,
        "excluded_feature_cols": sorted(EXCLUDE_FEATURE_COLS),
        "start_time_feature": "start_time_ts",
        "drop_object_cols": bool(args.drop_object_cols),
        "train_rows": n_train,
        "valid_rows": n_valid,
        "valid_runner_logloss": float(valid_logloss),
    }
    joblib.dump(bundle, out_path)
    print(f"Saved model bundle -> {out_path}")


if __name__ == "__main__":
    main()
