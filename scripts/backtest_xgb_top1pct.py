#!/usr/bin/env python3
"""
Backtest XGBoost per-race softmax probabilities vs market and report the top 1%.

This uses the runner_rankings table (historical/resulted rows), predicts raw
margins from the trained binary:logistic model, softmax-normalises within each
race_id to get a true within-field win probability, then compares vs market.

"Top 1%" here means: take all eligible resulted runners, sort descending by
edge = model_prob - market_prob, then keep the top 1% rows.

Usage
-----
.venv/bin/python scripts/backtest_xgb_top1pct.py --model outputs/xgb_runner_rankings.pkl
.venv/bin/python scripts/backtest_xgb_top1pct.py --top-frac 0.01 --min-edge 0.00
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


EXCLUDE_FEATURE_COLS = {
    "race_id",
    "is_winner",
    "finish_place",
    "ranked_at",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest XGB per-race softmax top 1% edges")
    p.add_argument("--db", default="database/race_reports.sqlite", help="SQLite DB path")
    p.add_argument("--model", default="outputs/xgb_runner_rankings.pkl", help="Model bundle .pkl")
    p.add_argument("--top-frac", type=float, default=0.01, help="Fraction to keep (default 0.01 = top 1%)")
    p.add_argument("--min-edge", type=float, default=0.0, help="Optional lower bound edge filter before ranking")
    p.add_argument(
        "--softmax-temperature",
        type=float,
        default=2.0,
        help="Softmax temperature applied to margins within each race (higher = flatter)",
    )
    p.add_argument(
        "--fit-temperature",
        action="store_true",
        help="Fit a single softmax temperature on a held-out race split (grid-search) and use it",
    )
    p.add_argument("--seed", type=int, default=13)
    p.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Time-based split: fraction of races (by start_time_iso) used for training (default 0.8)",
    )
    p.add_argument(
        "--train-in-script",
        action="store_true",
        help="Train a fresh model inside this script on the training split (ignores --model model weights but reuses its feature_columns)",
    )
    p.add_argument(
        "--drop-object-cols",
        action="store_true",
        help="Drop object/text columns instead of one-hot encoding them (reduces memorization via names)",
    )
    return p.parse_args()


def softmax_within_race(margins: pd.Series, temperature: float) -> pd.Series:
    m = pd.to_numeric(margins, errors="coerce")
    if m.notna().sum() == 0:
        if len(m) == 0:
            return pd.Series(dtype=float, index=margins.index)
        return pd.Series(1.0 / len(m), index=margins.index)
    m = m.fillna(0.0)
    shifted = m - m.max()
    t = float(temperature)
    if t <= 0:
        t = 1.0
    exps = np.exp(shifted / t)
    total = exps.sum()
    if total <= 0:
        return pd.Series(1.0 / len(m), index=margins.index)
    return pd.Series(exps / total, index=margins.index)


def load_resulted_rows(db_path: str) -> pd.DataFrame:
    q = """
    SELECT *
    FROM runner_rankings
    WHERE finish_place IS NOT NULL
      AND is_winner IS NOT NULL
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(q, conn)


def choose_market_price(df: pd.DataFrame) -> pd.Series:
    # Prefer SP for backtest consistency; fall back to live_price if missing.
    if "sp_starting_price" in df.columns:
        sp = pd.to_numeric(df["sp_starting_price"], errors="coerce")
    else:
        sp = pd.Series(np.nan, index=df.index, dtype=float)
    lp = pd.to_numeric(df.get("live_price"), errors="coerce")
    out = sp.where(sp.notna() & (sp > 0), lp)
    return out


def compute_fair_market_prob(df: pd.DataFrame, market_price: pd.Series) -> pd.Series:
    inv = (1.0 / market_price.where(market_price > 0)).fillna(0.0)
    return inv.groupby(df["race_id"]).transform(lambda s: s / s.sum() if s.sum() > 0 else 0.0)


def build_features(df: pd.DataFrame, feature_columns: list[str], drop_object_cols: bool) -> pd.DataFrame:
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

    if obj_cols and drop_object_cols:
        feats = feats.drop(columns=obj_cols)
        obj_cols = []

    if obj_cols:
        feats[obj_cols] = feats[obj_cols].fillna("")
        feats = pd.get_dummies(feats, columns=obj_cols, dummy_na=False)

    feats = feats.replace([np.inf, -np.inf], np.nan)
    return feats.reindex(columns=feature_columns, fill_value=0.0)

def time_split_races(df: pd.DataFrame, train_frac: float) -> tuple[set, set]:
    if "start_time_iso" not in df.columns:
        raise ValueError("runner_rankings must include start_time_iso for time-based backtest split.")
    races = (
        df[["race_id", "start_time_iso"]]
        .drop_duplicates()
        .assign(start_time=lambda t: pd.to_datetime(t["start_time_iso"], errors="coerce", utc=True))
    )
    races = races.dropna(subset=["start_time"]).sort_values("start_time")
    if races.empty:
        raise ValueError("No parseable start_time_iso values for time-based split.")
    n_train = int(len(races) * float(train_frac))
    n_train = max(1, min(n_train, len(races) - 1))
    train_races = set(races.head(n_train)["race_id"].values)
    test_races = set(races.tail(len(races) - n_train)["race_id"].values)
    return train_races, test_races


def split_races(race_ids: pd.Series, seed: int, valid_frac: float = 0.2) -> tuple[set, set]:
    races = race_ids.dropna().unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(races)
    n_valid = int(len(races) * valid_frac)
    valid = set(races[:n_valid])
    train = set(races[n_valid:])
    return train, valid


def race_logloss(df: pd.DataFrame, prob_col: str = "model_prob") -> float:
    # One winner per race: -log(p_winner), averaged across races.
    winners = df[df["is_winner"] == 1][["race_id", prob_col]].copy()
    winners[prob_col] = pd.to_numeric(winners[prob_col], errors="coerce")
    winners = winners.dropna(subset=[prob_col])
    if winners.empty:
        return float("nan")
    p = winners[prob_col].clip(1e-12, 1.0).values
    return float((-np.log(p)).mean())


def fit_temperature(work: pd.DataFrame, candidate_ts: list[float], seed: int) -> float:
    _, valid_races = split_races(work["race_id"], seed=seed, valid_frac=0.2)
    valid = work[work["race_id"].isin(valid_races)].copy()
    best_t = candidate_ts[0]
    best_ll = float("inf")
    for t in candidate_ts:
        valid["model_prob"] = valid.groupby("race_id")["xgb_margin"].transform(
            lambda s: softmax_within_race(s, temperature=t)
        )
        ll = race_logloss(valid, prob_col="model_prob")
        if np.isnan(ll):
            continue
        if ll < best_ll:
            best_ll = ll
            best_t = float(t)
    return best_t


def main() -> None:
    args = parse_args()

    db_path = Path(args.db)
    model_path = Path(args.model)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")
    if not model_path.exists():
        raise SystemExit(f"Model bundle not found: {model_path} (train it first)")

    bundle = joblib.load(model_path)
    feature_columns = bundle["feature_columns"]
    model = bundle["model"]

    df = load_resulted_rows(str(db_path))
    if df.empty:
        print("No resulted rows found in runner_rankings (finish_place IS NOT NULL).")
        return

    train_races, test_races = time_split_races(df, train_frac=float(args.train_frac))
    train_df = df[df["race_id"].isin(train_races)].copy()
    test_df = df[df["race_id"].isin(test_races)].copy()

    if args.train_in_script:
        X_train = build_features(train_df, feature_columns, drop_object_cols=bool(args.drop_object_cols))
        y_train = pd.to_numeric(train_df["is_winner"], errors="coerce").fillna(0).astype(int)
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=800,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=int(args.seed),
            n_jobs=-1,
        )
        model.fit(X_train, y_train, verbose=False)

    market_price = choose_market_price(test_df)
    market_prob = compute_fair_market_prob(test_df, market_price)

    X = build_features(test_df, feature_columns, drop_object_cols=bool(args.drop_object_cols))
    margins = model.predict(X, output_margin=True)

    work = test_df.copy()
    work["xgb_margin"] = margins

    temperature = float(args.softmax_temperature)
    if args.fit_temperature:
        grid = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        temperature = fit_temperature(work, grid, seed=int(args.seed))
        print(f"Fitted softmax temperature: {temperature:.3f} (grid={grid})")

    work["model_prob"] = work.groupby("race_id")["xgb_margin"].transform(
        lambda s: softmax_within_race(s, temperature=temperature)
    )
    work["market_price"] = market_price
    work["market_prob"] = market_prob
    work["edge"] = work["model_prob"] - work["market_prob"]

    base = work[(work["market_price"].notna()) & (work["market_price"] > 0)].copy()
    eligible = base[base["edge"] >= float(args.min_edge)].copy()
    if eligible.empty:
        print("No eligible rows after filters.")
        return

    eligible = eligible.sort_values("edge", ascending=False)
    k = max(1, int(len(eligible) * float(args.top_frac)))
    top = eligible.head(k).copy()
    top_key = set(zip(top["race_id"].values, top["runner_number"].values)) if "runner_number" in top.columns else None

    y = pd.to_numeric(top["is_winner"], errors="coerce").fillna(0).astype(int)
    price = pd.to_numeric(top["market_price"], errors="coerce")
    profit = np.where(y.values == 1, price.values - 1.0, -1.0)

    bets = len(top)
    wins = int(y.sum())
    roi = float(np.nanmean(profit))  # per 1u stake

    print(f"Train races: {len(train_races):,} | Test races: {len(test_races):,}")
    print(f"Base runners (test, priced & resulted): {len(base):,}")
    print(f"Eligible runners (edge >= {args.min_edge:.4f}): {len(eligible):,}")
    print(f"Top fraction     : {args.top_frac:.4f}  ->  bets={bets:,}")
    print(f"Wins             : {wins:,}  ({wins/bets:.2%})")
    print(f"ROI (1u flat)    : {roi:.4f}  (profit per bet)")
    print(f"Avg edge         : {float(top['edge'].mean()):.4f}")
    print(f"Avg model_prob   : {float(top['model_prob'].mean()):.4f}")
    print(f"Avg market_prob  : {float(top['market_prob'].mean()):.4f}")
    print(f"Avg market_price : {float(price.mean()):.2f}")

    # Per-race top-rated runner: is it in the global top-% set?
    per_race_top = (
        base.sort_values(["race_id", "model_prob"], ascending=[True, False])
        .groupby("race_id", as_index=False)
        .head(1)
        .copy()
    )
    if top_key is not None:
        per_race_top["in_top_set"] = [
            (rid, rnum) in top_key for rid, rnum in zip(per_race_top["race_id"], per_race_top["runner_number"])
        ]
        in_count = int(per_race_top["in_top_set"].sum())
        total_races = int(per_race_top["race_id"].nunique())
        print(f"\nPer-race top runner in top set: {in_count:,}/{total_races:,} races ({in_count/total_races:.2%})")

        picked = per_race_top[per_race_top["in_top_set"]].copy()
        if not picked.empty:
            y2 = pd.to_numeric(picked["is_winner"], errors="coerce").fillna(0).astype(int)
            p2 = pd.to_numeric(picked["market_price"], errors="coerce")
            profit2 = np.where(y2.values == 1, p2.values - 1.0, -1.0)
            print(f"Those picks win-rate: {int(y2.sum()):,}/{len(picked):,} ({int(y2.sum())/len(picked):.2%})")
            print(f"Those picks ROI (1u flat): {float(np.nanmean(profit2)):.4f}")

    # Small sanity preview
    show_cols = [
        "start_time_iso",
        "competition_name",
        "race_number",
        "race_id",
        "runner_number",
        "runner_name",
        "market_price",
        "model_prob",
        "market_prob",
        "edge",
        "is_winner",
    ]
    show_cols = [c for c in show_cols if c in top.columns]
    preview = top[show_cols].head(15).copy()
    for c in ["model_prob", "market_prob", "edge"]:
        if c in preview.columns:
            preview[c] = preview[c].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "")
    if "is_winner" in preview.columns:
        preview["is_winner"] = preview["is_winner"].astype(int)

    print("\nPreview (first 15):")
    try:
        print(preview.to_string(index=False))
    except Exception:
        print(preview)


if __name__ == "__main__":
    main()
