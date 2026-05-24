#!/usr/bin/env python3
"""Ranking quality metrics for the betting model."""

from __future__ import annotations

import argparse
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from betting.backtest import run_backtest, run_pipeline
from betting.config import CONFIG
from betting.db import load_backtest_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ranking quality metrics for the betting model")
    parser.add_argument("--db", default=None, help="Path to SQLite database")
    return parser.parse_args()


def score_all(config: dict) -> pd.DataFrame:
    cfg = dict(config)
    cfg["min_edge"] = -999.0
    cfg["min_raw_edge"] = -999.0
    cfg["min_ev"] = -999.0
    cfg["min_price"] = 0.0
    cfg["max_price"] = 9999.0
    cfg["min_field_size"] = 0
    cfg["max_field_size"] = 9999
    cfg["max_model_rank"] = 9999
    cfg["allow_multiple_bets_per_race"] = True
    cfg["exclude_runner_if_no_live_price"] = False
    cfg["exclude_race_if_price_coverage_below"] = 0.0
    cfg["min_recent_form_count"] = 0
    cfg["min_model_vs_market_ratio"] = 0.0
    db_path = cfg.get("database_path", "database/race_reports.sqlite")
    raw = load_backtest_data(db_path)
    return run_pipeline(raw, cfg)


def ranking_metrics(df: pd.DataFrame) -> dict:
    winner_col = CONFIG["winner_col"]
    results = []
    for race_id, grp in df.groupby("race_id"):
        winner = grp[grp[winner_col] == 1]
        if winner.empty:
            continue
        rank = winner["model_rank"].iloc[0]
        prob = winner["model_prob"].iloc[0]
        if pd.isna(rank) or pd.isna(prob):
            continue
        results.append(
            {
                "race_id": race_id,
                "winner_rank": int(rank),
                "winner_prob": float(prob),
            }
        )
    rdf = pd.DataFrame(results)
    total = len(rdf)
    if total == 0:
        return {
            "races": 0,
            "top1_hit_rate": float("nan"),
            "top2_containment": float("nan"),
            "top3_containment": float("nan"),
            "avg_winner_rank": float("nan"),
            "mrr": float("nan"),
            "log_loss": float("nan"),
        }
    return {
        "races": total,
        "top1_hit_rate": (rdf["winner_rank"] == 1).mean(),
        "top2_containment": (rdf["winner_rank"] <= 2).mean(),
        "top3_containment": (rdf["winner_rank"] <= 3).mean(),
        "avg_winner_rank": rdf["winner_rank"].mean(),
        "mrr": (1.0 / rdf["winner_rank"]).mean(),
        "log_loss": (-np.log(rdf["winner_prob"].clip(1e-6, 1.0))).mean(),
    }


def roi_by_model_rank(df_filtered: pd.DataFrame) -> pd.DataFrame:
    if df_filtered.empty:
        return pd.DataFrame(
            columns=[
                "model_rank",
                "bets",
                "wins",
                "total_staked",
                "total_profit",
                "roi",
                "strike_rate",
                "avg_price",
                "avg_edge",
            ]
        )

    grouped = df_filtered.groupby("model_rank", dropna=False)
    table = pd.DataFrame(
        {
            "bets": grouped["stake"].count(),
            "wins": grouped[CONFIG["winner_col"]].sum(),
            "total_staked": grouped["stake"].sum(),
            "total_profit": grouped["profit"].sum(),
            "avg_price": grouped["live_price"].mean(),
            "avg_edge": grouped["edge"].mean(),
        }
    ).reset_index()
    table["roi"] = table["total_profit"] / table["total_staked"].replace(0, np.nan)
    table["strike_rate"] = table["wins"] / table["bets"].replace(0, np.nan)
    table["model_rank"] = pd.to_numeric(table["model_rank"], errors="coerce").astype("Int64")
    return table[
        [
            "model_rank",
            "bets",
            "wins",
            "total_staked",
            "total_profit",
            "roi",
            "strike_rate",
            "avg_price",
            "avg_edge",
        ]
    ].sort_values("model_rank").reset_index(drop=True)


def sparse_subsets(df: pd.DataFrame) -> pd.DataFrame:
    """Top-3 containment for races where the winner had sparse data."""
    winner_col = CONFIG["winner_col"]
    rows = []
    subsets = {
        "distance_starts=0": df["distance_starts"] == 0,
        "track_starts=0": df["track_starts"] == 0,
        "jockey_starts=0": df["horse_jockey_starts"] == 0,
        "all_races": pd.Series(True, index=df.index),
    }
    for label, mask in subsets.items():
        sub = df.loc[mask]
        winner_rows = sub[sub[winner_col] == 1]
        if len(winner_rows) == 0:
            rows.append({"subset": label, "races": 0, "top3_pct": float("nan")})
            continue
        top3 = (winner_rows["model_rank"] <= 3).mean()
        rows.append({"subset": label, "races": len(winner_rows), "top3_pct": round(top3 * 100, 1)})
    return pd.DataFrame(rows)


def _format_roi_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in ["total_staked", "total_profit", "avg_price", "avg_edge"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(3)
    out["roi"] = (pd.to_numeric(out["roi"], errors="coerce") * 100).round(1)
    out["strike_rate"] = (pd.to_numeric(out["strike_rate"], errors="coerce") * 100).round(1)
    return out.rename(columns={"roi": "roi_pct", "strike_rate": "strike_rate_pct"})


def _quietly(func, *args, **kwargs):
    buffer = StringIO()
    with redirect_stdout(buffer):
        return func(*args, **kwargs)


def main() -> None:
    args = parse_args()
    config = dict(CONFIG)
    if args.db:
        config["database_path"] = args.db

    df_all = _quietly(score_all, config)
    df_filtered = _quietly(run_backtest, config)

    print("=== RANKING QUALITY METRICS ===")
    metrics = ranking_metrics(df_all)
    print(f"  Races scored:       {metrics['races']}")
    print(f"  Top-1 hit rate:     {metrics['top1_hit_rate']:.1%}")
    print(f"  Top-2 containment:  {metrics['top2_containment']:.1%}")
    print(f"  Top-3 containment:  {metrics['top3_containment']:.1%}")
    print(f"  Avg winner rank:    {metrics['avg_winner_rank']:.2f}")
    print(f"  MRR:                {metrics['mrr']:.4f}")
    print(f"  Log loss:           {metrics['log_loss']:.4f}")

    print("\n=== SPARSE DATA SUBSETS (top-3 containment) ===")
    print(sparse_subsets(df_all).to_string(index=False))

    print("\n=== ROI BY MODEL RANK (filtered bets only) ===")
    roi_table = _format_roi_table(roi_by_model_rank(df_filtered))
    if roi_table.empty:
        print("No filtered bets")
    else:
        print(roi_table.to_string(index=False))


if __name__ == "__main__":
    main()
