#!/usr/bin/env python3
"""Score and rank every runner in every race, writing results to runner_rankings table.

Usage examples
--------------
# Score all races (finished + live) and replace the table:
python3 scripts/rank_all_races.py

# Score only historical/finished races:
python3 scripts/rank_all_races.py --mode backtest

# Score only upcoming/live races:
python3 scripts/rank_all_races.py --mode live

# Append instead of replace (e.g. incremental updates):
python3 scripts/rank_all_races.py --mode live --if-exists append

# Use a different database:
python3 scripts/rank_all_races.py --db /path/to/other.sqlite
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from io import StringIO
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from betting.backtest import run_pipeline
from betting.config import CONFIG
from betting.db import (
    _BASE_SELECT,
    load_draw_bias_table,
    load_jockey_stats_table,
    load_trainer_stats_table,
)
from betting.explain import COMPONENT_DEFS
from betting.scoring import get_effective_weights

# Columns to persist — race meta, runner info, all model outputs, all component scores.
_RACE_COLS = [
    "race_id",
    "race_number",
    "race_name",
    "competition_name",
    "start_time_iso",
    "distance_m",
    "track_status",
    "field_size",
    "active_field_size",
]
_RUNNER_COLS = [
    "runner_number",
    "runner_name",
    "draw_number",
    "jockey",
    "trainer",
    "live_price",
    "open_price",
    "fluc1",
    "sp_starting_price",
    "finish_place",
    "is_winner",
]
_MODEL_COLS = [
    "model_rank",
    "model_score",
    "model_score_gap_to_next",
    "model_prob",
    "raw_market_prob",
    "fair_market_prob",
    "raw_edge",
    "fair_edge",
    "ev",
]
# Component score columns come from COMPONENT_DEFS
_COMPONENT_COLS = [scol for _, scol, _ in COMPONENT_DEFS]

_ALL_OUTPUT_COLS = _RACE_COLS + _RUNNER_COLS + _MODEL_COLS + _COMPONENT_COLS

# SQL queries for each mode --------------------------------------------------

_ALL_WHERE = """
WHERE result_code != 'V'
ORDER BY start_time_iso, race_id, runner_number
"""

_BACKTEST_WHERE = """
WHERE status = 'finished'
  AND result_code IN ('W', 'P', 'L')
ORDER BY start_time_iso, race_id, runner_number
"""

_LIVE_WHERE = """
WHERE status = 'no_result'
  AND result_code != 'V'
  AND (source_betting_status IS NULL OR source_betting_status <> 'RESULTED')
ORDER BY start_time_iso, race_id, runner_number
"""

_MODE_QUERIES = {
    "all": _BASE_SELECT + _ALL_WHERE,
    "backtest": _BASE_SELECT + _BACKTEST_WHERE,
    "live": _BASE_SELECT + _LIVE_WHERE,
}


def _open_config(args: argparse.Namespace) -> dict:
    cfg = dict(CONFIG)
    # Disable all filters so every runner is scored and ranked.
    cfg["min_edge"] = -999.0
    cfg["min_raw_edge"] = -999.0
    cfg["min_ev"] = -999.0
    cfg["min_price"] = 0.0
    cfg["max_price"] = 9999.0
    cfg["min_field_size"] = 0
    cfg["max_field_size"] = 9999
    cfg["max_model_rank"] = 9999
    cfg["min_model_probability"] = 0.0
    cfg["min_model_vs_market_ratio"] = 0.0
    cfg["min_score_gap_to_next"] = 0.0
    cfg["min_recent_form_count"] = 0
    cfg["allow_multiple_bets_per_race"] = True
    cfg["exclude_runner_if_no_live_price"] = False
    cfg["exclude_race_if_price_coverage_below"] = 0.0
    if args.db:
        cfg["database_path"] = args.db
    return cfg


def _load_runners(db_path: str, mode: str) -> pd.DataFrame:
    query = _MODE_QUERIES[mode]
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)


def _silence(fn, *args, **kwargs):
    """Call fn suppressing stdout (pipeline filter logs)."""
    buf = StringIO()
    with redirect_stdout(buf):
        return fn(*args, **kwargs)


def _select_output_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in _ALL_OUTPUT_COLS if c in df.columns]
    missing = [c for c in _ALL_OUTPUT_COLS if c not in df.columns]
    if missing:
        print(f"  note: {len(missing)} expected column(s) absent from pipeline output: {missing}")
    out = df[cols].copy()
    out["ranked_at"] = datetime.now(timezone.utc).isoformat()
    return out


def _print_ranking_stats(df: pd.DataFrame) -> None:
    """Print top-1/2/3/4 hit rates for races where the winner is known."""
    winner_col = "is_winner"
    if winner_col not in df.columns or "model_rank" not in df.columns:
        return

    # One row per race: what model_rank was the actual winner?
    winners = df[df[winner_col] == 1][["race_id", "model_rank", "runner_name", "runner_number"]].copy()
    winners["model_rank"] = pd.to_numeric(winners["model_rank"], errors="coerce")
    winners = winners.dropna(subset=["model_rank"])

    total = len(winners)
    if total == 0:
        print("  No races with a known winner — ranking stats unavailable.")
        return

    top1  = int((winners["model_rank"] == 1).sum())
    top2  = int((winners["model_rank"] <= 2).sum())
    top3  = int((winners["model_rank"] <= 3).sum())
    top4  = int((winners["model_rank"] <= 4).sum())
    avg   = winners["model_rank"].mean()
    mrr   = (1.0 / winners["model_rank"]).mean()

    w = 40
    print("─" * w)
    print(f"  RANKING ACCURACY  ({total:,} races with results)")
    print("─" * w)
    print(f"  Top-1 correct  : {top1:>5,}  ({top1/total:>6.1%})")
    print(f"  Top-2 correct  : {top2:>5,}  ({top2/total:>6.1%})")
    print(f"  Top-3 correct  : {top3:>5,}  ({top3/total:>6.1%})")
    print(f"  Top-4 correct  : {top4:>5,}  ({top4/total:>6.1%})")
    print(f"  Avg winner rank: {avg:>8.2f}")
    print(f"  MRR            : {mrr:>8.4f}")
    print("─" * w)

    # Distribution of winner ranks
    dist = winners["model_rank"].value_counts().sort_index().head(8)
    print("  Winner rank distribution:")
    for rank, count in dist.items():
        bar = "█" * int(count / total * 40)
        print(f"    Rank {int(rank):>2}: {count:>4,}  {bar}")
    print()


def _write_table(df: pd.DataFrame, db_path: str, if_exists: str) -> int:
    with sqlite3.connect(db_path) as conn:
        df.to_sql("runner_rankings", conn, if_exists=if_exists, index=False)
        count = conn.execute("SELECT COUNT(*) FROM runner_rankings").fetchone()[0]
    return count


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score and rank all runners, writing results to runner_rankings table"
    )
    p.add_argument("--db", default=None, help="Path to SQLite database (default: from config)")
    p.add_argument(
        "--mode",
        choices=["all", "backtest", "live"],
        default="all",
        help=(
            "Which races to score: "
            "'all' = every non-void row, "
            "'backtest' = finished/resulted races only, "
            "'live' = upcoming/unresulted races only (default: all)"
        ),
    )
    p.add_argument(
        "--if-exists",
        choices=["replace", "append"],
        default="replace",
        dest="if_exists",
        help="What to do if runner_rankings already exists (default: replace)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _open_config(args)
    db_path: str = cfg["database_path"]

    print(f"Database : {db_path}")
    print(f"Mode     : {args.mode}")
    print(f"If-exists: {args.if_exists}")
    print()

    # ── Load ──────────────────────────────────────────────────────────────
    print("Loading runners...", end=" ", flush=True)
    raw = _load_runners(db_path, args.mode)
    if raw.empty:
        print(f"\nNo rows found for mode={args.mode!r}. Nothing written.")
        return
    n_races = raw["race_id"].nunique()
    print(f"{len(raw):,} runners across {n_races:,} races")

    # ── Reference tables ──────────────────────────────────────────────────
    with sqlite3.connect(db_path) as conn:
        draw_bias_df = load_draw_bias_table(conn)
        jockey_stats_df = load_jockey_stats_table(conn)
        trainer_stats_df = load_trainer_stats_table(conn)

    # ── Score ─────────────────────────────────────────────────────────────
    print("Scoring (filters suppressed)...", end=" ", flush=True)
    scored = _silence(
        run_pipeline,
        raw,
        cfg,
        draw_bias_df=draw_bias_df,
        jockey_stats_df=jockey_stats_df,
        trainer_stats_df=trainer_stats_df,
        apply_filters_and_stakes=False,
    )
    print(f"done — {len(scored):,} rows scored")

    # ── Verify component sums ─────────────────────────────────────────────
    eff_weights, _ = get_effective_weights(cfg)
    mismatches = 0
    for _, row in scored.iterrows():
        comp_sum = sum(
            eff_weights.get(wkey, 0.0) * float(pd.to_numeric(row.get(scol, 0), errors="coerce") or 0)
            for wkey, scol, _ in COMPONENT_DEFS
        )
        ms = float(row.get("model_score", 0))
        if abs(comp_sum - ms) > 0.0005:
            mismatches += 1
    if mismatches:
        print(f"  WARNING: {mismatches} runner(s) had component-sum ≠ model_score (diff >0.0005)")
    else:
        print(f"  Component-sum check: all {len(scored):,} runners OK")

    # ── Select output columns ─────────────────────────────────────────────
    out = _select_output_cols(scored)

    # ── Write ─────────────────────────────────────────────────────────────
    action = "Replacing" if args.if_exists == "replace" else "Appending to"
    print(f"{action} runner_rankings...", end=" ", flush=True)
    total_rows = _write_table(out, db_path, args.if_exists)
    print(f"done")
    print()
    print(f"  Rows written this run : {len(out):,}")
    print(f"  Total rows in table   : {total_rows:,}")
    print(f"  Columns               : {len(out.columns)} ({', '.join(out.columns[:8])}, ...)")
    print()

    # ── Quick sanity: top-ranked runners in first few races ───────────────
    print("Sample — rank-1 runners from first 5 races:")
    rank1 = out[out["model_rank"] == 1].head(5)[
        ["race_id", "start_time_iso", "race_name", "runner_number", "runner_name",
         "model_score", "live_price", "finish_place"]
    ]
    print(rank1.to_string(index=False))
    print()

    # ── Ranking accuracy (only when results are available) ────────────────
    if args.mode in ("backtest", "all"):
        _print_ranking_stats(out)


if __name__ == "__main__":
    main()
