#!/usr/bin/env python3
"""Inspect a single race: show all runners with scores, probabilities, and edges."""

import argparse
import sqlite3
import sys
from numbers import Real
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from betting.backtest import run_pipeline
from betting.config import CONFIG
from betting.db import load_draw_bias_table, load_jockey_stats_table, load_single_race, load_trainer_stats_table
from betting.explain import build_ranking_table, format_runner_verbose, format_weight_header, transparent_component_score
from betting.filters import candidate_mask
from betting.scoring import get_effective_weights
from betting.staking import FibonacciStaker, _FIB_SEQUENCE, replay_fibonacci_level


def parse_args():
    p = argparse.ArgumentParser(description="Inspect a single race")
    p.add_argument("race_id", help="race_id to inspect")
    p.add_argument("--db", default=None)
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show per-runner raw score input breakdown",
    )
    p.add_argument(
        "--fib-level",
        type=int,
        default=None,
        help="Current Fibonacci level (0=base, 1=2 units, 2=3 units, etc). Defaults to 0.",
    )
    return p.parse_args()


def build_runtime_config(args):
    cfg = dict(CONFIG)
    cfg["min_edge"] = -999.0
    cfg["min_price"] = 0.0
    cfg["max_price"] = 9999.0
    cfg["min_field_size"] = 0
    cfg["max_field_size"] = 9999
    cfg["max_model_rank"] = 9999
    cfg["min_raw_edge"] = -999.0
    cfg["min_ev"] = -999.0
    cfg["min_score_gap_to_next"] = 0.0
    cfg["allow_multiple_bets_per_race"] = True
    cfg["exclude_runner_if_no_live_price"] = False
    cfg["exclude_race_if_price_coverage_below"] = 0.0
    cfg["min_recent_form_count"] = 0
    cfg["min_model_probability"] = 0.0
    cfg["min_model_vs_market_ratio"] = 0.0
    if args.db:
        cfg["database_path"] = args.db
    return cfg


def print_section(title: str) -> None:
    width = 100
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def _candidate_rejection_reason(row: pd.Series, config: dict) -> str:
    reasons: list[str] = []
    raw_edge = float(pd.to_numeric(row.get("raw_edge"), errors="coerce") or 0.0)
    min_raw_edge = float(config.get("min_raw_edge", 0.0))
    if raw_edge < min_raw_edge:
        reasons.append(f"raw_edge<{min_raw_edge:.0%}")

    ev = float(pd.to_numeric(row.get("ev"), errors="coerce") or 0.0)
    min_ev = float(config.get("min_ev", 0.0))
    if ev < min_ev:
        reasons.append(f"ev<{min_ev:.0%}")

    price = float(pd.to_numeric(row.get("live_price"), errors="coerce") or 0.0)
    min_price = float(config.get("min_price", 0.0))
    max_price = float(config.get("max_price", float("inf")))
    if not (min_price <= price <= max_price):
        reasons.append(f"price_outside_{min_price}-{max_price}")
    if price > 50:
        reasons.append("longshot_ev_unreliable")

    rank = float(pd.to_numeric(row.get("model_rank"), errors="coerce") or 9999.0)
    max_rank = float(config.get("max_model_rank", 9999))
    if rank > max_rank:
        reasons.append(f"rank>{int(max_rank)}")

    if not bool(row.get("race_integrity_ok", True)):
        reasons.append("race_integrity_fail")

    model_prob = float(pd.to_numeric(row.get("model_prob"), errors="coerce") or 0.0)
    market_prob = float(pd.to_numeric(row.get("raw_market_prob"), errors="coerce") or 0.0)
    min_model_prob = float(config.get("min_model_probability", 0.0))
    min_ratio = float(config.get("min_model_vs_market_ratio", 0.0))
    min_score_gap = float(config.get("min_score_gap_to_next", 0.0))
    if model_prob < min_model_prob:
        reasons.append(f"model_prob<{min_model_prob:.0%}")
    if market_prob <= 0:
        reasons.append("market_prob_missing")
    elif min_ratio > 0 and model_prob < market_prob * min_ratio:
        reasons.append(f"model_vs_market<{min_ratio:.2f}x")
    score_gap = float(pd.to_numeric(row.get("model_score_gap_to_next"), errors="coerce") or 0.0)
    if score_gap < min_score_gap:
        reasons.append(f"score_gap<{min_score_gap:.3f}")

    return "QUALIFIED" if not reasons else "; ".join(reasons)


def clamp_fibonacci_level(level: int, max_level: int) -> int:
    return max(0, min(int(level), max_level))


def discover_settled_bets_table(conn: sqlite3.Connection) -> tuple[str, str] | None:
    time_columns = ("race_start_time", "start_time_iso", "settled_at", "created_at")
    price_columns = {"live_price", "settlement_price", "sp_starting_price"}
    for (table_name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"):
        columns = {row[1] for row in conn.execute(f'PRAGMA table_info("{table_name}")')}
        if "stake" not in columns or "is_winner" not in columns or not (columns & price_columns):
            continue
        time_col = next((col for col in time_columns if col in columns), None)
        if time_col:
            return table_name, time_col
    return None


def determine_fibonacci_level(db_path: str, current_race_start: str | None, args: argparse.Namespace, config: dict) -> int:
    max_level = min(int(config.get("fib_max_level", 10)), len(_FIB_SEQUENCE) - 1)
    if args.fib_level is not None:
        return clamp_fibonacci_level(args.fib_level, max_level)

    with sqlite3.connect(db_path) as conn:
        source = discover_settled_bets_table(conn)
        if source is None:
            return 0
        table_name, time_col = source
        sql = f'SELECT "is_winner" FROM "{table_name}" WHERE "is_winner" IN (0, 1)'
        params: list[object] = []
        if current_race_start:
            sql += f' AND "{time_col}" < ?'
            params.append(current_race_start)
        sql += f' ORDER BY "{time_col}" DESC LIMIT ?'
        params.append(len(_FIB_SEQUENCE) * 10)
        rows = conn.execute(sql, params).fetchall()

    results = [int(row[0]) for row in reversed(rows)]
    return replay_fibonacci_level(results, variant=config.get("fib_variant", "two_back"), max_level=max_level)


def print_fibonacci_advisor(args: argparse.Namespace, config: dict, bets: pd.DataFrame, current_race_start: str | None) -> None:
    variant = config.get("fib_variant", "two_back")
    base_unit = float(config.get("fib_base_unit", 1.0))
    max_level = min(int(config.get("fib_max_level", 10)), len(_FIB_SEQUENCE) - 1)
    current_level = determine_fibonacci_level(config["database_path"], current_race_start, args, config)

    staker = FibonacciStaker(variant=variant, base_unit=base_unit, max_level=max_level)
    staker.level = current_level
    recommended_stake = staker.current_stake()

    win_staker = FibonacciStaker(variant=variant, base_unit=base_unit, max_level=max_level)
    win_staker.level = current_level
    win_staker.on_win()

    loss_staker = FibonacciStaker(variant=variant, base_unit=base_unit, max_level=max_level)
    loss_staker.level = current_level
    loss_staker.on_loss()

    print_section("FIBONACCI STAKE ADVISOR")
    print(f"  Sequence:  {' → '.join(str(value) for value in _FIB_SEQUENCE)}")
    print(f"  Variant:   {variant}")
    print(f"  Base unit: {base_unit:.1f}")
    print()
    print(f"  Current level: {current_level}  →  Recommended stake: {recommended_stake:.1f} units")

    if not bets.empty:
        top_bet = bets.iloc[0]
        print()
        print(
            f"  *** BET: #{int(top_bet['runner_number'])} {top_bet['runner_name']} "
            f"at {float(top_bet['live_price']):.2f} — stake {recommended_stake:.1f} units ***"
        )

    print()
    print(f"  On WIN  → drop to level {win_staker.level} (stake: {win_staker.current_stake():.1f} units)")
    print(f"  On LOSS → advance to level {loss_staker.level} (stake: {loss_staker.current_stake():.1f} units)")
    print()
    print("  Level  Sequence  Stake")

    start_level = max(0, current_level - 1)
    end_level = min(max_level, start_level + 3)
    start_level = max(0, end_level - 3)
    for level in range(start_level, end_level + 1):
        marker = "   ← current" if level == current_level else ""
        stake = base_unit * _FIB_SEQUENCE[level]
        print(f"  {level:>5}  {_FIB_SEQUENCE[level]:>8}  {stake:>5.1f}{marker}")


def main():
    args = parse_args()
    cfg = build_runtime_config(args)
    display_cfg = dict(CONFIG)
    if args.db:
        display_cfg["database_path"] = args.db
    df = load_single_race(cfg["database_path"], args.race_id)
    if df.empty:
        print(f"No runners found for race_id: {args.race_id}")
        return

    row0 = df.iloc[0]
    print(f"\nRace: {row0.get('race_name', '?')} | {row0.get('competition_name', '?')} | Race {row0.get('race_number', '?')}")
    declared_field = row0.get('field_size', '?')
    active_field = row0.get('active_field_size', len(df))
    print(
        f"Distance: {row0.get('distance_m', '?')}m | Track: {row0.get('track_status', '?')} | Declared Field: {declared_field} | Active: {active_field}"
    )
    if isinstance(declared_field, Real) and isinstance(active_field, Real) and declared_field != active_field:
        excluded = int(declared_field - active_field)
        print(f"warning_field_size_mismatch: declared={int(declared_field)}, active={int(active_field)}, excluded={excluded}")
    print(f"Start: {row0.get('start_time_iso', '?')}\n")

    conn = sqlite3.connect(cfg["database_path"])
    try:
        draw_bias_df = load_draw_bias_table(conn)
        jockey_stats_df = load_jockey_stats_table(conn)
        trainer_stats_df = load_trainer_stats_table(conn)
    finally:
        conn.close()

    try:
        df_pipeline = run_pipeline(
            df,
            cfg,
            draw_bias_df=draw_bias_df,
            jockey_stats_df=jockey_stats_df,
            trainer_stats_df=trainer_stats_df,
        )
    except Exception as exc:
        print(f"Pipeline error: {exc}")
        return

    effective_weights, _ = get_effective_weights(display_cfg)
    meta_rows = int(pd.to_numeric(df_pipeline.get("meta_model_score"), errors="coerce").fillna(0.0).gt(0).sum())
    if meta_rows > 0:
        print(
            f"info_score_blend: transparent weighted components are shown below as `Base`; "
            f"`Score` is the live score after the meta-model blend ({meta_rows} runners with meta signal)."
        )
    else:
        for _, row in df_pipeline.iterrows():
            component_sum = transparent_component_score(row, display_cfg)
            ms = float(row.get("model_score", 0))
            if abs(component_sum - ms) > 0.0001:
                print(
                    f"WARNING score mismatch {row['runner_name']}: "
                    f"components={component_sum:.6f} model_score={ms:.6f} diff={component_sum-ms:.6f}"
                )

    print_section(f"MODEL RANKING  ({format_weight_header(display_cfg)})")
    ranking = build_ranking_table(df_pipeline, display_cfg)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(ranking.to_string(index=False))

    if args.verbose:
        print_section("SCORE DETAIL  (raw inputs → component scores → weighted contributions)")
        for _, row in df_pipeline.sort_values("model_rank").iterrows():
            print(format_runner_verbose(row, display_cfg, field_df=df_pipeline))
    else:
        print("\n  Tip: run with --verbose to see per-runner raw score breakdowns")

    min_raw_edge = display_cfg.get("min_raw_edge", 0.0)
    min_ev = display_cfg.get("min_ev", 0.0)
    min_price = display_cfg.get("min_price", 2.0)
    max_price = display_cfg.get("max_price", 15.0)
    max_rank = display_cfg.get("max_model_rank", 3)
    min_model_prob = display_cfg.get("min_model_probability", 0.0)
    min_score_gap = display_cfg.get("min_score_gap_to_next", 0.0)
    min_ratio = display_cfg.get("min_model_vs_market_ratio", 0.0)
    mask = candidate_mask(df_pipeline, display_cfg)
    bets = df_pipeline[mask].sort_values("ev", ascending=False)
    ratio_text = "disabled" if min_ratio <= 0 else f"{min_ratio:.2f}x"
    print_section(
        f"BET CANDIDATES  (raw_edge≥{min_raw_edge:.0%}  ev≥{min_ev:.0%}  "
        f"price {min_price}–{max_price}  rank≤{max_rank}  model≥{min_model_prob:.0%}  "
        f"gap≥{min_score_gap:.3f}  "
        f"model/raw_mkt≥{ratio_text})"
    )
    if bets.empty:
        print("  No bets qualify under current config thresholds.")
    else:
        bet_cols = [
            "runner_number",
            "runner_name",
            "live_price",
            "model_rank",
            "model_prob",
            "raw_market_prob",
            "market_implied_prob",
            "raw_edge",
            "fair_edge",
            "ev",
            "model_score_gap_to_next",
        ]
        bet_show = [c for c in bet_cols if c in bets.columns]
        print(bets[bet_show].to_string(index=False))

    edge_watch = df_pipeline[df_pipeline["raw_edge"] > 0].copy()
    if not edge_watch.empty:
        edge_watch["candidate_reason"] = edge_watch.apply(
            _candidate_rejection_reason,
            axis=1,
            args=(display_cfg,),
        )
        print_section("EDGE WATCHLIST  (positive raw edge with qualification reason)")
        cols = [
            "runner_number",
            "runner_name",
            "live_price",
            "model_rank",
            "model_prob",
            "raw_market_prob",
            "market_implied_prob",
            "raw_edge",
            "fair_edge",
            "ev",
            "model_score_gap_to_next",
            "candidate_reason",
        ]
        show_cols = [c for c in cols if c in edge_watch.columns]
        print(edge_watch.sort_values("ev", ascending=False)[show_cols].to_string(index=False))

    print_fibonacci_advisor(args, cfg, bets, row0.get("start_time_iso"))


if __name__ == "__main__":
    main()
