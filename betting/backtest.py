"""Shared pipeline and historical backtest entry point."""

from __future__ import annotations

import sqlite3

import pandas as pd

from .config import CONFIG
from .db import load_draw_bias_table, load_jockey_stats_table, load_race_runners, load_trainer_stats_table
from .validation import validate_input
from .features import build_features
from .meta_model import add_meta_model_signal
from .scoring import score_runners
from .probabilities import assign_probabilities
from .calibration import calibrate_probabilities
from .edge import calculate_edges
from .filters import apply_filters
from .staking import assign_stakes
from .settlement import settle_bets


def run_pipeline(
    df: pd.DataFrame,
    config: dict,
    draw_bias_df: pd.DataFrame | None = None,
    jockey_stats_df: pd.DataFrame | None = None,
    trainer_stats_df: pd.DataFrame | None = None,
    apply_filters_and_stakes: bool = True,
) -> pd.DataFrame:
    """Run the shared pre-settlement betting pipeline."""
    if (draw_bias_df is None or jockey_stats_df is None or trainer_stats_df is None) and config.get("database_path"):
        with sqlite3.connect(config["database_path"]) as conn:
            if draw_bias_df is None:
                draw_bias_df = load_draw_bias_table(conn)
            if jockey_stats_df is None:
                jockey_stats_df = load_jockey_stats_table(conn)
            if trainer_stats_df is None:
                trainer_stats_df = load_trainer_stats_table(conn)
    df = validate_input(df, config)
    df = build_features(df, config, draw_bias_df, jockey_stats_df, trainer_stats_df)
    df = add_meta_model_signal(df, config, draw_bias_df, jockey_stats_df, trainer_stats_df)
    df = score_runners(df, config)
    df = assign_probabilities(df, config)
    df = calibrate_probabilities(df, config)
    df = calculate_edges(df, config)
    if apply_filters_and_stakes:
        df = apply_filters(df, config)
        df = assign_stakes(df, config)
    return df


def run_backtest(
    config: dict | None = None,
    return_scored: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Load historical rows, run the shared pipeline, and settle bets."""
    runtime_config = dict(CONFIG if config is None else config)
    db_path = runtime_config["database_path"]
    df = load_race_runners(db_path, "backtest", runtime_config)
    with sqlite3.connect(db_path) as conn:
        draw_bias_df = load_draw_bias_table(conn)
        jockey_stats_df = load_jockey_stats_table(conn)
        trainer_stats_df = load_trainer_stats_table(conn)
    scored = run_pipeline(
        df,
        runtime_config,
        draw_bias_df=draw_bias_df,
        jockey_stats_df=jockey_stats_df,
        trainer_stats_df=trainer_stats_df,
        apply_filters_and_stakes=False,
    )
    bets = assign_stakes(apply_filters(scored, runtime_config), runtime_config)
    settled = settle_bets(bets, runtime_config)
    if return_scored:
        return settled, scored
    return settled
