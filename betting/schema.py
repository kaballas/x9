"""Schema contracts for runner-level betting data."""

from __future__ import annotations

import pandas as pd

REQUIRED_FEATURE_COLUMNS: list[str] = [
    "race_id",
    "selection_id",
    "runner_name",
    "competition_name",
    "race_number",
    "start_time_iso",
    "distance_m",
    "track_status",
    "field_size",
    "active_field_size",
    "runner_number",
    "draw_number",
    "tempo",
    "speed_rating",
    "dry_rating",
    "wet_rating",
    "good_starts",
    "good_wins",
    "soft_starts",
    "soft_wins",
    "heavy_starts",
    "heavy_wins",
    "distance_starts",
    "distance_wins",
    "track_starts",
    "track_wins",
    "horse_jockey_starts",
    "horse_jockey_wins",
    "recent_runs_count",
    "recent_avg_place",
    "recent_avg_place_3",
    "recent_avg_place_5",
    "recent_win_rate_5",
    "recent_top3_rate_5",
    "open_price",
    "fluc1",
    "fluc2",
    "live_price",
    "price_quality",
]

REQUIRED_RESULT_COLUMNS: list[str] = [
    "finish_place",
    "result_code",
    "status",
    "is_winner",
    "top3_mask",
    "live_price",
]

REQUIRED_PRICE_COLUMNS: list[str] = [
    "open_price",
    "fluc1",
    "fluc2",
    "sp_starting_price",
    "live_price",
    "price_quality",
]

IDENTITY_COLUMNS: list[str] = [
    "race_id",
    "selection_id",
    "race_number",
    "race_name",
    "competition_id",
    "competition_name",
    "country",
    "start_time_iso",
    "runner_number",
    "runner_name",
    "draw_number",
]


def validate_schema(df: pd.DataFrame, required_cols: list[str]) -> None:
    """Raise ValueError listing all missing columns."""
    missing = sorted(set(required_cols) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
