"""SQLite loaders for canonical runner-level betting data."""

from __future__ import annotations

import sqlite3

import pandas as pd

from .config import CONFIG

_SELECT_COLUMNS = [
    "race_id",
    "race_number",
    "race_name",
    "competition_id",
    "competition_name",
    "country",
    "class_name",
    "grade",
    "tempo",
    "distance_m",
    "track_status",
    "start_time_iso",
    "field_size",
    "active_field_size",
    "selection_id",
    "runner_number",
    "runner_name",
    "draw_number",
    "jockey",
    "trainer",
    "trainer_location",
    "runner_country",
    "blinkers",
    "sire",
    "dam",
    "weight_kg",
    "age",
    "sex",
    "speed_rating",
    "dry_rating",
    "wet_rating",
    "win_percentage",
    "place_percentage",
    "prize_money",
    "career_starts",
    "career_wins",
    "career_seconds",
    "career_thirds",
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
    "first_up_starts",
    "first_up_wins",
    "second_up_starts",
    "second_up_wins",
    "horse_jockey_starts",
    "horse_jockey_wins",
    "last_six",
    "form_fig",
    "expected_settling_position",
    "recent_runs_count",
    "recent_wins",
    "recent_places",
    "recent_avg_place",
    "recent_best_place",
    "recent_avg_place_3",
    "recent_avg_place_5",
    "recent_win_rate_5",
    "recent_top3_rate_5",
    "recent_avg_margin",
    "recent_best_margin",
    "recent_avg_margin_3",
    "recent_avg_starting_price",
    "recent_same_distance_runs",
    "recent_same_track_runs",
    "recent_same_condition_runs",
    "recent_days_since_last_run",
    "recent_1_place", "recent_1_total_runners", "recent_1_time", "recent_1_distance_m",
    "recent_1_weight_kg", "recent_1_barrier", "recent_1_track_status_value", "recent_1_starting_price",
    "recent_1_margin", "recent_1_date", "recent_1_track_name", "recent_1_track_status",
    "recent_1_jockey", "recent_1_class",
    "recent_2_place", "recent_2_total_runners", "recent_2_time", "recent_2_distance_m",
    "recent_2_weight_kg", "recent_2_barrier", "recent_2_track_status_value", "recent_2_starting_price",
    "recent_2_margin", "recent_2_date", "recent_2_track_name", "recent_2_track_status",
    "recent_2_jockey", "recent_2_class",
    "recent_3_place", "recent_3_total_runners", "recent_3_time", "recent_3_distance_m",
    "recent_3_weight_kg", "recent_3_barrier", "recent_3_track_status_value", "recent_3_starting_price",
    "recent_3_margin", "recent_3_date", "recent_3_track_name", "recent_3_track_status",
    "recent_3_jockey", "recent_3_class",
    "recent_4_place", "recent_4_total_runners", "recent_4_time", "recent_4_distance_m",
    "recent_4_weight_kg", "recent_4_barrier", "recent_4_track_status_value", "recent_4_starting_price",
    "recent_4_margin", "recent_4_date", "recent_4_track_name", "recent_4_track_status",
    "recent_4_jockey", "recent_4_class",
    "recent_5_place", "recent_5_total_runners", "recent_5_time", "recent_5_distance_m",
    "recent_5_weight_kg", "recent_5_barrier", "recent_5_track_status_value", "recent_5_starting_price",
    "recent_5_margin", "recent_5_date", "recent_5_track_name", "recent_5_track_status",
    "recent_5_jockey", "recent_5_class",
    "recent_6_place", "recent_6_total_runners", "recent_6_time", "recent_6_distance_m",
    "recent_6_weight_kg", "recent_6_barrier", "recent_6_track_status_value", "recent_6_starting_price",
    "recent_6_margin", "recent_6_date", "recent_6_track_name", "recent_6_track_status",
    "recent_6_jockey", "recent_6_class",
    "open_price",
    "fluc1",
    "fluc2",
    "sp_starting_price",
    "live_price",
    "price_quality",
    "finish_place",
    "result_code",
    "status",
    "is_winner",
    "top3_mask",
]

_BASE_SELECT = """
SELECT
  race_id, race_number, race_name, competition_id, competition_name, country,
  class_name, grade, tempo, distance_m, track_status, start_time_iso, field_size, active_field_size,
  selection_id, runner_number, runner_name, draw_number,
  jockey, trainer, trainer_location, runner_country, blinkers, sire, dam, weight_kg, age, sex,
  speed_rating, dry_rating, wet_rating,
  win_percentage, place_percentage, prize_money,
  career_starts, career_wins, career_seconds, career_thirds,
  good_starts, good_wins, soft_starts, soft_wins, heavy_starts, heavy_wins,
  distance_starts, distance_wins, track_starts, track_wins,
  first_up_starts, first_up_wins, second_up_starts, second_up_wins,
  horse_jockey_starts, horse_jockey_wins,
  last_six, form_fig, expected_settling_position,
  recent_runs_count, recent_wins, recent_places, recent_avg_place, recent_best_place,
  recent_avg_place_3, recent_avg_place_5, recent_win_rate_5, recent_top3_rate_5,
  recent_avg_margin, recent_best_margin, recent_avg_margin_3,
  recent_avg_starting_price, recent_same_distance_runs, recent_same_track_runs,
  recent_same_condition_runs, recent_days_since_last_run,
  recent_1_place, recent_1_total_runners, recent_1_time, recent_1_distance_m,
  recent_1_weight_kg, recent_1_barrier, recent_1_track_status_value, recent_1_starting_price,
  recent_1_margin, recent_1_date, recent_1_track_name, recent_1_track_status, recent_1_jockey, recent_1_class,
  recent_2_place, recent_2_total_runners, recent_2_time, recent_2_distance_m,
  recent_2_weight_kg, recent_2_barrier, recent_2_track_status_value, recent_2_starting_price,
  recent_2_margin, recent_2_date, recent_2_track_name, recent_2_track_status, recent_2_jockey, recent_2_class,
  recent_3_place, recent_3_total_runners, recent_3_time, recent_3_distance_m,
  recent_3_weight_kg, recent_3_barrier, recent_3_track_status_value, recent_3_starting_price,
  recent_3_margin, recent_3_date, recent_3_track_name, recent_3_track_status, recent_3_jockey, recent_3_class,
  recent_4_place, recent_4_total_runners, recent_4_time, recent_4_distance_m,
  recent_4_weight_kg, recent_4_barrier, recent_4_track_status_value, recent_4_starting_price,
  recent_4_margin, recent_4_date, recent_4_track_name, recent_4_track_status, recent_4_jockey, recent_4_class,
  recent_5_place, recent_5_total_runners, recent_5_time, recent_5_distance_m,
  recent_5_weight_kg, recent_5_barrier, recent_5_track_status_value, recent_5_starting_price,
  recent_5_margin, recent_5_date, recent_5_track_name, recent_5_track_status, recent_5_jockey, recent_5_class,
  recent_6_place, recent_6_total_runners, recent_6_time, recent_6_distance_m,
  recent_6_weight_kg, recent_6_barrier, recent_6_track_status_value, recent_6_starting_price,
  recent_6_margin, recent_6_date, recent_6_track_name, recent_6_track_status, recent_6_jockey, recent_6_class,
  open_price, fluc1, fluc2,
  sp_starting_price,
  COALESCE(NULLIF(fluc2, 0), NULLIF(fluc1, 0), NULLIF(open_price, 0)) AS live_price,
  CASE
    WHEN fluc2 IS NOT NULL AND fluc2 > 0 THEN 'FLUC2'
    WHEN fluc1 IS NOT NULL AND fluc1 > 0 THEN 'FLUC1'
    WHEN open_price IS NOT NULL AND open_price > 0 THEN 'OPEN_ONLY'
    ELSE 'NO_PRICE'
  END AS price_quality,
  finish_place, result_code, status, is_winner, top3_mask
FROM race_runners
"""

_BACKTEST_WHERE = """
WHERE status = 'finished'
  AND result_code IN ('W', 'P', 'L') and race_number >= 6
ORDER BY start_time_iso, race_id, runner_number
"""

_LIVE_WHERE = """
WHERE status = 'no_result'
  AND result_code != 'V'
  AND (source_betting_status IS NULL OR source_betting_status <> 'RESULTED')
  AND COALESCE(NULLIF(fluc2, 0), NULLIF(fluc1, 0), NULLIF(open_price, 0)) IS NOT NULL
ORDER BY start_time_iso, race_id, runner_number
"""

_SINGLE_RACE_SQL = _BASE_SELECT + """
WHERE race_id = ?
  AND result_code != 'V'
ORDER BY runner_number
"""


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_SELECT_COLUMNS)


def _query_for_mode(query_mode: str) -> str:
    if query_mode == "backtest":
        return _BASE_SELECT + _BACKTEST_WHERE
    if query_mode == "live":
        return _BASE_SELECT + _LIVE_WHERE
    raise ValueError(f"Invalid query_mode: {query_mode}")


def load_race_runners(db_path: str, query_mode: str, config: dict) -> pd.DataFrame:
    """Load canonical runner rows for backtest or live mode."""
    sql = _query_for_mode(query_mode)
    del config
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn)
    return df if not df.empty else _empty_frame()


def load_backtest_data(db_path: str) -> pd.DataFrame:
    """Load all resulted runner rows for backtest analysis."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(_BASE_SELECT + _BACKTEST_WHERE, conn)
    return df if not df.empty else _empty_frame()


def get_race_ids(db_path: str, query_mode: str) -> list:
    """Return ordered race IDs for the requested query mode."""
    if query_mode == "backtest":
        where_clause = """
        WHERE status = 'finished'
          AND result_code IN ('W', 'P', 'L')  and race_number >= 6
        """
    elif query_mode == "live":
        where_clause = """
        WHERE status = 'no_result'
          AND result_code != 'V'
          AND (source_betting_status IS NULL OR source_betting_status <> 'RESULTED')
          AND COALESCE(NULLIF(fluc2, 0), NULLIF(fluc1, 0), NULLIF(open_price, 0)) IS NOT NULL
        """
    else:
        raise ValueError(f"Invalid query_mode: {query_mode}")

    sql = f"""
    SELECT race_id
    FROM race_runners
    {where_clause}
    GROUP BY race_id
    ORDER BY MIN(start_time_iso), race_id
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(sql).fetchall()
    return [row[0] for row in rows]


def load_single_race(db_path: str, race_id: str) -> pd.DataFrame:
    """Load one race with canonical derived price fields."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(_SINGLE_RACE_SQL, conn, params=[race_id])
    return df if not df.empty else _empty_frame()


def load_draw_bias_table(conn) -> pd.DataFrame:
    """Load historical draw/barrier win and place rates by track × distance × condition.

    Returns a DataFrame with columns:
      track_name, distance_m, track_condition, draw_number,
      starts, wins, places, win_rate_pct, place_rate_pct
    Only rows with starts >= 5 are included.
    """
    query = """
    SELECT
      COALESCE(NULLIF(TRIM(competition_name), ''), '(blank)') AS track_name,
      distance_m,
      COALESCE(NULLIF(TRIM(track_status), ''), '(blank)') AS track_condition,
      draw_number,
      COUNT(*) AS starts,
      SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) AS wins,
      SUM(CASE WHEN top3_mask = 1 THEN 1 ELSE 0 END) AS places,
      ROUND(100.0 * AVG(CASE WHEN is_winner = 1 THEN 1.0 ELSE 0 END), 2) AS win_rate_pct,
      ROUND(100.0 * AVG(CASE WHEN top3_mask = 1 THEN 1.0 ELSE 0 END), 2) AS place_rate_pct
    FROM race_runners
    WHERE runner_mask = 1
      AND draw_number IS NOT NULL
      AND distance_m IS NOT NULL
    GROUP BY track_name, distance_m, track_condition, draw_number
    HAVING starts >= 1
    """
    return pd.read_sql_query(query, conn)


def load_jockey_stats_table(conn) -> pd.DataFrame:
    """Load historical jockey win and place rates across all rides.

    Returns a DataFrame with columns:
      jockey_name, starts, runs, wins, places, win_rate_pct, place_rate_pct
    """
    query = """
    SELECT
      COALESCE(NULLIF(TRIM(jockey), ''), '(blank)') AS jockey_name,
      COUNT(*) AS starts,
      COUNT(*) AS runs,
      SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) AS wins,
      SUM(CASE WHEN top3_mask = 1 THEN 1 ELSE 0 END) AS places,
      ROUND(100.0 * AVG(CASE WHEN is_winner = 1 THEN 1.0 ELSE 0 END), 2) AS win_rate_pct,
      ROUND(100.0 * AVG(CASE WHEN top3_mask = 1 THEN 1.0 ELSE 0 END), 2) AS place_rate_pct
    FROM race_runners
    WHERE runner_mask = 1
    GROUP BY jockey_name
    """
    return pd.read_sql_query(query, conn)



def load_trainer_stats_table(conn) -> pd.DataFrame:
    """Load historical trainer win and place rates across all runners.

    Returns a DataFrame with columns:
      trainer_name, starts, runs, wins, places, win_rate_pct, place_rate_pct
    """
    query = """
    SELECT
      COALESCE(NULLIF(TRIM(trainer), ''), '(blank)') AS trainer_name,
      COUNT(*) AS starts,
      COUNT(*) AS runs,
      SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) AS wins,
      SUM(CASE WHEN top3_mask = 1 THEN 1 ELSE 0 END) AS places,
      ROUND(100.0 * AVG(CASE WHEN is_winner = 1 THEN 1.0 ELSE 0 END), 2) AS win_rate_pct,
      ROUND(100.0 * AVG(CASE WHEN top3_mask = 1 THEN 1.0 ELSE 0 END), 2) AS place_rate_pct
    FROM race_runners
    WHERE runner_mask = 1
    GROUP BY trainer_name
    """
    return pd.read_sql_query(query, conn)
