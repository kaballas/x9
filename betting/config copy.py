"""Static configuration for the betting framework."""

CONFIG = {
    "database_path": "database/race_reports.sqlite",  # Relative path to the SQLite database.
    "race_id_col": "race_id",  # Column holding the unique race identifier.
    "runner_id_col": "selection_id",  # Column holding the unique runner identifier.
    "result_col": "finish_place",  # Column containing the recorded finishing position.
    "winner_col": "is_winner",  # Column marking whether the runner won the race.
    "live_price_columns": ["fluc2", "fluc1", "open_price"],  # Ordered fallback inputs for live_price.
    "live_price_column": "live_price",  # Derived pre-race price column used for selection.
    "settlement_price_column": "live_price",  # Price column used to settle backtest bets.
    "sp_reference_column": "sp_starting_price",  # SP column kept for diagnostics only.
    "backtest_price_mode": "latest_pre_race",  # Documents that backtests use the latest pre-race price.
    "min_price": 2.000,  # Minimum allowed live_price for candidate bets.
    "max_price": 10.000,  # Maximum allowed live_price for candidate bets.
    "min_field_size": 7,  # Minimum active field size allowed for version 1.
    "max_field_size": 12,  # Maximum active field size allowed for version 1.
    "max_model_rank": 3,  # Highest model rank allowed to remain bet-eligible.
    "min_edge": 0.070,  # Minimum model edge required to keep a candidate.
    "stake": 1.0,  # Flat version-1 stake applied to every surviving runner.
    "allow_multiple_bets_per_race": False,  # Whether multiple runners from one race may survive filtering.
    "exclude_runner_if_no_live_price": True,  # Drop runners whose live_price is missing or invalid.
    "exclude_race_if_price_coverage_below": 0.80,  # Drop races with insufficient live-price coverage.
    "min_recent_form_count": 2,  # Minimum recent-runs sample before form is treated as usable.
    "min_races_for_bucket": 30,  # Reserved minimum sample size for future calibration buckets.
    "calibration_model_path": None,  # Path to fitted calibration model (.pkl). None = auto-detect beside calibration.py.
    "weight_speed_rating": 0.0280,  # Weight for the condition-aware speed rating component.
    "weight_recent_form": 0.200,  # Weight for the recent run form component.
    "weight_suitability": 0.180,  # Weight for the distance/track/condition fit component.
    "weight_connections": 0.070,  # Weight for the horse-jockey pairing history component.
    "weight_market_sanity": 0.038,  # Inverse-price tie-breaker weight.
    "weight_margin": 0.100,  # Weight for recent winning/beaten margins.
    "weight_freshness": 0.070,  # Weight for fitness and freshness timing.
    "weight_class": 0.050,  # Weight for career class quality.
    "prob_temperature": 4.0,  # Sharpening factor for within-race softmax (higher = more separation)
}
