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
    "min_price": 3.000,  # Minimum allowed live_price for candidate bets.
    "max_price": 15.000,  # Maximum allowed live_price for candidate bets.
    "min_field_size": 7,  # Minimum active field size allowed for version 1.
    "max_field_size": 12,  # Maximum active field size allowed for version 1.
    "max_model_rank": 1,  # Highest model rank allowed to remain bet-eligible.
    "min_edge": 0.000,  # Minimum fair edge required to keep a candidate.
    "min_raw_edge": 0.020,  # Minimum raw edge (model_prob - 1/price) required for a candidate.
    "min_ev": 0.050,  # Minimum expected value (model_prob * price - 1) required for a candidate.
    "min_model_probability": 0.080,  # Minimum model win probability required for a bet candidate.
    "min_score_gap_to_next": 0.010,  # Require top selections to clear a minimum model-score gap over the next runner.
    "min_model_vs_market_ratio": 0.000,  # Optional: require model_prob >= raw_market_prob * ratio when > 0.
    "stake": 1.0,  # Flat version-1 stake applied to every surviving runner.
    "staking_mode": "flat",  # Staking mode: flat | fibonacci
    "fib_variant": "two_back",  # Fibonacci variant: two_back | one_back | reset
    "fib_base_unit": 1.0,  # Base unit stake for Fibonacci (level 0 = 1 unit)
    "fib_max_level": 10,  # Maximum Fibonacci level (10 = 144 units cap)
    "allow_multiple_bets_per_race": True,  # Whether multiple runners from one race may survive filtering.
    "exclude_runner_if_no_live_price": True,  # Drop runners whose live_price is missing or invalid.
    "exclude_race_if_price_coverage_below": 0.80,  # Drop races with insufficient live-price coverage.
    "min_recent_form_count": 2,  # Minimum recent-runs sample before form is treated as usable.
    "min_races_for_bucket": 3,  # Reserved minimum sample size for future calibration buckets.
    "calibration_model_path": None,  # Path to fitted calibration model (.pkl). None = auto-detect beside calibration.py.
    "calibration_raw_blend": 0.20,  # Blend share of raw_model_prob back into calibrated output to reduce isotonic plateaus.
    "market_confirmation_top_rank": 3,  # Top market ranks eligible for a live-market confirmation uplift.
    "market_confirmation_min_steam_score": 0.95,  # Require an extreme late steam score before overriding the model.
    "market_confirmation_min_fair_market_prob": 0.15,  # Only intervene for runners the fair market already rates seriously.
    "market_confirmation_min_prob_gap": 0.08,  # Require a large model-vs-market under-rating before intervening.
    "market_confirmation_prob_floor": 0.12,  # Minimum post-calibration probability floor for qualifying runners.
    "market_confirmation_fair_share": 1.00,  # Target share of fair_market_prob when confirmation triggers.
    "market_confirmation_prob_cap": 0.20,  # Hard cap so the uplift cannot dominate the whole race.
    "weight_speed_rating": 0.294,      # Reduced from 0.276
    "weight_recent_form": 0.202,       # Reduced from 0.230
    "weight_suitability": 0.110,       # Slightly reduced from 0.110
    "weight_connections": 0.046,       # Slightly reduced from 0.037
    "weight_market_sanity": 0.009,     # Increased from 0.009
    "weight_steam": 0.080,             # Increased from 0.080
    "weight_margin": 0.110,            # Reduced from 0.110
    "weight_freshness": 0.009,         # Slightly increased from 0.009
    "weight_class": 0.009,             # Slightly increased from 0.018
    "weight_draw_bias": 0.046,         # Slightly reduced from 0.046
    "weight_jockey": 0.055,            # Slightly reduced from 0.046
    "weight_trainer": 0.030,           # Increased from 0.028
    "prob_temperature": 6.0,  # Sharpening factor for within-race softmax (higher = more separation)
}
