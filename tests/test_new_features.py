"""Tests for the new margin, freshness, and class features."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from betting.config import CONFIG
from betting.features import (
    add_class_score,
    add_freshness_score,
    add_historical_speed_score,
    add_margin_score,
    build_features,
)
from betting.filters import candidate_mask
from betting.scoring import compute_model_score


@pytest.fixture
def base_config():
    return CONFIG.copy()


def make_runner(**overrides):
    runner = {
        "race_id": "R1",
        "selection_id": 1,
        "track_status": "Good 4",
        "wet_rating": 70.0,
        "dry_rating": 75.0,
        "distance_wins": 1,
        "distance_starts": 4,
        "track_wins": 1,
        "track_starts": 4,
        "good_wins": 1,
        "good_starts": 4,
        "soft_wins": 0,
        "soft_starts": 1,
        "heavy_wins": 0,
        "heavy_starts": 1,
        "recent_win_rate_5": 0.2,
        "recent_top3_rate_5": 0.4,
        "recent_avg_place": 3.0,
        "recent_avg_place_3": 2.0,
        "horse_jockey_wins": 1,
        "horse_jockey_starts": 5,
        "recent_avg_margin_3": 0.5,
        "recent_best_margin": 1.0,
        "recent_avg_margin": 0.4,
        "recent_days_since_last_run": 21,
        "recent_1_time": "0:24.00",
        "recent_1_distance_m": 400,
        "recent_2_time": None,
        "recent_2_distance_m": None,
        "recent_3_time": None,
        "recent_3_distance_m": None,
        "recent_4_time": None,
        "recent_4_distance_m": None,
        "recent_5_time": None,
        "recent_5_distance_m": None,
        "recent_6_time": None,
        "recent_6_distance_m": None,
        "first_up_wins": 1,
        "first_up_starts": 4,
        "second_up_wins": 1,
        "second_up_starts": 4,
        "prize_money": 50000.0,
        "place_percentage": 50.0,
        "career_wins": 4,
        "career_starts": 20,
        "live_price": 4.0,
        "distance_m": 1400,
        "recent_runs_count": 3,
    }
    runner.update(overrides)
    return runner


def test_margin_score_range():
    df = pd.DataFrame(
        [
            make_runner(recent_avg_margin_3=-8.0, recent_best_margin=-5.0, recent_avg_margin=-6.0),
            make_runner(selection_id=2, recent_avg_margin_3=0.0, recent_best_margin=0.0, recent_avg_margin=0.0),
            make_runner(selection_id=3, recent_avg_margin_3=8.0, recent_best_margin=12.0, recent_avg_margin=6.0),
        ]
    )
    result = add_margin_score(df)
    assert result["margin_score"].between(0.0, 1.0).all()



def test_margin_score_winner_beats_loser():
    df = pd.DataFrame(
        [
            make_runner(recent_avg_margin_3=2.0, recent_best_margin=3.0, recent_avg_margin=1.5),
            make_runner(selection_id=2, recent_avg_margin_3=-2.0, recent_best_margin=-3.0, recent_avg_margin=-1.5),
        ]
    )
    result = add_margin_score(df)
    assert result.loc[1, "margin_score"] > result.loc[0, "margin_score"]



def test_margin_score_uses_tighter_caps_and_weights():
    df = pd.DataFrame(
        [
            make_runner(recent_avg_margin_3=1.40, recent_best_margin=0.10),
            make_runner(selection_id=2, recent_avg_margin_3=3.47, recent_best_margin=0.10),
            make_runner(selection_id=3, recent_avg_margin_3=9.47, recent_best_margin=0.50),
            make_runner(selection_id=4, recent_avg_margin_3=None, recent_best_margin=None),
        ]
    )

    result = add_margin_score(df)

    expected = [
        0.80 * (1.0 - (1.40 / 8.0)) + 0.20 * (1.0 - (0.10 / 4.0)),
        0.80 * (1.0 - (3.47 / 8.0)) + 0.20 * (1.0 - (0.10 / 4.0)),
        0.80 * 0.0 + 0.20 * (1.0 - (0.50 / 4.0)),
        0.0,
    ]

    assert result["margin_score"].tolist() == pytest.approx(expected)



def test_freshness_score_range():
    df = pd.DataFrame(
        [
            make_runner(recent_days_since_last_run=1),
            make_runner(selection_id=2, recent_days_since_last_run=21),
            make_runner(selection_id=3, recent_days_since_last_run=180),
        ]
    )
    result = add_freshness_score(df)
    assert result["freshness_score"].between(0.0, 1.0).all()



def test_freshness_peak_at_21_days():
    df = pd.DataFrame(
        [
            make_runner(recent_days_since_last_run=21),
            make_runner(selection_id=2, recent_days_since_last_run=180),
        ]
    )
    result = add_freshness_score(df)
    assert result.loc[0, "freshness_score"] > result.loc[1, "freshness_score"]



def test_class_score_range():
    df = pd.DataFrame(
        [
            make_runner(prize_money=0.0, place_percentage=0.0, career_wins=0, career_starts=1),
            make_runner(selection_id=2, prize_money=50000.0, place_percentage=50.0, career_wins=4, career_starts=20),
            make_runner(selection_id=3, prize_money=250000.0, place_percentage=80.0, career_wins=12, career_starts=20),
        ]
    )
    result = add_class_score(df)
    assert result["class_score"].between(0.0, 1.0).all()



def test_class_score_better_career():
    df = pd.DataFrame(
        [
            make_runner(prize_money=250000.0, place_percentage=75.0, career_wins=10, career_starts=20),
            make_runner(selection_id=2, prize_money=10000.0, place_percentage=25.0, career_wins=1, career_starts=20),
        ]
    )
    result = add_class_score(df)
    assert result.loc[0, "class_score"] > result.loc[1, "class_score"]



def test_build_features_has_new_columns(base_config):
    df = pd.DataFrame([make_runner(), make_runner(selection_id=2, live_price=6.0)])
    result = build_features(df, base_config)
    assert {
        "historical_speed_score",
        "speed_feature_score",
        "speed_feature_confidence",
        "speed_feature_source",
        "speed_consistency_std",
        "margin_score",
        "freshness_score",
        "class_score",
        "draw_bias_score",
        "jockey_score",
        "trainer_score",
    }.issubset(result.columns)


def test_historical_speed_score_uses_recent_run_time():
    df = pd.DataFrame(
        [
            make_runner(recent_1_time="1:24.00", recent_1_distance_m=1400),
            make_runner(selection_id=2, recent_1_time="1:36.00", recent_1_distance_m=1400),
        ]
    )
    result = add_historical_speed_score(df)
    assert result.loc[0, "historical_speed_score"] > result.loc[1, "historical_speed_score"]
    assert result.loc[0, "speed_feature_source"] == "history"


def test_speed_consistency_uses_relevant_distance_std():
    df = pd.DataFrame(
        [
            make_runner(
                distance_m=1600,
                recent_1_time="1:38.00",
                recent_1_distance_m=1600,
                recent_2_time="1:37.50",
                recent_2_distance_m=1600,
                recent_3_time="1:01.00",
                recent_3_distance_m=1000,
            ),
            make_runner(
                selection_id=2,
                distance_m=1600,
                recent_1_time="1:35.00",
                recent_1_distance_m=1600,
                recent_2_time="1:48.00",
                recent_2_distance_m=1600,
                recent_3_time="1:00.00",
                recent_3_distance_m=1000,
            ),
        ]
    )
    result = add_historical_speed_score(df)
    assert result.loc[0, "speed_consistency"] > result.loc[1, "speed_consistency"]
    assert result.loc[0, "speed_consistency"] > 0.80



def test_weight_sum():
    weight_total = sum(value for key, value in CONFIG.items() if key.startswith("weight_"))
    assert weight_total == pytest.approx(1.0)



def test_new_weights_in_config():
    assert "weight_margin" in CONFIG
    assert "weight_freshness" in CONFIG
    assert "weight_class" in CONFIG
    assert "weight_draw_bias" in CONFIG
    assert "weight_jockey" in CONFIG
    assert "weight_trainer" in CONFIG



def test_scoring_uses_new_weights(base_config):
    config = base_config.copy()
    for key in [name for name in config if name.startswith("weight_")]:
        config[key] = 0.0
    config["weight_margin"] = 0.5

    df = pd.DataFrame(
        {
            "condition_rating": [70.0, 75.0],
            "recent_form_score": [0.0, 0.0],
            "suitability_score": [0.0, 0.0],
            "connection_score": [0.0, 0.0],
            "margin_score": [0.2, 0.8],
            "freshness_score": [0.4, 0.4],
            "class_score": [0.3, 0.3],
            "live_price": [4.0, 8.0],
        }
    )

    result = compute_model_score(df, config)
    expected = pd.Series([0.1, 0.4])
    pd.testing.assert_series_equal(result.reset_index(drop=True), expected, check_names=False)


def test_scoring_uses_historical_speed_feature(base_config):
    config = base_config.copy()
    for key in [name for name in config if name.startswith("weight_")]:
        config[key] = 0.0
    config["weight_speed_rating"] = 1.0

    df = pd.DataFrame(
        {
            "race_id": ["R1", "R1"],
            "speed_feature_score": [16.2, 16.4],
            "condition_rating": [70.0, 75.0],
            "recent_form_score": [0.0, 0.0],
            "suitability_score": [0.0, 0.0],
            "connection_score": [0.0, 0.0],
            "margin_score": [0.0, 0.0],
            "freshness_score": [0.0, 0.0],
            "class_score": [0.0, 0.0],
            "live_price": [4.0, 8.0],
        }
    )

    result = compute_model_score(df, config)
    assert result.iloc[1] > result.iloc[0]
    assert (result > 0).all()


def test_candidate_mask_requires_value_overlay(base_config):
    config = base_config.copy()
    config["min_edge"] = 0.0
    config["min_raw_edge"] = 0.02
    config["min_ev"] = 0.05
    config["min_price"] = 2.5
    config["max_price"] = 12.0
    config["max_model_rank"] = 3
    config["min_model_probability"] = 0.10
    config["min_model_vs_market_ratio"] = 0.0

    df = pd.DataFrame(
        {
            "edge": [0.06, 0.06, 0.06, 0.02],
            "live_price": [5.0, 5.0, 5.0, 5.0],
            "model_rank": [2, 2, 4, 2],
            "model_prob": [0.24, 0.09, 0.18, 0.20],
            "raw_market_prob": [0.20, 0.20, 0.20, 0.20],
            "raw_edge": [0.04, -0.11, -0.02, 0.00],
            "ev": [0.20, -0.55, -0.10, 0.00],
            "market_implied_prob": [0.10, 0.08, 0.10, 0.10],
        }
    )
    mask = candidate_mask(df, config)
    assert mask.tolist() == [True, False, False, False]
