from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from betting.config import CONFIG
from betting.meta_model import add_meta_model_signal


def make_row(race_id: int, runner_number: int, is_winner: int, strength: float) -> dict:
    return {
        "race_id": race_id,
        "runner_number": runner_number,
        "is_winner": is_winner,
        "speed_feature_score": strength,
        "speed_consistency": strength,
        "recent_form_score": strength,
        "suitability_score": strength,
        "connection_score": strength,
        "market_movement_score": strength,
        "margin_score": strength,
        "freshness_score": strength,
        "class_score": strength,
        "draw_bias_score": strength,
        "jockey_score": strength,
        "trainer_score": strength,
        "live_price": 6.0 - strength,
        "market_rank": 1 if is_winner else 4,
        "recent_sp_score": strength,
        "class_movement_score": strength,
        "distance_change_score": strength,
        "weight_trend_score": strength,
        "barrier_transition_score": strength,
        "pedigree_score": strength,
        "form_string_score": strength,
        "jockey_continuity_score": strength,
        "travel_score": strength,
        "equipment_score": strength,
        "active_field_size": 8,
    }


def test_add_meta_model_signal_prefers_stronger_runner():
    rows = []
    for race_id in range(1, 11):
        rows.append(make_row(race_id, 1, 1, 0.95))
        rows.append(make_row(race_id, 2, 0, 0.25))
        rows.append(make_row(race_id, 3, 0, 0.15))

    result = add_meta_model_signal(pd.DataFrame(rows), CONFIG.copy())

    assert "meta_model_score" in result.columns
    winners = result.loc[result["is_winner"] == 1, "meta_model_score"]
    losers = result.loc[result["is_winner"] == 0, "meta_model_score"]
    assert winners.mean() > losers.mean()
