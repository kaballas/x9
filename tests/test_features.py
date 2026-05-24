"""Tests for feature engineering helpers."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from betting.features import (
    _position_quality_scalar,
    _smooth_rate,
    add_connection_score,
    add_draw_bias_score,
    add_jockey_score,
    add_recent_form_score,
    add_recent_place_quality_score,
    add_suitability_score,
    add_trainer_score,
)


def test_smooth_rate_returns_prior_for_unknown_history():
    result = _smooth_rate(pd.Series([0]), pd.Series([0]), prior=0.25, prior_starts=4)
    assert result.tolist() == pytest.approx([0.250])


def test_smooth_rate_keeps_bad_combo_above_hard_zero():
    result = _smooth_rate(pd.Series([0]), pd.Series([5]), prior=0.25, prior_starts=4)
    assert result.tolist() == pytest.approx([1 / 9])


def test_smooth_rate_blends_sparse_winner_history():
    result = _smooth_rate(pd.Series([1]), pd.Series([7]), prior=0.25, prior_starts=4)
    assert result.tolist() == pytest.approx([2 / 11])


def test_connection_score_uses_neutral_prior_for_unknown_combo():
    df = pd.DataFrame({"horse_jockey_wins": [0], "horse_jockey_starts": [0]})
    result = add_connection_score(df)
    assert result["connection_score"].tolist() == pytest.approx([0.15])


def test_suitability_score_uses_prior_for_unknown_distance_and_track():
    df = pd.DataFrame(
        {
            "track_status": ["Good 4"],
            "distance_wins": [0],
            "distance_starts": [0],
            "track_wins": [0],
            "track_starts": [0],
            "good_wins": [0],
            "good_starts": [1],
            "soft_wins": [0],
            "soft_starts": [0],
            "heavy_wins": [0],
            "heavy_starts": [0],
        }
    )
    result = add_suitability_score(df)
    assert result["suitability_score"].tolist() == pytest.approx([0.20])


# ── _position_quality_scalar ────────────────────────────────────────────────

def test_position_quality_winner_scores_one():
    assert _position_quality_scalar(1, 10) == pytest.approx(1.0)


def test_position_quality_last_scores_zero():
    # last in 10-runner field: (10-10)/(10-1) = 0.0
    assert _position_quality_scalar(10, 10) == pytest.approx(0.0)


def test_position_quality_mid_field():
    # 5th in 10-runner field: (10-5)/(10-1) = 5/9
    assert _position_quality_scalar(5, 10) == pytest.approx(5 / 9)


def test_position_quality_no_runners_falls_back_to_cap():
    # place 1 with no field info → (12-1)/(12-1) = 1.0
    assert _position_quality_scalar(1, None) == pytest.approx(1.0)


def test_position_quality_missing_place_returns_prior():
    assert _position_quality_scalar(None, 10) == pytest.approx(0.30)


def test_position_quality_place_beyond_field_clips_to_zero():
    # place 15 in 10-runner field → clipped to 0
    assert _position_quality_scalar(15, 10) == pytest.approx(0.0)


# ── add_recent_place_quality_score ──────────────────────────────────────────

def _make_runner(**kwargs):
    """Build a minimal one-row DataFrame for feature tests."""
    base = {f"recent_{i}_place": None for i in range(1, 7)}
    base.update({f"recent_{i}_total_runners": None for i in range(1, 7)})
    base.update(kwargs)
    return pd.DataFrame([base])


def test_place_quality_all_wins_scores_one():
    df = _make_runner(
        **{f"recent_{i}_place": 1 for i in range(1, 7)},
        **{f"recent_{i}_total_runners": 10 for i in range(1, 7)},
    )
    result = add_recent_place_quality_score(df)
    assert result["recent_place_quality_score"].iloc[0] == pytest.approx(1.0)


def test_place_quality_no_history_scores_neutral_prior():
    df = _make_runner()  # all None
    result = add_recent_place_quality_score(df)
    assert result["recent_place_quality_score"].iloc[0] == pytest.approx(0.30)


def test_place_quality_recent_runs_weighted_more_than_older():
    """Runner A won last run only; Runner B won run-2 only (all other slots = prior).
    Runner A should score higher because r1 has weight 0.35 vs r2's 0.25."""
    n = 10
    df_a = _make_runner(recent_1_place=1, recent_1_total_runners=n)
    df_b = _make_runner(recent_2_place=1, recent_2_total_runners=n)
    score_a = add_recent_place_quality_score(df_a)["recent_place_quality_score"].iloc[0]
    score_b = add_recent_place_quality_score(df_b)["recent_place_quality_score"].iloc[0]
    assert score_a > score_b


def test_place_quality_damps_low_class_results_vs_today():
    """Same placing in weaker class should count less for form quality."""
    n = 10
    today = {"class_name": "Bm74"}
    df_same_class = _make_runner(
        **today,
        recent_1_place=1,
        recent_1_total_runners=n,
        recent_1_class="Bm74",
    )
    df_lower_class = _make_runner(
        **today,
        recent_1_place=1,
        recent_1_total_runners=n,
        recent_1_class="Mdn",
    )
    score_same = add_recent_place_quality_score(df_same_class)["recent_place_quality_score"].iloc[0]
    score_lower = add_recent_place_quality_score(df_lower_class)["recent_place_quality_score"].iloc[0]
    assert score_same > score_lower


# ── add_recent_form_score (blended) ─────────────────────────────────────────

def _make_form_runner(**kwargs):
    base = {
        "recent_win_rate_5": 0.0,
        "recent_top3_rate_5": 0.0,
        "recent_avg_place": 10.0,
        "recent_avg_place_3": 10.0,
        **{f"recent_{i}_place": None for i in range(1, 7)},
        **{f"recent_{i}_total_runners": None for i in range(1, 7)},
    }
    base.update(kwargs)
    return pd.DataFrame([base])


def test_form_score_is_blend_of_agg_and_position_quality():
    df = _make_form_runner(
        recent_win_rate_5=1.0,
        recent_top3_rate_5=1.0,
        recent_avg_place=1.0,
        recent_avg_place_3=1.0,
        **{f"recent_{i}_place": 1 for i in range(1, 7)},
        **{f"recent_{i}_total_runners": 10 for i in range(1, 7)},
    )
    result = add_recent_form_score(df)
    # Both agg and pos quality approach max → blended ≈ 0.925
    assert result["recent_form_score"].iloc[0] == pytest.approx(0.925, abs=0.05)


def test_form_score_no_history_returns_low_but_nonzero():
    df = _make_form_runner()  # all zeros/None
    result = add_recent_form_score(df)
    score = result["recent_form_score"].iloc[0]
    # agg = 0.0 (zeros), pos_quality = 0.30 (prior) → blended = 0.15
    assert 0.10 <= score <= 0.20


# ── add_draw_bias_score ───────────────────────────────────────────────────────

def _make_draw_runner(**kwargs):
    base = {
        "competition_name": "TestTrack",
        "distance_m": 1200,
        "track_status": "Good",
        "draw_number": 3,
    }
    base.update(kwargs)
    return pd.DataFrame([base])


def test_draw_bias_score_no_lookup():
    """When draw_bias_df is None, score defaults to neutral prior 0.25."""
    result = add_draw_bias_score(_make_draw_runner())
    assert result["draw_bias_score"].tolist() == pytest.approx([0.25])
    assert result["draw_bias_starts"].tolist() == [0]
    assert pd.isna(result["draw_bias_win_rate"].iloc[0])


def test_draw_bias_score_match():
    """Matching draw gets smooth_rate applied."""
    lookup = pd.DataFrame(
        [
            {
                "track_name": "TestTrack",
                "distance_m": 1200,
                "track_condition": "Good",
                "draw_number": 3,
                "starts": 10,
                "wins": 5,
                "places": 7,
                "win_rate_pct": 50.0,
                "place_rate_pct": 70.0,
            }
        ]
    )
    result = add_draw_bias_score(_make_draw_runner(), lookup)
    # prior=0.25, prior_starts=20 -> (5 + 5) / (10 + 20) = 1/3
    assert result["draw_bias_score"].tolist() == pytest.approx([10.0 / 30.0])
    assert result["draw_bias_starts"].tolist() == pytest.approx([10.0])
    assert result["draw_bias_win_rate"].tolist() == pytest.approx([50.0])


def test_draw_bias_score_no_match():
    """Runner whose draw is not in the lookup gets the prior 0.25."""
    lookup = pd.DataFrame(
        [
            {
                "track_name": "TestTrack",
                "distance_m": 1200,
                "track_condition": "Good",
                "draw_number": 2,
                "starts": 10,
                "wins": 5,
                "places": 7,
                "win_rate_pct": 50.0,
                "place_rate_pct": 70.0,
            }
        ]
    )
    result = add_draw_bias_score(_make_draw_runner(), lookup)
    assert result["draw_bias_score"].tolist() == pytest.approx([0.25])
    assert result["draw_bias_starts"].tolist() == pytest.approx([0.0])
    assert pd.isna(result["draw_bias_win_rate"].iloc[0])


def test_draw_bias_score_zero_starts_in_lookup():
    """Rows not meeting HAVING starts>=5 won't be in the table → prior."""
    result = add_draw_bias_score(_make_draw_runner(), pd.DataFrame())
    assert result["draw_bias_score"].tolist() == pytest.approx([0.25])
    assert result["draw_bias_starts"].tolist() == [0]
    assert pd.isna(result["draw_bias_win_rate"].iloc[0])


# ── add_jockey_score ─────────────────────────────────────────────────────────


def _make_jockey_runner(**kwargs):
    base = {"jockey": "Test Jockey"}
    base.update(kwargs)
    return pd.DataFrame([base])


def test_jockey_score_no_lookup():
    result = add_jockey_score(_make_jockey_runner())
    assert result["jockey_score"].tolist() == pytest.approx([0.15])
    assert result["jockey_starts"].tolist() == [0]
    assert pd.isna(result["jockey_win_rate"].iloc[0])


def test_jockey_score_match():
    lookup = pd.DataFrame(
        [{"jockey_name": "Test Jockey", "wins": 20, "starts": 100, "win_rate_pct": 20.0}]
    )
    result = add_jockey_score(_make_jockey_runner(), lookup)
    assert result["jockey_score"].tolist() == pytest.approx([21.2 / 108.0])
    assert result["jockey_starts"].tolist() == pytest.approx([100.0])
    assert result["jockey_win_rate"].tolist() == pytest.approx([20.0])


def test_jockey_score_no_match():
    lookup = pd.DataFrame(
        [{"jockey_name": "Other Jockey", "wins": 20, "starts": 100, "win_rate_pct": 20.0}]
    )
    result = add_jockey_score(_make_jockey_runner(), lookup)
    assert result["jockey_score"].tolist() == pytest.approx([0.15])
    assert result["jockey_starts"].tolist() == pytest.approx([0.0])
    assert pd.isna(result["jockey_win_rate"].iloc[0])


def test_jockey_score_blank_name():
    lookup = pd.DataFrame(
        [{"jockey_name": "(blank)", "wins": 3, "starts": 10, "win_rate_pct": 30.0}]
    )
    result_blank = add_jockey_score(_make_jockey_runner(jockey="  "), lookup)
    result_none = add_jockey_score(_make_jockey_runner(jockey=None), lookup)
    expected_score = (3 + 0.15 * 8) / (10 + 8)
    assert result_blank["jockey_score"].tolist() == pytest.approx([expected_score])
    assert result_none["jockey_score"].tolist() == pytest.approx([expected_score])
    assert result_blank["jockey_win_rate"].tolist() == pytest.approx([30.0])
    assert result_none["jockey_win_rate"].tolist() == pytest.approx([30.0])


# ── add_trainer_score ────────────────────────────────────────────────────────


def _make_trainer_runner(**kwargs):
    base = {"trainer": "Test Trainer"}
    base.update(kwargs)
    return pd.DataFrame([base])


def test_trainer_score_no_lookup():
    result = add_trainer_score(_make_trainer_runner())
    assert result["trainer_score"].tolist() == pytest.approx([0.15])
    assert result["trainer_starts"].tolist() == [0]
    assert pd.isna(result["trainer_win_rate"].iloc[0])


def test_trainer_score_match():
    lookup = pd.DataFrame(
        [{"trainer_name": "Test Trainer", "wins": 20, "starts": 100, "win_rate_pct": 20.0}]
    )
    result = add_trainer_score(_make_trainer_runner(), lookup)
    assert result["trainer_score"].tolist() == pytest.approx([21.2 / 108.0])
    assert result["trainer_starts"].tolist() == pytest.approx([100.0])
    assert result["trainer_win_rate"].tolist() == pytest.approx([20.0])


def test_trainer_score_no_match():
    lookup = pd.DataFrame(
        [{"trainer_name": "Other Trainer", "wins": 20, "starts": 100, "win_rate_pct": 20.0}]
    )
    result = add_trainer_score(_make_trainer_runner(), lookup)
    assert result["trainer_score"].tolist() == pytest.approx([0.15])
    assert result["trainer_starts"].tolist() == pytest.approx([0.0])
    assert pd.isna(result["trainer_win_rate"].iloc[0])


def test_trainer_score_blank_name():
    lookup = pd.DataFrame(
        [{"trainer_name": "(blank)", "wins": 3, "starts": 10, "win_rate_pct": 30.0}]
    )
    result_blank = add_trainer_score(_make_trainer_runner(trainer="  "), lookup)
    result_none = add_trainer_score(_make_trainer_runner(trainer=None), lookup)
    expected_score = (3 + 0.15 * 8) / (10 + 8)
    assert result_blank["trainer_score"].tolist() == pytest.approx([expected_score])
    assert result_none["trainer_score"].tolist() == pytest.approx([expected_score])
    assert result_blank["trainer_win_rate"].tolist() == pytest.approx([30.0])
    assert result_none["trainer_win_rate"].tolist() == pytest.approx([30.0])
