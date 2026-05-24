"""Tests for probability assignment: softmax and market-implied."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from betting.config import CONFIG
from betting.probabilities import (
    assign_probabilities,
    compute_market_implied_prob,
    normalise_market_prob,
    softmax_within_race,
)


@pytest.fixture
def base_config():
    return CONFIG.copy()


@pytest.fixture
def sample_score_race():
    return pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R1"],
            "model_score": [3.0, 2.0, 1.0],
            "live_price": [2.5, 4.0, 8.0],
        }
    )


def make_race_df(scores, prices):
    return pd.DataFrame(
        {
            "race_id": ["R1"] * len(scores),
            "model_score": scores,
            "live_price": prices,
        }
    )


def test_softmax_sums_to_one():
    scores = pd.Series([2.0, 1.5, 0.8, 0.3])
    probs = softmax_within_race(scores)
    assert abs(probs.sum() - 1.0) < 1e-9


def test_softmax_highest_score_gets_highest_prob():
    scores = pd.Series([3.0, 1.0, 0.5])
    probs = softmax_within_race(scores)
    assert probs.iloc[0] == probs.max()


def test_softmax_equal_scores_give_equal_probs():
    scores = pd.Series([1.0, 1.0, 1.0])
    probs = softmax_within_race(scores)
    assert abs(probs.std()) < 1e-9


def test_softmax_numerically_stable_large_scores():
    scores = pd.Series([1000.0, 999.0, 998.0])
    probs = softmax_within_race(scores)
    assert abs(probs.sum() - 1.0) < 1e-9
    assert all(probs >= 0)


def test_softmax_temperature_increases_separation():
    scores = pd.Series([0.620, 0.558, 0.463, 0.237])
    cool = softmax_within_race(scores, temperature=1.0)
    warm = softmax_within_race(scores, temperature=4.0)
    assert warm.iloc[0] > cool.iloc[0]
    assert warm.iloc[-1] < cool.iloc[-1]


def test_softmax_single_runner():
    scores = pd.Series([5.0])
    probs = softmax_within_race(scores)
    assert abs(probs.iloc[0] - 1.0) < 1e-9


def test_market_implied_prob_reciprocal():
    prices = pd.Series([2.0, 4.0, 5.0, 10.0])
    probs = compute_market_implied_prob(prices)
    expected = pd.Series([0.5, 0.25, 0.2, 0.1])
    pd.testing.assert_series_equal(probs, expected, check_names=False)


def test_market_implied_prob_null_price_gives_zero():
    prices = pd.Series([4.0, None, 0.0, -1.0])
    probs = compute_market_implied_prob(prices)
    assert probs.iloc[1] == 0.0
    assert probs.iloc[2] == 0.0
    assert probs.iloc[3] == 0.0


def test_normalise_market_prob_removes_overround():
    prices = pd.Series([4.0, 5.0])
    probs = normalise_market_prob(prices)
    expected = pd.Series([5.0 / 9.0, 4.0 / 9.0])
    pd.testing.assert_series_equal(probs, expected, check_names=False)
    assert abs(probs.sum() - 1.0) < 1e-9


def test_assign_probabilities_adds_columns(sample_score_race, base_config):
    result = assign_probabilities(sample_score_race, base_config)
    assert "raw_model_prob" in result.columns
    assert "model_prob" in result.columns
    assert "raw_market_prob" in result.columns
    assert "fair_market_prob" in result.columns
    assert "market_implied_prob" in result.columns


def test_assign_probabilities_sums_to_one_per_race(sample_score_race, base_config):
    result = assign_probabilities(sample_score_race, base_config)
    total = result.groupby("race_id")["raw_model_prob"].sum().iloc[0]
    assert abs(total - 1.0) < 1e-9


def test_assign_probabilities_market_prob_from_live_price(base_config):
    df = make_race_df([3.0, 2.0], [4.0, 5.0])
    result = assign_probabilities(df, base_config)
    assert abs(result.iloc[0]["raw_market_prob"] - 0.25) < 1e-9
    assert abs(result.iloc[1]["raw_market_prob"] - 0.20) < 1e-9
    assert abs(result.iloc[0]["fair_market_prob"] - (5.0 / 9.0)) < 1e-9
    assert abs(result.iloc[1]["fair_market_prob"] - (4.0 / 9.0)) < 1e-9
    assert abs(result.iloc[0]["market_implied_prob"] - (5.0 / 9.0)) < 1e-9
    assert abs(result.iloc[1]["market_implied_prob"] - (4.0 / 9.0)) < 1e-9


def test_assign_probabilities_uses_configured_temperature(base_config):
    df = make_race_df([0.620, 0.237], [2.9, 27.0])
    result = assign_probabilities(df, base_config)
    assert result.iloc[0]["raw_model_prob"] > 0.8
    assert result.iloc[1]["raw_model_prob"] < 0.2
