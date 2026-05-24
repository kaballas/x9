"""Tests for edge calculation."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from betting.config import CONFIG
from betting.edge import calculate_edges, compute_edge


@pytest.fixture
def base_config():
    return CONFIG.copy()


def make_edge_df(model_probs, market_probs, live_prices, sp_prices=None):
    size = len(model_probs)
    return pd.DataFrame(
        {
            "race_id": ["R1"] * size,
            "selection_id": list(range(size)),
            "model_prob": model_probs,
            "market_implied_prob": market_probs,
            "live_price": live_prices,
            "sp_starting_price": sp_prices if sp_prices is not None else [None] * size,
        }
    )


def test_positive_edge_when_model_exceeds_market(base_config):
    df = make_edge_df([0.4], [0.25], [4.0])
    result = calculate_edges(df, base_config)
    assert result["edge"].iloc[0] > 0


def test_negative_edge_when_market_exceeds_model(base_config):
    df = make_edge_df([0.2], [0.5], [2.0])
    result = calculate_edges(df, base_config)
    assert result["edge"].iloc[0] < 0


def test_zero_edge_when_equal(base_config):
    df = make_edge_df([0.25], [0.25], [4.0])
    result = calculate_edges(df, base_config)
    assert abs(result["edge"].iloc[0]) < 1e-9


def test_edge_pct_is_100x_edge(base_config):
    df = make_edge_df([0.40], [0.25], [4.0])
    result = calculate_edges(df, base_config)
    assert abs(result["edge_pct"].iloc[0] - result["edge"].iloc[0] * 100) < 1e-9


def test_compute_edge_scalar():
    assert abs(compute_edge(0.4, 0.25) - 0.15) < 1e-9
    assert abs(compute_edge(0.1, 0.5) + 0.4) < 1e-9
    assert compute_edge(0.3, 0.3) == 0.0


def test_edge_column_present(base_config):
    df = make_edge_df([0.3, 0.2], [0.25, 0.25], [4.0, 4.0])
    result = calculate_edges(df, base_config)
    assert "edge" in result.columns
    assert "edge_pct" in result.columns
    assert "raw_edge" in result.columns
    assert "fair_edge" in result.columns
    assert "ev" in result.columns


def test_sp_not_in_edge_calculation(base_config):
    df_one = make_edge_df([0.4], [0.25], [4.0], sp_prices=[3.0])
    df_two = make_edge_df([0.4], [0.25], [4.0], sp_prices=[99.0])
    result_one = calculate_edges(df_one, base_config)
    result_two = calculate_edges(df_two, base_config)
    assert abs(result_one["edge"].iloc[0] - result_two["edge"].iloc[0]) < 1e-9


def test_price_vs_sp_is_reference_only(base_config):
    df = make_edge_df([0.4], [0.25], [4.0], sp_prices=[3.5])
    result = calculate_edges(df, base_config)
    assert "price_vs_sp" in result.columns
    assert pd.notna(result["price_vs_sp"].iloc[0])
    assert result["edge"].iloc[0] > 0


def test_raw_edge_uses_reciprocal_price(base_config):
    df = make_edge_df([0.30], [0.25], [4.0], sp_prices=[3.5])
    result = calculate_edges(df, base_config)
    assert abs(result["raw_edge"].iloc[0] - 0.05) < 1e-9


def test_ev_matches_prob_times_price_minus_one(base_config):
    df = make_edge_df([0.30], [0.25], [4.0], sp_prices=[3.5])
    result = calculate_edges(df, base_config)
    assert abs(result["ev"].iloc[0] - 0.20) < 1e-9


def test_price_vs_sp_null_when_sp_null(base_config):
    df = make_edge_df([0.4], [0.25], [4.0], sp_prices=[None])
    result = calculate_edges(df, base_config)
    assert pd.isna(result["price_vs_sp"].iloc[0])
