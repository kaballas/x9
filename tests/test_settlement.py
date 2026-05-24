"""Tests for flat-stake win-only settlement using live_price."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from betting.config import CONFIG
from betting.settlement import compute_roi, settle_bets


@pytest.fixture
def base_config():
    return CONFIG.copy()


def make_bet_df(live_prices, is_winners, sp_prices=None, stake=1.0):
    size = len(live_prices)
    return pd.DataFrame(
        {
            "race_id": [f"R{idx}" for idx in range(size)],
            "selection_id": list(range(size)),
            "live_price": live_prices,
            "sp_starting_price": sp_prices if sp_prices is not None else [None] * size,
            "is_winner": is_winners,
            "stake": [stake] * size,
        }
    )


def test_winner_profit_is_price_minus_one(base_config):
    df = make_bet_df([5.0], [1])
    result = settle_bets(df, base_config)
    assert abs(result["profit"].iloc[0] - 4.0) < 1e-9


def test_loser_profit_is_minus_stake(base_config):
    df = make_bet_df([5.0], [0])
    result = settle_bets(df, base_config)
    assert abs(result["profit"].iloc[0] + 1.0) < 1e-9


def test_winner_profit_scales_with_stake(base_config):
    df = make_bet_df([5.0], [1], stake=2.0)
    result = settle_bets(df, base_config)
    assert abs(result["profit"].iloc[0] - 8.0) < 1e-9


def test_loser_profit_scales_with_stake(base_config):
    df = make_bet_df([5.0], [0], stake=2.0)
    result = settle_bets(df, base_config)
    assert abs(result["profit"].iloc[0] + 2.0) < 1e-9


def test_profit_uses_live_price_not_sp(base_config):
    df_one = make_bet_df([5.0], [1], sp_prices=[4.0])
    df_two = make_bet_df([5.0], [1], sp_prices=[7.0])
    result_one = settle_bets(df_one, base_config)
    result_two = settle_bets(df_two, base_config)
    assert abs(result_one["profit"].iloc[0] - result_two["profit"].iloc[0]) < 1e-9


def test_profit_uses_live_price_not_sp_loser(base_config):
    df_one = make_bet_df([5.0], [0], sp_prices=[4.0])
    df_two = make_bet_df([5.0], [0], sp_prices=[99.0])
    result_one = settle_bets(df_one, base_config)
    result_two = settle_bets(df_two, base_config)
    assert abs(result_one["profit"].iloc[0] - result_two["profit"].iloc[0]) < 1e-9


def test_settlement_profit_column_exists(base_config):
    df = make_bet_df([4.0, 8.0], [1, 0])
    result = settle_bets(df, base_config)
    assert "profit" in result.columns
    assert "profit_net" in result.columns


def test_roi_calculation():
    profits = pd.Series([4.0, -1.0, -1.0, 9.0])
    stakes = pd.Series([1.0, 1.0, 1.0, 1.0])
    roi = compute_roi(profits, stakes)
    assert abs(roi - (11.0 / 4.0)) < 1e-9


def test_roi_zero_when_no_bets():
    roi = compute_roi(pd.Series([], dtype=float), pd.Series([], dtype=float))
    assert roi == 0.0


def test_settle_multiple_bets(base_config):
    df = make_bet_df([3.0, 5.0, 7.0], [1, 0, 1])
    result = settle_bets(df, base_config)
    assert abs(result["profit"].iloc[0] - 2.0) < 1e-9
    assert abs(result["profit"].iloc[1] + 1.0) < 1e-9
    assert abs(result["profit"].iloc[2] - 6.0) < 1e-9
