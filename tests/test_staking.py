"""Tests for staking assignment helpers."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from betting.config import CONFIG
from betting.staking import FibonacciStaker, assign_stakes, fibonacci_stakes, replay_fibonacci_level


@pytest.fixture
def base_config():
    return CONFIG.copy()


def test_fibonacci_staker_initial_level_and_stake():
    staker = FibonacciStaker(base_unit=2.0)
    assert staker.level == 0
    assert staker.current_stake() == pytest.approx(2.0)


def test_fibonacci_staker_loss_advances_level():
    staker = FibonacciStaker()
    staker.on_loss()
    assert staker.level == 1
    staker.on_loss()
    assert staker.level == 2


def test_fibonacci_staker_two_back_win_moves_back_two_levels():
    staker = FibonacciStaker(variant="two_back")
    staker.level = 4
    staker.on_win()
    assert staker.level == 2


def test_fibonacci_staker_two_back_win_stops_at_zero():
    staker = FibonacciStaker(variant="two_back")
    staker.level = 1
    staker.on_win()
    assert staker.level == 0


def test_fibonacci_staker_one_back_win_moves_back_one_level():
    staker = FibonacciStaker(variant="one_back")
    staker.level = 3
    staker.on_win()
    assert staker.level == 2


def test_fibonacci_staker_reset_variant_returns_to_zero():
    staker = FibonacciStaker(variant="reset")
    staker.level = 5
    staker.on_win()
    assert staker.level == 0


def test_fibonacci_staker_caps_level_at_max_level():
    staker = FibonacciStaker(max_level=2)
    for _ in range(10):
        staker.on_loss()
    assert staker.level == 2


def test_fibonacci_staker_current_stake_uses_multiplier():
    staker = FibonacciStaker(base_unit=2.5)
    staker.level = 3
    assert staker.current_stake() == pytest.approx(12.5)


def test_replay_fibonacci_level_returns_next_level_after_history():
    assert replay_fibonacci_level([1, 0, 0, 1, 0], variant="two_back", max_level=10) == 1


def test_replay_fibonacci_level_respects_start_level_and_cap():
    assert replay_fibonacci_level([0, 0], variant="one_back", max_level=2, start_level=1) == 2


def test_fibonacci_stakes_adds_expected_columns_and_values(base_config):
    config = base_config.copy()
    config.update(
        {
            "staking_mode": "fibonacci",
            "fib_variant": "two_back",
            "fib_base_unit": 1.0,
            "fib_max_level": 10,
        }
    )
    df = pd.DataFrame(
        {
            "race_id": [f"R{idx}" for idx in range(5)],
            "selection_id": list(range(5)),
            "live_price": [4.0, 5.0, 6.0, 3.0, 10.0],
            "is_winner": [1, 0, 0, 1, 0],
        }
    )

    result = fibonacci_stakes(df, config)

    assert result["fib_stake"].tolist() == pytest.approx([1.0, 1.0, 2.0, 3.0, 1.0])
    assert result["fib_profit"].tolist() == pytest.approx([3.0, -1.0, -2.0, 6.0, -1.0])
    assert result["fib_level"].tolist() == [0, 0, 1, 2, 0]
    assert result["stake"].tolist() == pytest.approx([1.0, 1.0, 2.0, 3.0, 1.0])


def test_fibonacci_stakes_win_and_loss_rules_at_level_zero(base_config):
    config = base_config.copy()
    config.update(
        {
            "staking_mode": "fibonacci",
            "fib_variant": "two_back",
            "fib_base_unit": 1.0,
            "fib_max_level": 10,
        }
    )
    df = pd.DataFrame(
        {
            "race_id": ["R1", "R2"],
            "selection_id": [1, 2],
            "live_price": [5.0, 5.0],
            "is_winner": [1, 0],
        }
    )

    result = fibonacci_stakes(df, config)

    assert result.loc[0, "fib_stake"] == pytest.approx(1.0)
    assert result.loc[0, "fib_profit"] == pytest.approx(4.0)
    assert result.loc[0, "fib_level"] == 0
    assert result.loc[1, "fib_stake"] == pytest.approx(1.0)
    assert result.loc[1, "fib_profit"] == pytest.approx(-1.0)
    assert result.loc[1, "fib_level"] == 0


def test_assign_stakes_uses_fibonacci_mode(base_config):
    config = base_config.copy()
    config.update(
        {
            "staking_mode": "fibonacci",
            "fib_variant": "two_back",
            "fib_base_unit": 2.0,
            "fib_max_level": 10,
        }
    )
    df = pd.DataFrame(
        {
            "race_id": ["R1", "R2"],
            "selection_id": [1, 2],
            "live_price": [3.0, 4.0],
            "is_winner": [0, 0],
        }
    )

    result = assign_stakes(df, config)

    assert result["stake"].tolist() == pytest.approx([2.0, 4.0])
    assert result["fib_level"].tolist() == [0, 1]
