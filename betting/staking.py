"""Stake assignment for selected betting candidates."""

from __future__ import annotations

import pandas as pd

_FIB_SEQUENCE = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]


class FibonacciStaker:
    """Fibonacci staking tracker for sequential bet series."""

    def __init__(self, variant: str = "two_back", base_unit: float = 1.0, max_level: int = 10):
        self.variant = variant
        self.base_unit = base_unit
        self.max_level = min(max_level, len(_FIB_SEQUENCE) - 1)
        self.level = 0

    def on_win(self) -> None:
        if self.variant == "two_back":
            self.level = max(0, self.level - 2)
        elif self.variant == "one_back":
            self.level = max(0, self.level - 1)
        else:
            self.level = 0

    def on_loss(self) -> None:
        self.level = min(self.level + 1, self.max_level)

    def current_stake(self) -> float:
        return self.base_unit * _FIB_SEQUENCE[self.level]

    def reset(self) -> None:
        self.level = 0

    @property
    def current_multiplier(self) -> int:
        return _FIB_SEQUENCE[self.level]


def replay_fibonacci_level(
    results: list[int] | pd.Series,
    variant: str = "two_back",
    max_level: int = 10,
    start_level: int = 0,
) -> int:
    """Replay win/loss results to determine the next Fibonacci level."""
    staker = FibonacciStaker(variant=variant, max_level=max_level)
    staker.level = max(0, min(int(start_level), staker.max_level))
    for result in results:
        if int(result) == 1:
            staker.on_win()
        else:
            staker.on_loss()
    return staker.level


def assign_stakes(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Assign stakes. Uses flat staking by default; fibonacci if configured."""
    mode = config.get("staking_mode", "flat")
    if mode == "fibonacci":
        return fibonacci_stakes(df, config)
    return flat_stake(df, config)


def flat_stake(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply the configured flat stake to every row."""
    result = df.copy()
    result["stake"] = config["stake"]
    return result


def fibonacci_stakes(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply Fibonacci staking to a sequence of bets in chronological order."""
    result = df.copy()
    staker = FibonacciStaker(
        variant=config.get("fib_variant", "two_back"),
        base_unit=float(config.get("fib_base_unit", 1.0)),
        max_level=int(config.get("fib_max_level", 10)),
    )

    fib_stakes: list[float] = []
    fib_profits: list[float] = []
    fib_levels: list[int] = []

    for row in result.itertuples(index=False):
        fib_level = staker.level
        fib_stake = staker.current_stake()
        live_price = float(row.live_price)
        is_winner = int(row.is_winner)
        fib_profit = fib_stake * (live_price - 1.0) if is_winner == 1 else -fib_stake

        fib_levels.append(fib_level)
        fib_stakes.append(fib_stake)
        fib_profits.append(fib_profit)

        if is_winner == 1:
            staker.on_win()
        else:
            staker.on_loss()

    result["fib_level"] = fib_levels
    result["fib_stake"] = fib_stakes
    result["fib_profit"] = fib_profits
    result["stake"] = result["fib_stake"]
    return result
