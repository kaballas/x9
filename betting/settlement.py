"""Backtest settlement and ROI helpers."""

from __future__ import annotations

import pandas as pd


def settle_bets(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Settle historical bets at live_price using flat win-only rules."""
    result = df.copy()
    settlement_col = config["settlement_price_column"]
    result["settlement_price"] = result[settlement_col]
    result["profit"] = result.apply(compute_profit, axis=1)
    result["profit_net"] = result["profit"]
    return result


def compute_profit(row: pd.Series) -> float:
    """Compute profit for one settled runner bet."""
    stake = float(row["stake"])
    settlement_price = float(row["settlement_price"])
    return (settlement_price - 1.0) * stake if int(row["is_winner"]) == 1 else -1.0 * stake


def compute_roi(profit_series: pd.Series, stake_series: pd.Series) -> float:
    """Compute aggregate ROI as profit divided by stake."""
    total_stake = float(stake_series.sum())
    if total_stake == 0:
        return 0.0
    return float(profit_series.sum()) / total_stake
