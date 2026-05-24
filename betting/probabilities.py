"""Probability assignment utilities for runner scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_probabilities(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add within-race model probabilities plus raw and fair market probabilities."""
    result = df.copy()
    if result.empty:
        result["raw_model_prob"] = pd.Series(dtype=float)
        result["model_prob"] = pd.Series(dtype=float)
        result["raw_market_prob"] = pd.Series(dtype=float)
        result["book_overround"] = pd.Series(dtype=float)
        result["fair_market_prob"] = pd.Series(dtype=float)
        result["market_implied_prob"] = pd.Series(dtype=float)
        return result

    temperature = float(config.get("prob_temperature", 4.0))
    result["raw_model_prob"] = result.groupby("race_id")["model_score"].transform(
        lambda s: softmax_within_race(s, temperature=temperature)
    )
    result["model_prob"] = result["raw_model_prob"]

    result["raw_market_prob"] = compute_market_implied_prob(result["live_price"])
    if "race_id" in result.columns:
        result["book_overround"] = result.groupby("race_id")["raw_market_prob"].transform("sum")
        result["fair_market_prob"] = result.groupby("race_id")["live_price"].transform(
            normalise_market_prob
        )
    else:
        result["book_overround"] = pd.Series(result["raw_market_prob"].sum(), index=result.index)
        result["fair_market_prob"] = normalise_market_prob(result["live_price"])

    # Backward-compatible alias used across the codebase.
    result["market_implied_prob"] = result["fair_market_prob"]
    return result


def softmax_within_race(scores: pd.Series, temperature: float = 4.0) -> pd.Series:
    """Compute temperature-scaled softmax probabilities for one race."""
    numeric = pd.to_numeric(scores, errors="coerce")
    if numeric.notna().sum() == 0:
        count = len(scores)
        if count == 0:
            return pd.Series(dtype=float, index=scores.index)
        return pd.Series(1.0 / count, index=scores.index)

    numeric = numeric.fillna(0.0)
    shifted = numeric - numeric.max()
    exp_scores = np.exp(temperature * shifted)
    total = exp_scores.sum()
    if total == 0:
        return pd.Series(1.0 / len(scores), index=scores.index)
    return pd.Series(exp_scores / total, index=scores.index)


def compute_market_implied_prob(live_price: pd.Series) -> pd.Series:
    """Convert live price to reciprocal implied probability."""
    prices = pd.to_numeric(live_price, errors="coerce")
    valid_prices = prices.where(prices > 0)
    return (1.0 / valid_prices).fillna(0.0)


def normalise_market_prob(live_price: pd.Series) -> pd.Series:
    """Convert prices to fair probabilities by removing bookmaker overround."""
    prices = pd.to_numeric(live_price, errors="coerce")
    raw_inv = (1.0 / prices.where(prices > 0)).fillna(0.0)
    total = raw_inv.sum()
    if total <= 0:
        return pd.Series(0.0, index=live_price.index)
    return raw_inv / total
