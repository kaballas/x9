"""Probability assignment utilities for runner scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_probabilities(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add within-race model probabilities plus raw and fair market probabilities."""
    if df.empty:
        return df.assign(
            raw_model_prob=pd.Series(dtype=float),
            model_prob=pd.Series(dtype=float),
            raw_market_prob=pd.Series(dtype=float),
            book_overround=pd.Series(dtype=float),
            fair_market_prob=pd.Series(dtype=float),
            market_implied_prob=pd.Series(dtype=float)
        )

    temperature = float(config.get("prob_temperature", 4.0))
    raw_model_prob = df.groupby("race_id")["model_score"].transform(
        lambda s: softmax_within_race(s, temperature=temperature)
    )
    raw_market_prob = compute_market_implied_prob(df["live_price"])
    
    if "race_id" in df.columns:
        book_overround = raw_market_prob.groupby(df["race_id"]).transform("sum")
        fair_market_prob = df.groupby("race_id")["live_price"].transform(normalise_market_prob)
    else:
        book_overround = pd.Series(raw_market_prob.sum(), index=df.index)
        fair_market_prob = normalise_market_prob(df["live_price"])

    # Backward-compatible alias used across the codebase.
    return df.assign(
        raw_model_prob=raw_model_prob,
        model_prob=raw_model_prob,
        raw_market_prob=raw_market_prob,
        book_overround=book_overround,
        fair_market_prob=fair_market_prob,
        market_implied_prob=fair_market_prob
    )


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
