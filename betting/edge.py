"""Edge calculations comparing model and market probabilities."""

from __future__ import annotations

import pandas as pd


def _series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def calculate_edges(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add edge metrics and SP diagnostics.

    sp_starting_price is reference-only and must not feed selection logic.
    """
    result = df.copy()
    model_prob = _series(result, "model_prob", default=0.0).fillna(0.0)
    fair_market_prob = _series(
        result,
        "fair_market_prob" if "fair_market_prob" in result.columns else "market_implied_prob",
        default=0.0,
    ).fillna(0.0)
    raw_market_prob = _series(result, "raw_market_prob", default=float("nan"))
    if raw_market_prob.isna().any():
        price = _series(result, config["live_price_column"], default=float("nan"))
        raw_market_prob = raw_market_prob.fillna(
            (1.0 / price.where(price > 0)).fillna(0.0)
        )

    result["fair_edge"] = model_prob - fair_market_prob
    result["raw_edge"] = model_prob - raw_market_prob
    result["ev"] = compute_expected_value(model_prob, _series(result, config["live_price_column"], default=float("nan")))
    result["fair_edge_pct"] = result["fair_edge"] * 100.0
    result["raw_edge_pct"] = result["raw_edge"] * 100.0
    result["ev_pct"] = result["ev"] * 100.0

    # Backward-compatible aliases.
    result["edge"] = result["fair_edge"]
    result["edge_pct"] = result["fair_edge_pct"]
    sp_column = config["sp_reference_column"]
    live_column = config["live_price_column"]
    result["price_vs_sp"] = result[sp_column] - result[live_column]
    result.loc[result[sp_column].isna(), "price_vs_sp"] = pd.NA
    return result


def compute_edge(model_prob: float, market_implied_prob: float) -> float:
    """Return model probability edge over the market."""
    model_prob = 0.0 if pd.isna(model_prob) else float(model_prob)
    market_implied_prob = 0.0 if pd.isna(market_implied_prob) else float(market_implied_prob)
    return model_prob - market_implied_prob


def compute_expected_value(model_prob: pd.Series, live_price: pd.Series) -> pd.Series:
    """Expected value of a 1-unit win bet: model_prob * price - 1."""
    prob = pd.to_numeric(model_prob, errors="coerce").fillna(0.0)
    price = pd.to_numeric(live_price, errors="coerce")
    valid_price = price.where(price > 0)
    return (prob * valid_price - 1.0).fillna(-1.0)
