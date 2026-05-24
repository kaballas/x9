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
    model_prob = _series(df, "model_prob", default=0.0).fillna(0.0)
    fair_market_prob = _series(
        df,
        "fair_market_prob" if "fair_market_prob" in df.columns else "market_implied_prob",
        default=0.0,
    ).fillna(0.0)
    raw_market_prob = _series(df, "raw_market_prob", default=float("nan"))
    if raw_market_prob.isna().any():
        price = _series(df, config["live_price_column"], default=float("nan"))
        raw_market_prob = raw_market_prob.fillna(
            (1.0 / price.where(price > 0)).fillna(0.0)
        )

    fair_edge = model_prob - fair_market_prob
    raw_edge = model_prob - raw_market_prob
    ev = compute_expected_value(model_prob, _series(df, config["live_price_column"], default=float("nan")))
    
    sp_column = config["sp_reference_column"]
    live_column = config["live_price_column"]
    price_vs_sp = df[sp_column] - df[live_column]
    price_vs_sp = price_vs_sp.where(df[sp_column].notna(), pd.NA)
    
    return df.assign(
        fair_edge=fair_edge,
        raw_edge=raw_edge,
        ev=ev,
        fair_edge_pct=fair_edge * 100.0,
        raw_edge_pct=raw_edge * 100.0,
        ev_pct=ev * 100.0,
        edge=fair_edge,  # Backward-compatible alias
        edge_pct=fair_edge * 100.0,  # Backward-compatible alias
        price_vs_sp=price_vs_sp
    )


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
