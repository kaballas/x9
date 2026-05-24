"""Runner scoring and race-level model ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHT_KEYS = [
    "weight_speed_rating",
    "weight_recent_form",
    "weight_suitability",
    "weight_connections",
    "weight_market_sanity",
    "weight_steam",
    "weight_margin",
    "weight_freshness",
    "weight_class",
    "weight_draw_bias",
    "weight_jockey",
    "weight_trainer",
]


def get_effective_weights(config: dict) -> tuple[dict[str, float], float]:
    """Return effective component weights and the raw pre-normalization total.

    Behavior:
    - raw_total <= 1.0: use raw weights as-is (backward-compatible behavior)
    - raw_total > 1.0: normalize down so effective weights sum to 1.0
    """
    raw: dict[str, float] = {}
    for key in WEIGHT_KEYS:
        try:
            value = float(config.get(key, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        raw[key] = value if np.isfinite(value) else 0.0
    raw_total = float(sum(raw.values()))
    if raw_total <= 0.0:
        return {key: 0.0 for key in WEIGHT_KEYS}, 0.0
    if raw_total <= 1.0:
        return raw, raw_total
    return {key: value / raw_total for key, value in raw.items()}, raw_total


def _numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    raw = df[column] if column in df.columns else pd.Series(default, index=df.index)
    if not isinstance(raw, pd.Series):
        raw = pd.Series(raw, index=df.index)
    return pd.to_numeric(raw, errors="coerce").fillna(default)


def _rank_normalize(series: pd.Series) -> pd.Series:
    """Normalize by within-field rank: slowest=0.0, fastest=1.0, evenly spaced.

    Unlike min-max, this is purely ordinal — the absolute rating gap between
    runners does not affect the spread.  A field of [76, 77, 100] produces
    the same [0, 0.5, 1.0] output as [76, 88, 100].  This prevents a single
    high-rated runner from collapsing the rest of the field near zero.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    n = numeric.notna().sum()
    if n <= 1:
        return pd.Series(0.0, index=series.index)
    # rank(ascending=True) → slowest gets rank 1, fastest gets rank n
    ranks = numeric.rank(method="average", ascending=True, na_option="keep")
    return ((ranks - 1) / (n - 1)).fillna(0.0)


def _min_max_normalize(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(0.0, index=series.index)
    min_value = numeric.min()
    max_value = numeric.max()
    if pd.isna(min_value) or pd.isna(max_value) or min_value == max_value:
        return pd.Series(0.0, index=series.index)
    return ((numeric - min_value) / (max_value - min_value)).fillna(0.0)


def _market_sanity_component(df: pd.DataFrame) -> pd.Series:
    live_price = pd.to_numeric(df["live_price"], errors="coerce")
    inverse_price = 1.0 / live_price.where(live_price > 0)
    return _min_max_normalize(inverse_price)


def score_runners(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute model_score and within-race model_rank."""
    result = df.copy()
    speed_source = result["speed_feature_score"] if "speed_feature_score" in result else result["condition_rating"]
    speed_source = pd.to_numeric(speed_source, errors="coerce").fillna(0.0)
    speed_confidence = pd.to_numeric(result.get("speed_feature_confidence", 1.0), errors="coerce").fillna(1.0)
    if "race_id" in result.columns:
        field_mean = speed_source.groupby(result["race_id"]).transform("mean")
    else:
        field_mean = pd.Series(speed_source.mean(), index=result.index)
    base_speed_score = 1.0 / (1.0 + np.exp(-(speed_source - field_mean) / 0.60))
    result["speed_score_norm"] = base_speed_score * speed_confidence + 0.5 * (1.0 - speed_confidence)
    result["market_sanity_score"] = _market_sanity_component(result)
    result["model_score"] = compute_model_score(result, config)
    return add_model_rank(result)


def compute_model_score(df: pd.DataFrame, config: dict) -> pd.Series:
    """Compute the version-1 weighted transparent score."""
    weights, _ = get_effective_weights(config)
    if "speed_score_norm" in df:
        speed_component = pd.to_numeric(df["speed_score_norm"], errors="coerce").fillna(0.0)
    else:
        speed_source = df["speed_feature_score"] if "speed_feature_score" in df else df["condition_rating"]
        speed_source = pd.to_numeric(speed_source, errors="coerce").fillna(0.0)
        speed_conf_raw = df.get("speed_feature_confidence")
        if speed_conf_raw is None:
            speed_confidence = pd.Series(1.0, index=df.index)
        else:
            speed_confidence = pd.to_numeric(speed_conf_raw, errors="coerce").fillna(1.0)
        if "race_id" in df.columns:
            field_mean = speed_source.groupby(df["race_id"]).transform("mean")
        else:
            field_mean = pd.Series(speed_source.mean(), index=df.index)
        base_speed_score = 1.0 / (1.0 + np.exp(-(speed_source - field_mean) / 0.60))
        speed_component = base_speed_score * speed_confidence + 0.5 * (1.0 - speed_confidence)
    form_component = pd.to_numeric(df["recent_form_score"], errors="coerce").fillna(0.0)
    suitability_component = pd.to_numeric(df["suitability_score"], errors="coerce").fillna(0.0)
    connection_component = pd.to_numeric(df["connection_score"], errors="coerce").fillna(0.0)
    if "market_sanity_score" in df:
        market_sanity = pd.to_numeric(df["market_sanity_score"], errors="coerce").fillna(0.0)
    else:
        market_sanity = _market_sanity_component(df)
    if "market_movement_score" in df:
        steam_component = pd.to_numeric(df["market_movement_score"], errors="coerce").fillna(0.5)
    else:
        steam_component = pd.Series(0.5, index=df.index, dtype=float)
    margin_component = pd.to_numeric(df["margin_score"], errors="coerce").fillna(0.5)
    freshness_component = pd.to_numeric(df["freshness_score"], errors="coerce").fillna(0.5)
    class_component = pd.to_numeric(df["class_score"], errors="coerce").fillna(0.0)
    if "draw_bias_score" in df:
        draw_bias_component = pd.to_numeric(df["draw_bias_score"], errors="coerce").fillna(0.25)
    else:
        draw_bias_component = pd.Series(0.25, index=df.index)
    if "jockey_score" in df:
        jockey_component = pd.to_numeric(df["jockey_score"], errors="coerce").fillna(0.15)
    else:
        jockey_component = pd.Series(0.15, index=df.index)
    if "trainer_score" in df:
        trainer_component = pd.to_numeric(df["trainer_score"], errors="coerce").fillna(0.15)
    else:
        trainer_component = pd.Series(0.15, index=df.index)

    model_score = (
        weights["weight_speed_rating"] * speed_component
        + weights["weight_recent_form"] * form_component
        + weights["weight_suitability"] * suitability_component
        + weights["weight_connections"] * connection_component
        + weights["weight_market_sanity"] * market_sanity
        + weights["weight_steam"] * steam_component
        + weights["weight_margin"] * margin_component
        + weights["weight_freshness"] * freshness_component
        + weights["weight_class"] * class_component
        + weights["weight_draw_bias"] * draw_bias_component
        + weights["weight_jockey"] * jockey_component
        + weights["weight_trainer"] * trainer_component
    )
    meta_raw = df.get("meta_model_score")
    if isinstance(meta_raw, pd.Series):
        meta_component = pd.to_numeric(meta_raw, errors="coerce").fillna(0.0)
    else:
        meta_component = pd.Series(0.0, index=df.index, dtype=float)
    if meta_component.nunique(dropna=False) > 1:
        model_score = 0.20 * model_score + 0.80 * meta_component
    return pd.to_numeric(model_score, errors="coerce").fillna(0.0)


def add_model_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Rank runners by descending model_score within each race."""
    result = df.copy()
    result["model_rank"] = result.groupby("race_id")["model_score"].rank(
        method="min", ascending=False
    )
    sort_cols = ["race_id", "model_score"]
    ascending = [True, False]
    if "runner_number" in result.columns:
        sort_cols.append("runner_number")
        ascending.append(True)
    ordering = result.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    gap = ordering["model_score"] - ordering.groupby("race_id")["model_score"].shift(-1)
    result["model_score_gap_to_next"] = gap.reindex(result.index).fillna(0.0)
    return result
