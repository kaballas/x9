"""Selection filters for betting candidates."""

from __future__ import annotations

import pandas as pd


def _log_filter(name: str, before_rows: int, after_rows: int) -> None:
    print(f"{name}: rows {before_rows} -> {after_rows}")


def apply_filters(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply the version-1 filter stack in the planned order."""
    df = filter_no_live_price(df, config)
    df = filter_price_coverage(df, config)
    df = filter_price_range(df, config)
    df = filter_field_size(df, config)
    df = filter_model_rank(df, config)
    df = filter_value_overlay(df, config)
    df = filter_sparse_form(df, config)
    df = filter_one_per_race(df, config)
    return df


def filter_no_live_price(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Remove runners without a valid live price when configured to do so."""
    before = len(df)
    result = df.copy()
    if "price_coverage" not in result.columns:
        race_counts = result.groupby("race_id").size().rename("race_total_runners")
        priced_counts = result.groupby("race_id")["has_valid_live_price"].sum().rename("race_priced_runners")
        result = result.join(race_counts, on="race_id")
        result = result.join(priced_counts, on="race_id")
        result["price_coverage"] = result["race_priced_runners"] / result["race_total_runners"]
    if config["exclude_runner_if_no_live_price"]:
        result = result.loc[result["has_valid_live_price"]].copy()
    _log_filter("filter_no_live_price", before, len(result))
    return result


def filter_price_range(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Keep only runners inside the tested live-price range.

    Runners with no live price are passed through when
    ``exclude_runner_if_no_live_price`` is False so that pre-race (unpriced)
    runners are not silently dropped here after surviving filter_no_live_price.
    """
    before = len(df)
    price = df["live_price"]
    in_range = price.notna() & price.between(config["min_price"], config["max_price"])
    if not config.get("exclude_runner_if_no_live_price", True):
        in_range = in_range | price.isna()
    result = df.loc[in_range].copy()
    _log_filter("filter_price_range", before, len(result))
    return result


def filter_field_size(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Drop runners from races with invalid declared field sizes."""
    del config
    before = len(df)
    result = df.loc[df["is_valid_field_size"]].copy()
    _log_filter("filter_field_size", before, len(result))
    return result


def filter_min_edge(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Keep only runners whose edge meets the configured minimum."""
    before = len(df)
    result = df.loc[df["edge"].notna() & (df["edge"] >= config["min_edge"])].copy()
    _log_filter("filter_min_edge", before, len(result))
    return result


def filter_model_rank(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Keep only top-ranked model runners."""
    before = len(df)
    result = df.loc[df["model_rank"].notna() & (df["model_rank"] <= config["max_model_rank"])].copy()
    _log_filter("filter_model_rank", before, len(result))
    return result


def _value_overlay_mask(df: pd.DataFrame, config: dict) -> pd.Series:
    min_model_prob = float(config.get("min_model_probability", 0.0))
    min_ratio = float(config.get("min_model_vs_market_ratio", 0.0))
    min_raw_edge = float(config.get("min_raw_edge", 0.0))
    min_ev = float(config.get("min_ev", 0.0))
    min_score_gap = float(config.get("min_score_gap_to_next", 0.0))

    model_prob = pd.to_numeric(df.get("model_prob"), errors="coerce")
    raw_market_prob = pd.to_numeric(df.get("raw_market_prob"), errors="coerce")
    if raw_market_prob.isna().any():
        live_price = pd.to_numeric(df.get("live_price"), errors="coerce")
        raw_market_prob = raw_market_prob.fillna((1.0 / live_price.where(live_price > 0)).fillna(0.0))

    raw_edge = pd.to_numeric(df.get("raw_edge"), errors="coerce")
    if raw_edge.isna().any():
        raw_edge = raw_edge.fillna(model_prob - raw_market_prob)

    ev = pd.to_numeric(df.get("ev"), errors="coerce")
    if ev.isna().any():
        live_price = pd.to_numeric(df.get("live_price"), errors="coerce")
        ev = ev.fillna(model_prob * live_price - 1.0)
    score_gap_src = df.get("model_score_gap_to_next")
    if isinstance(score_gap_src, pd.Series):
        score_gap = pd.to_numeric(score_gap_src, errors="coerce").fillna(0.0)
        score_gap_mask = score_gap >= min_score_gap
    else:
        score_gap = pd.Series(0.0, index=df.index, dtype=float)
        score_gap_mask = pd.Series(True, index=df.index, dtype=bool)

    mask = (
        model_prob.notna()
        & raw_market_prob.notna()
        & (raw_market_prob > 0)
        & (model_prob >= min_model_prob)
        & raw_edge.notna()
        & (raw_edge >= min_raw_edge)
        & ev.notna()
        & (ev >= min_ev)
        & score_gap_mask
    )
    if min_ratio > 0:
        mask = mask & (model_prob >= raw_market_prob * min_ratio)

    # When unpriced runners are allowed through (pre-race / no live price),
    # pass them through value overlay too — EV and edge cannot be computed
    # without a price and should not gate display-only tools.
    if not config.get("exclude_runner_if_no_live_price", True):
        no_price = pd.to_numeric(df.get("live_price"), errors="coerce").isna()
        mask = mask | no_price

    return mask


def filter_value_overlay(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Keep only runners that clear probability, raw-edge, and EV requirements."""
    before = len(df)
    result = df.loc[_value_overlay_mask(df, config)].copy()
    _log_filter("filter_value_overlay", before, len(result))
    return result


def candidate_mask(df: pd.DataFrame, config: dict) -> pd.Series:
    """Return the full candidate mask used for bet qualification."""
    min_price = float(config.get("min_price", 0.0))
    max_price = float(config.get("max_price", float("inf")))
    max_rank = float(config.get("max_model_rank", float("inf")))
    live_price = pd.to_numeric(df.get("live_price"), errors="coerce")
    model_rank = pd.to_numeric(df.get("model_rank"), errors="coerce")
    race_integrity = df.get("race_integrity_ok")
    if race_integrity is None:
        race_integrity_mask = pd.Series(True, index=df.index)
    else:
        race_integrity_mask = pd.Series(race_integrity, index=df.index).fillna(False).astype(bool)
    return (
        race_integrity_mask
        & live_price.notna()
        & (live_price >= min_price)
        & (live_price <= max_price)
        & model_rank.notna()
        & (model_rank <= max_rank)
        & _value_overlay_mask(df, config)
    )


def filter_one_per_race(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Keep one candidate per race when multiple bets are disabled."""
    before = len(df)
    if config["allow_multiple_bets_per_race"]:
        _log_filter("filter_one_per_race", before, len(df))
        return df.copy()

    sort_columns = ["race_id", "ev", "raw_edge", "model_rank", "runner_number"]
    ascending = [True, False, False, True, True]
    result = (
        df.sort_values(sort_columns, ascending=ascending)
        .drop_duplicates(subset=["race_id"], keep="first")
        .copy()
    )
    _log_filter("filter_one_per_race", before, len(result))
    return result


def filter_sparse_form(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Drop runners with insufficient recent-form evidence."""
    del config
    before = len(df)
    result = df.loc[~df["has_sparse_recent_form"]].copy()
    _log_filter("filter_sparse_form", before, len(result))
    return result


def filter_price_coverage(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Drop entire races whose live-price coverage is below threshold."""
    before = len(df)
    result = df.loc[
        df["price_coverage"].notna()
        & (df["price_coverage"] >= config["exclude_race_if_price_coverage_below"])
    ].copy()
    _log_filter("filter_price_coverage", before, len(result))
    return result
