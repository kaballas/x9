"""Input validation for the runner-level betting pipeline."""

from __future__ import annotations

import pandas as pd

from .schema import (
    IDENTITY_COLUMNS,
    REQUIRED_FEATURE_COLUMNS,
    REQUIRED_PRICE_COLUMNS,
    REQUIRED_RESULT_COLUMNS,
    validate_schema,
)


def _is_backtest_frame(df: pd.DataFrame, config: dict) -> bool:
    winner_col = config["winner_col"]
    if winner_col in df.columns and df[winner_col].notna().any():
        return True
    return "status" in df.columns and df["status"].eq("finished").any()


def validate_input(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Run validation checks in order and return the cleaned DataFrame."""
    check_required_columns(df, config)
    df = check_duplicate_runners(df, config)
    df = check_live_price(df, config)
    df = check_field_size(df, config)
    df = check_result_labels(df, config)
    return df


def check_required_columns(df: pd.DataFrame, config: dict) -> None:
    """Validate that all required columns are present for the current mode."""
    required = IDENTITY_COLUMNS + REQUIRED_FEATURE_COLUMNS + REQUIRED_PRICE_COLUMNS
    validate_schema(df, required)
    if _is_backtest_frame(df, config):
        validate_schema(df, REQUIRED_RESULT_COLUMNS)


def check_duplicate_runners(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Flag and drop duplicate race/runner rows."""
    if df.empty:
        result = df.copy()
        result["is_duplicate_runner"] = pd.Series(dtype=bool)
        return result

    race_id_col = config["race_id_col"]
    runner_id_col = config["runner_id_col"]
    result = df.copy()
    result["is_duplicate_runner"] = result.duplicated(
        subset=[race_id_col, runner_id_col], keep=False
    )
    duplicate_count = int(result["is_duplicate_runner"].sum())
    print(f"check_duplicate_runners: duplicates={duplicate_count}")
    return result.loc[~result["is_duplicate_runner"]].copy()


def check_live_price(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Flag invalid live prices and optionally drop those rows."""
    result = df.copy()
    live_price_col = config["live_price_column"]
    result["has_valid_live_price"] = result[live_price_col].notna() & (result[live_price_col] > 0)
    race_counts = result.groupby(config["race_id_col"]).size().rename("race_total_runners")
    priced_counts = result.groupby(config["race_id_col"])["has_valid_live_price"].sum().rename("race_priced_runners")
    result = result.join(race_counts, on=config["race_id_col"])
    result = result.join(priced_counts, on=config["race_id_col"])
    result["price_coverage"] = result["race_priced_runners"] / result["race_total_runners"]
    invalid_count = int((~result["has_valid_live_price"]).sum())
    print(f"check_live_price: invalid_live_price_rows={invalid_count}")
    if config["exclude_runner_if_no_live_price"]:
        return result.loc[result["has_valid_live_price"]].copy()
    return result


def check_field_size(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Flag rows belonging to races outside the configured active field-size window.

    Uses active_field_size (runners after scratchings), not declared field_size.
    """
    result = df.copy()
    field_size = result["active_field_size"]
    result["is_valid_field_size"] = field_size.notna() & field_size.between(
        config["min_field_size"], config["max_field_size"]
    )
    invalid_count = int((~result["is_valid_field_size"]).sum())
    print(f"check_field_size: invalid_field_size_rows={invalid_count}")
    return result


def check_result_labels(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Flag missing settlement labels in backtest mode only."""
    result = df.copy()
    winner_col = config["winner_col"]
    if _is_backtest_frame(result, config):
        result["has_valid_result_labels"] = result[winner_col].notna()
    else:
        result["has_valid_result_labels"] = True

    result["race_integrity_ok"] = (
        result.get("has_valid_live_price", True)
        & result.get("is_valid_field_size", True)
        & result["has_valid_result_labels"]
    )
    invalid_count = int((~result["has_valid_result_labels"]).sum())
    print(f"check_result_labels: invalid_result_label_rows={invalid_count}")
    return result
