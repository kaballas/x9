"""Lightweight historical meta-model signal for runner ranking."""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .db import load_backtest_data, load_draw_bias_table, load_jockey_stats_table, load_trainer_stats_table
from .features import build_features
from .validation import validate_input

_META_MODEL_COLUMNS = [
    "speed_feature_score",
    "speed_consistency",
    "recent_form_score",
    "suitability_score",
    "connection_score",
    "market_movement_score",
    "margin_score",
    "freshness_score",
    "class_score",
    "draw_bias_score",
    "jockey_score",
    "trainer_score",
    "live_price",
    "market_rank",
    "recent_sp_score",
    "class_movement_score",
    "distance_change_score",
    "weight_trend_score",
    "barrier_transition_score",
    "pedigree_score",
    "form_string_score",
    "jockey_continuity_score",
    "travel_score",
    "equipment_score",
    "active_field_size",
]


def _meta_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    work: dict[str, pd.Series] = {}
    for column in _META_MODEL_COLUMNS:
        if column in df.columns:
            work[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            work[column] = pd.Series(np.nan, index=df.index, dtype=float)
    return pd.DataFrame(work, index=df.index)


def _fit_model(feature_df: pd.DataFrame, labels: pd.Series) -> HistGradientBoostingClassifier | None:
    y = pd.to_numeric(labels, errors="coerce").fillna(0).astype(int)
    if len(feature_df) < 20 or y.nunique() < 2:
        return None
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=4,
        max_iter=200,
        random_state=13,
    )
    model.fit(feature_df, y)
    return model


def _ordered_race_folds(race_ids: pd.Series, n_folds: int = 5) -> pd.Series:
    unique_races = pd.Series(race_ids).drop_duplicates().tolist()
    fold_map = {race_id: idx % n_folds for idx, race_id in enumerate(unique_races)}
    return race_ids.map(fold_map).fillna(0).astype(int)


def _build_training_frame(
    config: dict,
    draw_bias_df: pd.DataFrame | None,
    jockey_stats_df: pd.DataFrame | None,
    trainer_stats_df: pd.DataFrame | None,
) -> pd.DataFrame:
    raw = load_backtest_data(config["database_path"])
    if raw.empty:
        return raw
    validated = validate_input(raw, config)
    return build_features(validated, config, draw_bias_df, jockey_stats_df, trainer_stats_df)


def add_meta_model_signal(
    df: pd.DataFrame,
    config: dict,
    draw_bias_df: pd.DataFrame | None = None,
    jockey_stats_df: pd.DataFrame | None = None,
    trainer_stats_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add a learned runner win signal from historical numeric features."""
    result = df.copy()
    feature_df = _meta_feature_frame(result)
    if result.empty:
        result["meta_model_score"] = pd.Series(dtype=float)
        return result

    labels = pd.to_numeric(result.get("is_winner"), errors="coerce")
    has_result_labels = labels.notna().any() and labels.nunique(dropna=True) >= 2

    if has_result_labels:
        folds = _ordered_race_folds(result["race_id"])
        predictions = pd.Series(0.0, index=result.index, dtype=float)
        for fold in sorted(folds.unique()):
            train_mask = folds != fold
            test_mask = folds == fold
            model = _fit_model(feature_df.loc[train_mask], labels.loc[train_mask])
            if model is None:
                continue
            predictions.loc[test_mask] = model.predict_proba(feature_df.loc[test_mask])[:, 1]
        result["meta_model_score"] = predictions.fillna(0.0)
        return result

    if (draw_bias_df is None or jockey_stats_df is None or trainer_stats_df is None) and config.get("database_path"):
        with sqlite3.connect(config["database_path"]) as conn:
            if draw_bias_df is None:
                draw_bias_df = load_draw_bias_table(conn)
            if jockey_stats_df is None:
                jockey_stats_df = load_jockey_stats_table(conn)
            if trainer_stats_df is None:
                trainer_stats_df = load_trainer_stats_table(conn)

    training_df = _build_training_frame(config, draw_bias_df, jockey_stats_df, trainer_stats_df)
    training_labels = pd.to_numeric(training_df.get("is_winner"), errors="coerce")
    model = _fit_model(_meta_feature_frame(training_df), training_labels)
    if model is None:
        result["meta_model_score"] = pd.Series(0.0, index=result.index, dtype=float)
        return result

    result["meta_model_score"] = model.predict_proba(feature_df)[:, 1]
    return result
