"""Probability calibration hooks for the betting framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Module-level cache so the pkl is only read from disk once per process.
_CALIBRATION_MODEL: Any = None
_CALIBRATION_MODEL_PATH: str | None = None


def _load_model(path: str) -> Any | None:
    """Load a joblib-serialised calibration model, with a per-process cache.

    Returns None if the file does not exist (passthrough mode).
    """
    global _CALIBRATION_MODEL, _CALIBRATION_MODEL_PATH
    if _CALIBRATION_MODEL_PATH == path:
        return _CALIBRATION_MODEL
    _CALIBRATION_MODEL_PATH = path
    if Path(path).exists():
        import joblib
        _CALIBRATION_MODEL = joblib.load(path)
    else:
        _CALIBRATION_MODEL = None
    return _CALIBRATION_MODEL


def _default_model_path() -> str:
    """Return the default calibration model path (sibling of this file)."""
    return str(Path(__file__).parent / "calibration_model.pkl")


def _predict_calibrated(model: Any, raw_probs: pd.Series) -> np.ndarray:
    """Predict calibrated probabilities with optional monotonic interpolation.

    sklearn IsotonicRegression.predict is stepwise-constant, which can create
    visible probability plateaus across runners. When threshold arrays are
    available, interpolate between isotonic knots to keep monotonic ordering
    while avoiding hard probability steps.
    """
    raw = raw_probs.to_numpy(dtype=float)
    if hasattr(model, "X_thresholds_") and hasattr(model, "y_thresholds_"):
        xk = np.asarray(getattr(model, "X_thresholds_"), dtype=float)
        yk = np.asarray(getattr(model, "y_thresholds_"), dtype=float)
        if xk.size >= 2 and yk.size >= 2:
            return np.interp(raw, xk, yk, left=yk[0], right=yk[-1])
    return np.asarray(model.predict(raw), dtype=float)


def _apply_market_confirmation_race(group: pd.DataFrame, config: dict) -> pd.DataFrame:
    top_rank = int(config.get("market_confirmation_top_rank", 0) or 0)
    if top_rank <= 0:
        return group

    min_steam = float(config.get("market_confirmation_min_steam_score", 1.0))
    min_fair = float(config.get("market_confirmation_min_fair_market_prob", 1.0))
    min_gap = float(config.get("market_confirmation_min_prob_gap", 1.0))
    floor = float(config.get("market_confirmation_prob_floor", 0.0))
    fair_share = float(config.get("market_confirmation_fair_share", 0.0))
    prob_cap = float(config.get("market_confirmation_prob_cap", 1.0))

    probs = pd.to_numeric(group.get("model_prob"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    fair_market = pd.to_numeric(group.get("fair_market_prob"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    steam = pd.to_numeric(group.get("market_movement_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    market_rank = pd.to_numeric(group.get("market_rank"), errors="coerce")

    qualifies = (
        market_rank.notna()
        & (market_rank <= top_rank)
        & (steam >= min_steam)
        & (fair_market >= min_fair)
        & ((fair_market - probs) >= min_gap)
    )
    if not qualifies.any():
        return group

    target = pd.Series(
        np.minimum(prob_cap, np.maximum(floor, fair_market * fair_share)),
        index=group.index,
        dtype=float,
    )
    target = target.where(qualifies, probs)
    target = np.maximum(target, probs)

    qsum = float(target.loc[qualifies].sum())
    if qsum >= 1.0:
        adjusted = probs.copy()
        adjusted.loc[qualifies] = target.loc[qualifies] / qsum
        adjusted.loc[~qualifies] = 0.0
    else:
        other_sum = float(probs.loc[~qualifies].sum())
        adjusted = probs.copy()
        adjusted.loc[qualifies] = target.loc[qualifies]
        if other_sum > 0:
            adjusted.loc[~qualifies] = probs.loc[~qualifies] * ((1.0 - qsum) / other_sum)
        else:
            adjusted.loc[~qualifies] = 0.0

    total = float(adjusted.sum())
    if total > 0:
        adjusted = adjusted / total
    result = group.copy()
    result["model_prob"] = adjusted.clip(0.0, 1.0)
    return result


def _apply_market_confirmation(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    required = {"model_prob", "fair_market_prob", "market_movement_score", "market_rank"}
    if not required.issubset(df.columns):
        return df
    if "race_id" not in df.columns:
        return _apply_market_confirmation_race(df, config)
    parts = []
    for _, group in df.groupby("race_id", sort=False):
        parts.append(_apply_market_confirmation_race(group, config))
    return pd.concat(parts, axis=0).reindex(df.index)


def _refresh_live_rank_fields(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "model_prob" not in result.columns:
        return result
    if "race_id" not in result.columns:
        result["model_rank"] = result["model_prob"].rank(method="min", ascending=False)
        result["model_score_gap_to_next"] = (
            pd.to_numeric(result["model_prob"], errors="coerce").fillna(0.0).sort_values(ascending=False).diff(-1).abs()
        )
        return result

    result["model_rank"] = result.groupby("race_id")["model_prob"].rank(method="min", ascending=False)
    sort_cols = ["race_id", "model_prob"]
    ascending = [True, False]
    if "runner_number" in result.columns:
        sort_cols.append("runner_number")
        ascending.append(True)
    ordering = result.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    gap = ordering["model_prob"] - ordering.groupby("race_id")["model_prob"].shift(-1)
    result["model_score_gap_to_next"] = gap.reindex(result.index).fillna(0.0)
    return result


def calibrate_probabilities(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply learned calibration if a model file exists, otherwise passthrough.

    The calibration model maps raw_model_prob → actual win rate using isotonic
    regression fitted by scripts/fit_calibration.py.  After applying the
    mapping, probabilities are re-normalised within each race so that they sum
    to 1 and remain comparable with the normalised market_implied_prob used in
    edge calculation.
    """
    configured_path = config.get("calibration_model_path")
    model_path = str(configured_path) if configured_path else _default_model_path()
    model = _load_model(model_path)

    if model is None:
        return passthrough_calibration(df)

    raw = df["raw_model_prob"].clip(0.0, 1.0)
    calibrated = _predict_calibrated(model, raw)
    raw_blend = float(config.get("calibration_raw_blend", 0.0))
    raw_blend = max(0.0, min(1.0, raw_blend))
    if raw_blend > 0.0:
        calibrated = (1.0 - raw_blend) * calibrated + raw_blend * raw.to_numpy(dtype=float)
    model_prob = np.clip(calibrated, 0.0, 1.0)

    # Re-normalise within race so model_prob sums to 1 (matches market_implied_prob).
    if "race_id" in df.columns:
        model_prob_series = pd.Series(model_prob, index=df.index)
        model_prob = model_prob_series.groupby(df["race_id"]).transform(
            lambda s: s / s.sum() if s.sum() > 0 else s
        ).to_numpy()

    result = df.assign(model_prob=model_prob)
    return _refresh_live_rank_fields(_apply_market_confirmation(result, config))


def passthrough_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Copy raw_model_prob to model_prob (no adjustment).

    Used when no calibration model file is present.  scripts/fit_calibration.py
    produces the model file that replaces this with learned calibration.
    """
    result = df.copy()
    result["model_prob"] = result["raw_model_prob"]
    return result


def clear_model_cache() -> None:
    """Reset the module-level cache (used in tests)."""
    global _CALIBRATION_MODEL, _CALIBRATION_MODEL_PATH
    _CALIBRATION_MODEL = None
    _CALIBRATION_MODEL_PATH = None
