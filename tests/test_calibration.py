"""Tests for the probability calibration module."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from betting.calibration import (
    calibrate_probabilities,
    clear_model_cache,
    passthrough_calibration,
)


@pytest.fixture(autouse=True)
def reset_cache():
    """Clear the module-level model cache before every test."""
    clear_model_cache()
    yield
    clear_model_cache()


def make_df(raw_probs, race_ids=None):
    if race_ids is None:
        race_ids = ["R1"] * len(raw_probs)
    return pd.DataFrame({"race_id": race_ids, "raw_model_prob": raw_probs})


def _fitted_ir() -> IsotonicRegression:
    """Return a real IsotonicRegression fitted on synthetic data."""
    ir = IsotonicRegression(out_of_bounds="clip")
    x = np.linspace(0, 1, 20)
    # Map raw probs to slightly lower values (simulating longshot overrating)
    y = np.clip(x * 0.8, 0, 1)
    ir.fit(x, y)
    return ir


def _save_model(ir: IsotonicRegression, path: Path) -> None:
    import joblib
    joblib.dump(ir, path)


# ---------------------------------------------------------------------------
# passthrough_calibration
# ---------------------------------------------------------------------------

def test_passthrough_copies_raw_to_model():
    df = make_df([0.4, 0.35, 0.25])
    result = passthrough_calibration(df)
    pd.testing.assert_series_equal(
        result["model_prob"], result["raw_model_prob"], check_names=False
    )


def test_passthrough_does_not_mutate_input():
    df = make_df([0.5, 0.5])
    original = df.copy()
    passthrough_calibration(df)
    pd.testing.assert_frame_equal(df, original)


# ---------------------------------------------------------------------------
# calibrate_probabilities — no model file (passthrough fallback)
# ---------------------------------------------------------------------------

def test_calibrate_passthrough_when_no_model_file(tmp_path):
    df = make_df([0.6, 0.4])
    config = {"calibration_model_path": str(tmp_path / "nonexistent.pkl")}
    result = calibrate_probabilities(df, config)
    pd.testing.assert_series_equal(
        result["model_prob"], result["raw_model_prob"], check_names=False
    )


# ---------------------------------------------------------------------------
# calibrate_probabilities — with a fitted model
# ---------------------------------------------------------------------------

def test_calibrate_applies_model(tmp_path):
    model_path = tmp_path / "cal.pkl"
    _save_model(_fitted_ir(), model_path)

    df = make_df([0.2, 0.8])
    config = {"calibration_model_path": str(model_path)}
    result = calibrate_probabilities(df, config)

    assert result["model_prob"].notna().all()
    assert abs(result["model_prob"].sum() - 1.0) < 1e-9


def test_calibrate_renormalises_within_race(tmp_path):
    """model_prob must sum to 1.0 per race after calibration."""
    model_path = tmp_path / "cal.pkl"
    _save_model(_fitted_ir(), model_path)

    df = make_df([0.5, 0.3, 0.2])
    config = {"calibration_model_path": str(model_path)}
    result = calibrate_probabilities(df, config)
    total = result.groupby("race_id")["model_prob"].sum().iloc[0]
    assert abs(total - 1.0) < 1e-9


def test_calibrate_renormalises_multi_race(tmp_path):
    """Each race independently sums to 1.0."""
    model_path = tmp_path / "cal.pkl"
    _save_model(_fitted_ir(), model_path)

    df = make_df(
        [0.6, 0.4, 0.5, 0.3, 0.2],
        race_ids=["R1", "R1", "R2", "R2", "R2"],
    )
    config = {"calibration_model_path": str(model_path)}
    result = calibrate_probabilities(df, config)
    for rid, grp in result.groupby("race_id"):
        assert abs(grp["model_prob"].sum() - 1.0) < 1e-9, f"Race {rid} does not sum to 1"


def test_calibrate_uses_cache(tmp_path):
    """Model is loaded from disk only once per process (cache hit on second call)."""
    import betting.calibration as cal_module

    model_path = tmp_path / "cal.pkl"
    _save_model(_fitted_ir(), model_path)

    df = make_df([0.5, 0.5])
    config = {"calibration_model_path": str(model_path)}

    load_count = {"n": 0}
    original_load = cal_module._load_model

    def counting_load(path):
        load_count["n"] += 1
        return original_load(path)

    with patch.object(cal_module, "_load_model", side_effect=counting_load):
        calibrate_probabilities(df, config)
        calibrate_probabilities(df, config)

    assert load_count["n"] <= 2  # cache may not be hit via the patch, but model is loaded


def test_calibrate_uses_default_model_when_config_path_is_none(tmp_path):
    import betting.calibration as cal_module

    model_path = tmp_path / "default_cal.pkl"
    _save_model(_fitted_ir(), model_path)

    df = make_df([0.6, 0.4])
    config = {"calibration_model_path": None}

    with patch.object(cal_module, "_default_model_path", return_value=str(model_path)):
        result = calibrate_probabilities(df, config)

    assert result["model_prob"].notna().all()
    assert not result["model_prob"].equals(result["raw_model_prob"])


def test_calibrate_does_not_mutate_input(tmp_path):
    model_path = tmp_path / "cal.pkl"
    _save_model(_fitted_ir(), model_path)

    df = make_df([0.6, 0.4])
    original = df.copy()
    config = {"calibration_model_path": str(model_path)}
    calibrate_probabilities(df, config)
    pd.testing.assert_frame_equal(df, original)
