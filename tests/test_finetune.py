"""Tests for finetune filter-sweep behavior."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.finetune import FILTER_GRID, _vectorized_settle


def _base_arrays():
    return {
        "race_ids": np.array(["R1", "R2"], dtype=object),
        "sel_ids": np.array([1, 2], dtype=int),
        "live_price": np.array([5.0, 6.0], dtype=float),
        "active_field_size": np.array([8.0, 8.0], dtype=float),
        "edge": np.array([0.06, 0.06], dtype=float),
        "raw_edge": np.array([0.02, 0.02], dtype=float),
        "ev": np.array([0.10, 0.10], dtype=float),
        "model_rank": np.array([1.0, 1.0], dtype=float),
        "model_prob": np.array([0.09, 0.09], dtype=float),
        "raw_market_prob": np.array([0.20, 0.1666666667], dtype=float),
        "market_implied_prob": np.array([0.07, 0.12], dtype=float),
        "runner_number": np.array([1.0, 1.0], dtype=float),
        "has_valid_price": np.array([True, True], dtype=bool),
        "has_sparse": np.array([False, False], dtype=bool),
        "is_winner": np.array([0, 1], dtype=int),
        "coverage": np.array([1.0, 1.0], dtype=float),
        "cov_threshold": 0.8,
    }


def _base_patch():
    return {
        "min_price": 2.5,
        "max_price": 12.0,
        "min_field_size": 5,
        "max_field_size": 14,
        "min_edge": 0.04,
        "min_raw_edge": 0.02,
        "min_ev": 0.05,
        "max_model_rank": 3,
        "min_model_probability": 0.10,
        "min_model_vs_market_ratio": 0.0,
    }


def test_filter_grid_includes_value_overlay_keys():
    assert "min_model_probability" in FILTER_GRID
    assert "min_model_vs_market_ratio" in FILTER_GRID
    assert "min_raw_edge" in FILTER_GRID
    assert "min_ev" in FILTER_GRID


def test_vectorized_settle_enforces_value_overlay_filters():
    arrays = _base_arrays()
    patch = _base_patch()

    metrics, _ = _vectorized_settle(arrays, patch, min_bets=1)
    assert metrics is None  # neither row qualifies value overlay

    arrays["model_prob"][1] = 0.16
    metrics, _ = _vectorized_settle(arrays, patch, min_bets=1)
    assert metrics is not None
    assert metrics["total_bets"] == 1
