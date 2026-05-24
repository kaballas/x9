"""Tests for backtest reporting output."""

from argparse import Namespace
from pathlib import Path
import sys

import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from betting.config import CONFIG
from betting.reporting import build_report, print_report
from scripts.run_backtest import build_runtime_config


@pytest.fixture
def base_config():
    return CONFIG.copy()


def make_report_df():
    return pd.DataFrame(
        {
            "start_time_iso": ["2024-01-03T12:00:00", "2024-01-01T12:00:00", "2024-01-02T12:00:00"],
            "live_price": [3.0, 5.0, 6.0],
            "is_winner": [1, 0, 0],
            "stake": [1.0, 1.0, 1.0],
            "profit": [2.0, -1.0, -1.0],
            "edge": [0.1, 0.2, 0.3],
            "model_prob": [0.12, 0.09, 0.15],
            "market_implied_prob": [0.10, 0.12, 0.11],
            "track": ["Track A", "Track A", "Track B"],
            "distance_m": [1200, 1200, 1600],
            "condition": ["Good", "Good", "Soft"],
            "field_size": [8, 8, 10],
            "market_rank": [1, 2, 1],
            "model_rank": [1, 2, 3],
            "price_quality": ["FLUC2", "FLUC2", "OPEN_ONLY"],
        }
    )


def test_build_report_adds_flat_and_fibonacci_staking_metrics(base_config):
    report = build_report(make_report_df(), base_config)

    flat = report["staking_comparison"]["flat"]
    fibonacci = report["staking_comparison"]["fibonacci"]

    assert flat == pytest.approx({"staked": 3.0, "profit": 0.0, "roi": 0.0})
    assert fibonacci == pytest.approx(
        {"staked": 6.0, "profit": 3.0, "roi": 0.5, "max_level": 2, "max_stake": 3.0}
    )


def test_print_report_shows_staking_comparison(base_config, capsys):
    report = build_report(make_report_df(), base_config)

    print_report(report)

    out = capsys.readouterr().out
    assert "Staking comparison:" in out
    assert "Flat stake  ROI: 0.0%  profit: 0.00  staked: 3.00" in out
    assert "Fibonacci   ROI: 50.0%  profit: 3.00  staked: 6.00  max_level: 2  max_stake: 3.00 units" in out
    assert "MODEL PROBABILITY CALIBRATION" in out


def test_build_report_uses_calibration_source_df(base_config):
    result_df = make_report_df()
    calibration_source_df = result_df.copy()
    calibration_source_df.loc[:, "model_prob"] = [0.03, 0.06, 0.12]
    calibration_source_df.loc[:, "raw_market_prob"] = [0.02, 0.05, 0.10]

    report = build_report(result_df, base_config, calibration_source_df=calibration_source_df)
    buckets = set(report["model_prob_calibration"]["model_prob_bucket"].tolist())

    assert "0-5%" in buckets
    assert "5-7%" in buckets
    assert "10-15%" in buckets


def test_build_runtime_config_applies_staking_mode():
    args = Namespace(
        db=None,
        output_dir="outputs/backtests",
        min_edge=None,
        min_price=None,
        max_price=None,
        min_field_size=None,
        max_field_size=None,
        staking="fibonacci",
    )

    cfg = build_runtime_config(args)

    assert cfg["staking_mode"] == "fibonacci"
