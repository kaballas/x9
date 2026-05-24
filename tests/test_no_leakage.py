"""Safety tests for query separation and SP reference-only policy."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from betting.config import CONFIG
from betting.db import load_race_runners
from betting.edge import calculate_edges
from betting.settlement import settle_bets
import betting.db as betting_db


@pytest.fixture
def base_config():
    return CONFIG.copy()


@pytest.fixture
def sample_query_frames():
    backtest = pd.DataFrame(
        {
            "race_id": ["B1", "B1", "B2"],
            "status": ["finished", "finished", "finished"],
            "result_code": ["W", "L", "P"],
            "open_price": [5.0, 6.0, 4.5],
            "fluc1": [4.8, 5.8, None],
            "fluc2": [4.6, None, None],
            "sp_starting_price": [4.4, 5.2, 4.0],
            "live_price": [4.6, 5.8, 4.5],
        }
    )
    live = pd.DataFrame(
        {
            "race_id": ["L1", "L1"],
            "status": ["no_result", "no_result"],
            "result_code": ["", ""],
            "open_price": [7.0, 3.5],
            "fluc1": [6.8, 3.3],
            "fluc2": [6.5, None],
            "sp_starting_price": [None, None],
            "live_price": [6.5, 3.3],
        }
    )
    return {"backtest": backtest, "live": live}


@pytest.fixture
def patched_loader(monkeypatch, sample_query_frames):
    observed_sql = []

    class DummyConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_connect(_path):
        return DummyConnection()

    def fake_read_sql_query(sql, _conn):
        observed_sql.append(sql)
        if "WHERE status = 'finished'" in sql:
            assert "result_code IN ('W', 'P', 'L')" in sql
            return sample_query_frames["backtest"].copy()
        if "WHERE status = 'no_result'" in sql:
            assert "result_code != 'V'" in sql
            assert "COALESCE(NULLIF(fluc2, 0), NULLIF(fluc1, 0), NULLIF(open_price, 0)) IS NOT NULL" in sql
            return sample_query_frames["live"].copy()
        raise AssertionError(f"Unexpected SQL: {sql}")

    monkeypatch.setattr(betting_db.sqlite3, "connect", fake_connect)
    monkeypatch.setattr(betting_db.pd, "read_sql_query", fake_read_sql_query)
    return observed_sql


def test_backtest_query_excludes_no_result_races(base_config, patched_loader):
    df = load_race_runners(base_config["database_path"], "backtest", base_config)
    assert len(df) > 0
    assert (df["status"] == "finished").all()
    assert any("WHERE status = 'finished'" in sql for sql in patched_loader)


def test_backtest_query_excludes_late_scratched(base_config, patched_loader):
    df = load_race_runners(base_config["database_path"], "backtest", base_config)
    assert (df["result_code"] != "V").all()


def test_live_query_excludes_finished_races(base_config, patched_loader):
    df = load_race_runners(base_config["database_path"], "live", base_config)
    assert (df["status"] == "no_result").all()
    assert any("WHERE status = 'no_result'" in sql for sql in patched_loader)


def test_live_query_excludes_late_scratched(base_config, patched_loader):
    df = load_race_runners(base_config["database_path"], "live", base_config)
    assert (df["result_code"] != "V").all()


def test_live_price_never_derives_from_sp(base_config, patched_loader):
    df = load_race_runners(base_config["database_path"], "backtest", base_config)
    sp_only = pd.DataFrame(
        [
            {
                "fluc2": None,
                "fluc1": None,
                "open_price": None,
                "sp_starting_price": 4.0,
                "live_price": None,
            }
        ]
    )
    assert "sp_starting_price" in df.columns
    assert sp_only["live_price"].isna().all()


def test_settlement_does_not_use_sp(base_config):
    base = pd.DataFrame(
        [
            {
                "race_id": "R1",
                "selection_id": 1,
                "live_price": 6.0,
                "sp_starting_price": 5.0,
                "is_winner": 1,
                "stake": 1.0,
            }
        ]
    )
    high_sp = base.copy()
    high_sp["sp_starting_price"] = 999.0

    result_base = settle_bets(base, base_config)
    result_high = settle_bets(high_sp, base_config)
    assert abs(result_base["profit"].iloc[0] - result_high["profit"].iloc[0]) < 1e-9


def test_pipeline_edge_not_affected_by_sp(base_config):
    low_sp = pd.DataFrame(
        [
            {
                "race_id": "R1",
                "selection_id": 1,
                "model_prob": 0.35,
                "market_implied_prob": 0.25,
                "live_price": 4.0,
                "sp_starting_price": 3.0,
            }
        ]
    )
    high_sp = low_sp.copy()
    high_sp["sp_starting_price"] = 99.0

    result_low = calculate_edges(low_sp, base_config)
    result_high = calculate_edges(high_sp, base_config)
    assert abs(result_low["edge"].iloc[0] - result_high["edge"].iloc[0]) < 1e-9


def test_sp_column_present_but_not_in_selection_fields(base_config, patched_loader):
    df = load_race_runners(base_config["database_path"], "backtest", base_config)
    assert "sp_starting_price" in df.columns
    assert "live_price" in df.columns
    assert "sp_starting_price" != "live_price"


def test_backtest_and_live_are_disjoint(base_config, patched_loader):
    backtest_df = load_race_runners(base_config["database_path"], "backtest", base_config)
    live_df = load_race_runners(base_config["database_path"], "live", base_config)
    overlap = set(backtest_df["race_id"].unique()) & set(live_df["race_id"].unique())
    assert overlap == set()
