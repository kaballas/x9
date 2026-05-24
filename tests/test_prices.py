"""Tests for live_price derivation and price_quality classification."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import pytest


@pytest.fixture
def sample_price_rows():
    return [
        {"open_price": 5.0, "fluc1": 4.5, "fluc2": 4.0, "sp": 3.8},
        {"open_price": 5.0, "fluc1": 4.5, "fluc2": None, "sp": 3.8},
        {"open_price": 5.0, "fluc1": None, "fluc2": None, "sp": 3.8},
        {"open_price": None, "fluc1": None, "fluc2": None, "sp": 3.8},
    ]


def make_price_row(open_price=None, fluc1=None, fluc2=None, sp=None):
    """Helper to build a single-row DataFrame mimicking db.py output."""

    def coalesce(*values):
        for value in values:
            if value is not None and value > 0:
                return value
        return None

    live_price = coalesce(fluc2, fluc1, open_price)

    if fluc2 is not None and fluc2 > 0:
        quality = "FLUC2"
    elif fluc1 is not None and fluc1 > 0:
        quality = "FLUC1"
    elif open_price is not None and open_price > 0:
        quality = "OPEN_ONLY"
    else:
        quality = "NO_PRICE"

    return pd.DataFrame(
        [
            {
                "open_price": open_price,
                "fluc1": fluc1,
                "fluc2": fluc2,
                "sp_starting_price": sp,
                "live_price": live_price,
                "price_quality": quality,
            }
        ]
    )


def test_live_price_prefers_fluc2(sample_price_rows):
    row = make_price_row(**sample_price_rows[0])
    assert row["live_price"].iloc[0] == 4.0


def test_live_price_falls_back_to_fluc1_when_fluc2_null(sample_price_rows):
    row = make_price_row(**sample_price_rows[1])
    assert row["live_price"].iloc[0] == 4.5


def test_live_price_falls_back_to_fluc1_when_fluc2_zero():
    row = make_price_row(open_price=5.0, fluc1=4.5, fluc2=0, sp=3.8)
    assert row["live_price"].iloc[0] == 4.5


def test_live_price_falls_back_to_open_price(sample_price_rows):
    row = make_price_row(**sample_price_rows[2])
    assert row["live_price"].iloc[0] == 5.0


def test_live_price_is_null_when_all_prices_null(sample_price_rows):
    row = make_price_row(**sample_price_rows[3])
    assert row["live_price"].iloc[0] is None


def test_sp_not_used_in_live_price():
    row = make_price_row(open_price=None, fluc1=None, fluc2=None, sp=4.0)
    assert row["live_price"].iloc[0] is None


def test_sp_not_used_even_when_live_prices_exist():
    row_one = make_price_row(open_price=5.0, fluc1=None, fluc2=None, sp=3.0)
    row_two = make_price_row(open_price=5.0, fluc1=None, fluc2=None, sp=99.0)
    assert row_one["live_price"].iloc[0] == row_two["live_price"].iloc[0]


def test_price_quality_fluc2():
    row = make_price_row(open_price=5.0, fluc1=4.5, fluc2=4.0)
    assert row["price_quality"].iloc[0] == "FLUC2"


def test_price_quality_fluc1():
    row = make_price_row(open_price=5.0, fluc1=4.5, fluc2=None)
    assert row["price_quality"].iloc[0] == "FLUC1"


def test_price_quality_open_only():
    row = make_price_row(open_price=5.0, fluc1=None, fluc2=None)
    assert row["price_quality"].iloc[0] == "OPEN_ONLY"


def test_price_quality_no_price():
    row = make_price_row(open_price=None, fluc1=None, fluc2=None)
    assert row["price_quality"].iloc[0] == "NO_PRICE"


def test_price_quality_no_price_when_sp_only():
    row = make_price_row(open_price=None, fluc1=None, fluc2=None, sp=5.0)
    assert row["price_quality"].iloc[0] == "NO_PRICE"
