"""Tests for validation helpers."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from betting.validation import check_field_size


def test_check_field_size_uses_active_field_size_not_declared():
    df = pd.DataFrame(
        {
            "field_size": [20, 20],
            "active_field_size": [10, 10],
        }
    )

    result = check_field_size(df, {"min_field_size": 7, "max_field_size": 14})

    assert result["is_valid_field_size"].tolist() == [True, True]


def test_check_field_size_declared_too_large_but_active_in_range():
    df = pd.DataFrame(
        {
            "field_size": [20],
            "active_field_size": [10],
        }
    )

    result = check_field_size(df, {"min_field_size": 5, "max_field_size": 12})

    assert result.loc[0, "is_valid_field_size"]


def test_check_field_size_invalid_when_active_field_too_small():
    df = pd.DataFrame(
        {
            "field_size": [20],
            "active_field_size": [3],
        }
    )

    result = check_field_size(df, {"min_field_size": 5, "max_field_size": 12})

    assert not result.loc[0, "is_valid_field_size"]
