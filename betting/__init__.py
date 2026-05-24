"""Betting framework package."""

from .config import CONFIG
from .backtest import run_backtest
from .live_candidates import run_live_candidates

__all__ = ["CONFIG", "run_backtest", "run_live_candidates"]
