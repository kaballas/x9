"""Live candidate entry point built on the shared pipeline."""

from __future__ import annotations

import pandas as pd

from .backtest import run_pipeline
from .config import CONFIG
from .db import load_race_runners


def run_live_candidates(config: dict | None = None) -> pd.DataFrame:
    """Load unresolved races and return filtered candidates sorted by edge."""
    runtime_config = dict(CONFIG if config is None else config)
    df = load_race_runners(runtime_config["database_path"], "live", runtime_config)
    candidates = run_pipeline(df, runtime_config)
    if candidates.empty:
        return candidates
    sort_cols = [c for c in ["ev", "raw_edge", "model_rank", "runner_number"] if c in candidates.columns]
    ascending = [False, False, True, True][: len(sort_cols)]
    return candidates.sort_values(sort_cols, ascending=ascending)
