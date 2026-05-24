#!/usr/bin/env python3
"""
Grid-search over strategy parameters.

Sweeps every combination of the tunable thresholds and scoring weights,
runs the shared backtest pipeline for each, and prints a ranked results table
sorted by ROI (descending).  Only combinations that produce at least
--min-bets bets are included in the ranking.

Performance design
------------------
- In `filters` mode: data is loaded and fully scored ONCE (features, scoring,
  probs, edges).  Only the cheap filter+settlement step is repeated per combo.
  This makes a 1,700+ combo filter sweep complete in a few minutes.
- In `weights` mode: data is loaded once; features → scoring → probs → edges
  are re-run per weight set (10 combos), then default filters are applied once.
- In `full` mode: each unique weight set is pre-scored once, then all filter
  combos are swept against that scored dataset.

Deduplication
-------------
Different parameter combos that produce the exact same set of bets are
collapsed into one row.  The output shows:
  - duplicate_combos: how many combos hit that identical bet set
  - non_binding_params: which parameters were irrelevant for that set

Sample-size warning
-------------------
Any result with fewer than 100 bets is flagged LOW (<100 bets).
ROI figures on small samples are statistically unreliable.

Usage
-----
# Sweep filter thresholds (fastest — default)
python3 scripts/finetune.py

# Sweep scoring weights only
python3 scripts/finetune.py --mode weights

# Sweep everything (filter × weight combos)
python3 scripts/finetune.py --mode full

# Restrict minimum bets to make results more meaningful
python3 scripts/finetune.py --min-bets 50

# Export deduplicated results to CSV
python3 scripts/finetune.py --output outputs/finetune/results.csv

# Show only the top N rows, sort by profit
python3 scripts/finetune.py --top 20 --sort-by profit
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import itertools
import sqlite3
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from betting.config import CONFIG
from betting.db import load_draw_bias_table, load_jockey_stats_table, load_race_runners, load_trainer_stats_table
from betting.features import build_features
from betting.meta_model import add_meta_model_signal
from betting.scoring import score_runners
from betting.probabilities import assign_probabilities
from betting.calibration import calibrate_probabilities
from betting.edge import calculate_edges
from betting.filters import apply_filters
from betting.staking import assign_stakes
from betting.settlement import settle_bets
from betting.reporting import summary_metrics
from betting.validation import validate_input

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

FILTER_GRID: dict[str, list] = {
    "min_price":        [2.0, 5, 7],
    "max_price":        [10.0, 12.0, 13.0, 14.0, 15.0],
    "min_field_size":   [5, 10, 11],
    "max_field_size":   [12, 14, 16],
    "min_raw_edge":     [0.02],
    "min_ev":           [0.05],
    "min_score_gap_to_next": [0.00, 0.005, 0.010],
    "max_model_rank":   [1, 2, 3 , 4 , 5],
    "min_model_probability": [0.08, 0.10, 0.12],
    "min_model_vs_market_ratio": [0.1],
}

LOW_SAMPLE_THRESHOLD = 100  # Combinations below this bet count get a warning

# Each tuple must sum to 1.0.
# Swept components: speed_rating, recent_form, suitability, connections,
# market_sanity, margin, freshness, class, draw_bias, jockey, trainer
WEIGHT_GRID: list[tuple[float, ...]] = [
    (0.28, 0.22, 0.15, 0.06, 0.01, 0.00, 0.10, 0.04, 0.02, 0.05, 0.04, 0.03),  # default
    (0.32, 0.22, 0.12, 0.05, 0.01, 0.00, 0.12, 0.01, 0.01, 0.05, 0.06, 0.03),  # speed+margin+jockey heavy
    (0.25, 0.28, 0.15, 0.06, 0.01, 0.00, 0.08, 0.03, 0.02, 0.05, 0.04, 0.03),  # form heavy
    (0.28, 0.20, 0.18, 0.06, 0.01, 0.00, 0.10, 0.03, 0.02, 0.05, 0.04, 0.03),  # suitability heavy
    (0.28, 0.22, 0.13, 0.06, 0.01, 0.00, 0.10, 0.06, 0.02, 0.05, 0.04, 0.03),  # freshness heavy
    (0.28, 0.22, 0.13, 0.06, 0.01, 0.00, 0.10, 0.01, 0.07, 0.05, 0.04, 0.03),  # class heavy
    (0.30, 0.25, 0.12, 0.04, 0.01, 0.00, 0.12, 0.01, 0.02, 0.05, 0.05, 0.03),  # speed+form balanced
    (0.25, 0.20, 0.15, 0.04, 0.01, 0.00, 0.15, 0.06, 0.02, 0.05, 0.04, 0.03),  # margin+freshness
    (0.30, 0.20, 0.15, 0.09, 0.01, 0.00, 0.08, 0.03, 0.02, 0.05, 0.04, 0.03),  # connections boost
    (0.28, 0.22, 0.15, 0.04, 0.01, 0.00, 0.12, 0.04, 0.02, 0.05, 0.04, 0.03),  # margin up
    (0.35, 0.18, 0.12, 0.06, 0.01, 0.00, 0.10, 0.04, 0.02, 0.05, 0.04, 0.03),  # speed dominant
    (0.22, 0.28, 0.15, 0.06, 0.01, 0.00, 0.10, 0.04, 0.02, 0.05, 0.04, 0.03),  # form dominant
    (0.24, 0.18, 0.12, 0.05, 0.04, 0.00, 0.09, 0.03, 0.01, 0.04, 0.04, 0.03),  # market aware
    (0.22, 0.17, 0.10, 0.05, 0.06, 0.00, 0.09, 0.03, 0.01, 0.04, 0.04, 0.03),  # stronger market aware
    (0.24, 0.16, 0.10, 0.05, 0.05, 0.00, 0.08, 0.03, 0.01, 0.04, 0.04, 0.04),  # market+trainer tempered
    (0.23, 0.16, 0.10, 0.05, 0.04, 0.00, 0.08, 0.02, 0.01, 0.04, 0.04, 0.03),  # room for higher steam
]

# Additional steam-weight candidates swept across each base profile.
STEAM_WEIGHT_GRID: list[float] = [0.00, 0.03, 0.05, 0.08, 0.12, 0.16]

WEIGHT_KEYS = [
    "weight_speed_rating",
    "weight_recent_form",
    "weight_suitability",
    "weight_connections",
    "weight_market_sanity",
    "weight_steam",
    "weight_margin",
    "weight_freshness",
    "weight_class",
    "weight_draw_bias",
    "weight_jockey",
    "weight_trainer",
]

MARKET_CONFIRMATION_KEYS = [
    "market_confirmation_top_rank",
    "market_confirmation_min_steam_score",
    "market_confirmation_min_fair_market_prob",
    "market_confirmation_min_prob_gap",
    "market_confirmation_prob_floor",
    "market_confirmation_fair_share",
    "market_confirmation_prob_cap",
]

MARKET_CONFIRMATION_PATCHES: list[dict[str, float | int]] = [
    {
        "market_confirmation_top_rank": 0,
        "market_confirmation_min_steam_score": 1.0,
        "market_confirmation_min_fair_market_prob": 1.0,
        "market_confirmation_min_prob_gap": 1.0,
        "market_confirmation_prob_floor": 0.0,
        "market_confirmation_fair_share": 0.0,
        "market_confirmation_prob_cap": 1.0,
    },  # disabled baseline
    {
        "market_confirmation_top_rank": 3,
        "market_confirmation_min_steam_score": 0.95,
        "market_confirmation_min_fair_market_prob": 0.15,
        "market_confirmation_min_prob_gap": 0.08,
        "market_confirmation_prob_floor": 0.12,
        "market_confirmation_fair_share": 1.00,
        "market_confirmation_prob_cap": 0.20,
    },  # current default
    {
        "market_confirmation_top_rank": 3,
        "market_confirmation_min_steam_score": 0.90,
        "market_confirmation_min_fair_market_prob": 0.14,
        "market_confirmation_min_prob_gap": 0.06,
        "market_confirmation_prob_floor": 0.12,
        "market_confirmation_fair_share": 1.00,
        "market_confirmation_prob_cap": 0.20,
    },  # earlier trigger
    {
        "market_confirmation_top_rank": 3,
        "market_confirmation_min_steam_score": 0.90,
        "market_confirmation_min_fair_market_prob": 0.14,
        "market_confirmation_min_prob_gap": 0.06,
        "market_confirmation_prob_floor": 0.15,
        "market_confirmation_fair_share": 1.00,
        "market_confirmation_prob_cap": 0.22,
    },  # stronger floor
    {
        "market_confirmation_top_rank": 4,
        "market_confirmation_min_steam_score": 0.88,
        "market_confirmation_min_fair_market_prob": 0.12,
        "market_confirmation_min_prob_gap": 0.05,
        "market_confirmation_prob_floor": 0.12,
        "market_confirmation_fair_share": 0.95,
        "market_confirmation_prob_cap": 0.20,
    },  # wider rank coverage
]


# ---------------------------------------------------------------------------
# Pre-scoring helpers (the expensive work, done once per weight set)
# ---------------------------------------------------------------------------

def _prescored_df(raw_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Run everything up to and including edge calculation."""
    with sqlite3.connect(cfg["database_path"]) as conn:
        draw_bias_df = load_draw_bias_table(conn)
        jockey_stats_df = load_jockey_stats_table(conn)
        trainer_stats_df = load_trainer_stats_table(conn)
    df = validate_input(raw_df, cfg)
    df = build_features(df, cfg, draw_bias_df, jockey_stats_df, trainer_stats_df)
    df = add_meta_model_signal(df, cfg, draw_bias_df, jockey_stats_df, trainer_stats_df)
    return _score_from_feature_frame(df, cfg)


def _score_from_feature_frame(feature_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Run the score -> probability -> calibration -> edge steps from a prepared feature frame."""
    df = feature_df.copy()
    df = score_runners(df, cfg)
    df = assign_probabilities(df, cfg)
    df = calibrate_probabilities(df, cfg)
    df = calculate_edges(df, cfg)
    return df


def _prepare_feature_frame(raw_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Build validated features and meta-model inputs once for repeated weight rescoring."""
    with sqlite3.connect(cfg["database_path"]) as conn:
        draw_bias_df = load_draw_bias_table(conn)
        jockey_stats_df = load_jockey_stats_table(conn)
        trainer_stats_df = load_trainer_stats_table(conn)
    df = validate_input(raw_df, cfg)
    df = build_features(df, cfg, draw_bias_df, jockey_stats_df, trainer_stats_df)
    return add_meta_model_signal(df, cfg, draw_bias_df, jockey_stats_df, trainer_stats_df)


def _precompute_arrays(scored_df: pd.DataFrame, base_cfg: dict) -> dict:
    """
    Extract numpy arrays and pre-compute per-row price_coverage once.
    Avoids re-running the groupby inside filter_no_live_price on every combo.
    Also applies active_field_size directly from the column (not pre-baked flag),
    so min/max_field_size sweeps are meaningful.
    """
    import numpy as np

    race_sizes = scored_df.groupby("race_id").size()
    priced_sizes = scored_df.groupby("race_id")["has_valid_live_price"].sum()
    coverage_by_race = (priced_sizes / race_sizes).to_dict()

    race_ids = scored_df["race_id"].to_numpy()
    sel_ids = scored_df["selection_id"].to_numpy()
    live_price = pd.to_numeric(scored_df["live_price"], errors="coerce").to_numpy(dtype=float)
    active_field_size = pd.to_numeric(scored_df["active_field_size"], errors="coerce").to_numpy(dtype=float)
    edge_arr = pd.to_numeric(scored_df["edge"], errors="coerce").to_numpy(dtype=float)
    raw_edge_src = scored_df["raw_edge"] if "raw_edge" in scored_df.columns else pd.Series(np.nan, index=scored_df.index)
    raw_edge_arr = pd.to_numeric(raw_edge_src, errors="coerce").to_numpy(dtype=float)
    ev_src = scored_df["ev"] if "ev" in scored_df.columns else pd.Series(np.nan, index=scored_df.index)
    ev_arr = pd.to_numeric(ev_src, errors="coerce").to_numpy(dtype=float)
    model_rank = pd.to_numeric(scored_df["model_rank"], errors="coerce").to_numpy(dtype=float)
    model_prob = pd.to_numeric(scored_df["model_prob"], errors="coerce").to_numpy(dtype=float)
    market_implied_prob = pd.to_numeric(scored_df["market_implied_prob"], errors="coerce").to_numpy(dtype=float)
    raw_market_src = scored_df["raw_market_prob"] if "raw_market_prob" in scored_df.columns else pd.Series(np.nan, index=scored_df.index)
    raw_market_prob = pd.to_numeric(raw_market_src, errors="coerce").to_numpy(dtype=float)
    rn_col = scored_df["runner_number"] if "runner_number" in scored_df.columns else pd.Series(0, index=scored_df.index)
    runner_number = pd.to_numeric(rn_col, errors="coerce").to_numpy(dtype=float)
    score_gap_to_next = pd.to_numeric(
        scored_df.get("model_score_gap_to_next", pd.Series(0.0, index=scored_df.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)
    has_valid_price = scored_df["has_valid_live_price"].to_numpy(dtype=bool)
    has_sparse = scored_df["has_sparse_recent_form"].to_numpy(dtype=bool)
    race_integrity = (
        scored_df["race_integrity_ok"].fillna(False).astype(bool).to_numpy()
        if "race_integrity_ok" in scored_df.columns
        else np.ones(len(scored_df), dtype=bool)
    )
    is_winner = pd.to_numeric(scored_df["is_winner"], errors="coerce").fillna(0).to_numpy(dtype=int)
    coverage = np.array([coverage_by_race.get(r, 0.0) for r in race_ids])

    return {
        "race_ids": race_ids,
        "sel_ids": sel_ids,
        "live_price": live_price,
        "active_field_size": active_field_size,
        "edge": edge_arr,
        "raw_edge": raw_edge_arr,
        "ev": ev_arr,
        "model_rank": model_rank,
        "model_prob": model_prob,
        "market_implied_prob": market_implied_prob,
        "raw_market_prob": raw_market_prob,
        "runner_number": runner_number,
        "score_gap_to_next": score_gap_to_next,
        "has_valid_price": has_valid_price,
        "has_sparse": has_sparse,
        "race_integrity": race_integrity,
        "is_winner": is_winner,
        "coverage": coverage,
        "cov_threshold": base_cfg["exclude_race_if_price_coverage_below"],
        "allow_multiple_bets_per_race": bool(base_cfg.get("allow_multiple_bets_per_race", False)),
    }


def _vectorized_settle(arrays: dict, patch: dict, min_bets: int) -> "tuple[dict, frozenset] | tuple[None, None]":
    """
    Apply filters and compute settlement using numpy arrays only.
    ~50x faster than the DataFrame path — no .copy(), no groupby per combo.
    active field-size filtering is applied directly from the column so sweeping
    min/max_field_size actually changes the bet set.
    """
    import numpy as np

    lp = arrays["live_price"]
    nan_safe = ~np.isnan(lp)

    mask = (
        arrays.get("race_integrity", np.ones(len(lp), dtype=bool)) &
        arrays["has_valid_price"] &
        nan_safe &
        (arrays["coverage"] >= arrays["cov_threshold"]) &
        (lp >= patch["min_price"]) & (lp <= patch["max_price"]) &
        (arrays["active_field_size"] >= patch["min_field_size"]) &
        (arrays["active_field_size"] <= patch["max_field_size"]) &
        (~np.isnan(arrays["raw_edge"])) & (arrays["raw_edge"] >= patch["min_raw_edge"]) &
        (~np.isnan(arrays["ev"])) & (arrays["ev"] >= patch["min_ev"]) &
        (~np.isnan(arrays["model_rank"])) & (arrays["model_rank"] <= patch["max_model_rank"]) &
        (~np.isnan(arrays["model_prob"])) & (arrays["model_prob"] >= patch["min_model_probability"]) &
        (~np.isnan(arrays.get("score_gap_to_next", np.zeros(len(lp), dtype=float))))
        & (arrays.get("score_gap_to_next", np.zeros(len(lp), dtype=float)) >= patch.get("min_score_gap_to_next", 0.0)) &
        (~np.isnan(arrays["raw_market_prob"])) & (arrays["raw_market_prob"] > 0.0) &
        (~arrays["has_sparse"])
    )
    if patch["min_model_vs_market_ratio"] > 0:
        mask = mask & (
            arrays["model_prob"] >= arrays["raw_market_prob"] * patch["min_model_vs_market_ratio"]
        )

    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None, None

    if arrays.get("allow_multiple_bets_per_race", False):
        final_indices = indices
    else:
        # filter_one_per_race: keep highest EV, then raw edge, then rank, then runner number.
        best: dict = {}
        for idx in indices:
            rid = arrays["race_ids"][idx]
            ev = arrays["ev"][idx]
            re = arrays["raw_edge"][idx]
            r = arrays["model_rank"][idx]
            rn = arrays["runner_number"][idx]
            if rid not in best:
                best[rid] = (ev, re, r, rn, int(idx))
            else:
                pev, pre, pr, prn, _ = best[rid]
                if (
                    ev > pev
                    or (ev == pev and re > pre)
                    or (ev == pev and re == pre and r < pr)
                    or (ev == pev and re == pre and r == pr and rn < prn)
                ):
                    best[rid] = (ev, re, r, rn, int(idx))

        final_indices = [v[4] for v in best.values()]
    n_bets = len(final_indices)
    if n_bets < min_bets:
        return None, None

    prices = lp[final_indices]
    winners = arrays["is_winner"][final_indices]
    profits = np.where(winners == 1, prices - 1.0, -1.0)

    total_profit = float(profits.sum())
    n_wins = int(winners.sum())
    fp = frozenset(zip(arrays["race_ids"][final_indices], arrays["sel_ids"][final_indices]))

    m = {
        "total_bets": n_bets,
        "wins": n_wins,
        "total_staked": float(n_bets),
        "total_profit": round(total_profit, 3),
        "roi": total_profit / n_bets,
        "strike_rate": n_wins / n_bets,
        "avg_price": float(prices.mean()),
        "avg_edge": float(arrays["edge"][final_indices].mean()),
    }
    return m, fp


def _bet_fingerprint(settled_df: pd.DataFrame) -> frozenset:
    """A hashable identity for exactly which bets were selected."""
    return frozenset(zip(settled_df["race_id"], settled_df["selection_id"]))


def _apply_and_settle(
    scored_df: pd.DataFrame, cfg: dict
) -> "tuple[dict, frozenset] | tuple[None, None]":
    """Apply filters + stake + settle via full DataFrame path (used for weights/full modes)."""
    try:
        with redirect_stdout(io.StringIO()):
            df = apply_filters(scored_df.copy(), cfg)
            df = assign_stakes(df, cfg)
            df = settle_bets(df, cfg)
    except Exception:
        return None, None
    if df.empty:
        return None, None
    return summary_metrics(df), _bet_fingerprint(df)


# ---------------------------------------------------------------------------
# Combo builders
# ---------------------------------------------------------------------------

def _build_filter_combos() -> list[dict]:
    keys = list(FILTER_GRID.keys())
    values = list(FILTER_GRID.values())
    combos = []
    for vals in itertools.product(*values):
        patch = dict(zip(keys, vals))
        if patch["min_price"] >= patch["max_price"]:
            continue
        if patch["min_field_size"] >= patch["max_field_size"]:
            continue
        combos.append(patch)
    return combos


def _build_weight_patches() -> list[dict]:
    patches: list[dict] = []
    for w in WEIGHT_GRID:
        base = dict(zip(WEIGHT_KEYS, w))
        non_steam_total = sum(
            float(base[key]) for key in WEIGHT_KEYS if key != "weight_steam"
        )
        if non_steam_total <= 0:
            continue
        for steam_weight in STEAM_WEIGHT_GRID:
            steam = float(steam_weight)
            if steam < 0.0 or steam >= 1.0:
                continue
            scale = (1.0 - steam) / non_steam_total
            patch = {}
            for key in WEIGHT_KEYS:
                if key == "weight_steam":
                    patch[key] = steam
                else:
                    patch[key] = float(base[key]) * scale
            for confirmation_patch in MARKET_CONFIRMATION_PATCHES:
                patches.append({**patch, **confirmation_patch})
    return patches


# ---------------------------------------------------------------------------
# Deduplication: collapse identical bet sets, identify binding parameters
# ---------------------------------------------------------------------------

def _deduplicate(rows: list[dict]) -> pd.DataFrame:
    """
    Group rows that produced the same bet set (same fingerprint).
    For each unique bet set:
    - Keep only the first (tightest/most conservative) parameter combo.
    - Add a 'duplicate_combos' count showing how many combos hit the same bets.
    - Add 'non_binding' listing parameters that didn't matter.
    """
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    fp_col = "__fp__"

    # Group by fingerprint
    groups: dict[frozenset, list[int]] = {}
    for i, row in enumerate(rows):
        fp = row.pop(fp_col, None)
        if fp is not None:
            groups.setdefault(fp, []).append(i)

    # For each group, find which filter params vary (non-binding)
    filter_param_keys = list(FILTER_GRID.keys())
    deduped = []
    for fp, indices in groups.items():
        group_rows = [rows[i] for i in indices]
        representative = group_rows[0].copy()
        representative["duplicate_combos"] = len(indices)

        if len(indices) > 1:
            non_binding = []
            for key in filter_param_keys:
                vals = {r[key] for r in group_rows if key in r}
                if len(vals) > 1:
                    non_binding.append(key)
            representative["non_binding_params"] = ", ".join(non_binding) if non_binding else "—"
        else:
            representative["non_binding_params"] = "—"

        deduped.append(representative)

    return pd.DataFrame(deduped)


# ---------------------------------------------------------------------------
# Sweep modes
# ---------------------------------------------------------------------------

def _sweep_filters(
    raw_df: pd.DataFrame,
    base_cfg: dict,
    min_bets: int,
    progress_callback=None,
) -> list[dict]:
    """
    Pre-score once with base weights, then sweep all filter combos using the
    vectorized numpy path (no DataFrame copies or per-combo groupby).
    """
    if progress_callback is None:
        print("Pre-scoring dataset (once)…", flush=True)
    with redirect_stdout(io.StringIO()):
        scored = _prescored_df(raw_df, base_cfg)

    if progress_callback is None:
        print("Pre-computing coverage arrays…", flush=True)
    arrays = _precompute_arrays(scored, base_cfg)

    combos = _build_filter_combos()
    weight_defaults = dict(zip(WEIGHT_KEYS, WEIGHT_GRID[0]))
    total = len(combos)
    if progress_callback is None:
        print(f"Sweeping {total} filter combinations (vectorized)…", flush=True)

    rows = []
    for i, patch in enumerate(combos, 1):
        m, fp = _vectorized_settle(arrays, patch, min_bets)
        if m is not None:
            rows.append({**patch, **weight_defaults, **m, "__fp__": fp})
        if progress_callback is not None:
            progress_callback(i, total, len(rows))
        elif i % max(1, total // 10) == 0 or i == total:
            print(f"  {i}/{total} ({100*i//total}%)  kept={len(rows)}", end="\r", flush=True)
    if progress_callback is None:
        print()
    return rows


def _sweep_weights(
    raw_df: pd.DataFrame,
    base_cfg: dict,
    min_bets: int,
    progress_callback=None,
    max_workers: int = 1,
) -> list[dict]:
    """Sweep weight combos with default filter thresholds."""
    filter_defaults = {k: base_cfg[k] for k in FILTER_GRID}
    patches = _build_weight_patches()
    total = len(patches)
    if progress_callback is None:
        print(f"Sweeping {total} weight combinations…", flush=True)

    rows = []
    feature_df = _prepare_feature_frame(raw_df, base_cfg)

    def _run_patch(patch: dict) -> dict | None:
        cfg = {**base_cfg, **filter_defaults, **patch}
        scored = _score_from_feature_frame(feature_df, cfg)
        m, fp = _apply_and_settle(scored, cfg)
        if m and m["total_bets"] >= min_bets:
            return {**filter_defaults, **patch, **m, "__fp__": fp}
        return None

    if max_workers and max_workers > 1:
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_patch, patch) for patch in patches]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    rows.append(result)
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total, len(rows))
                else:
                    print(f"  {completed}/{total}  kept={len(rows)}", end="\r", flush=True)
    else:
        for i, patch in enumerate(patches, 1):
            result = _run_patch(patch)
            if result is not None:
                rows.append(result)
            if progress_callback is not None:
                progress_callback(i, total, len(rows))
            else:
                print(f"  {i}/{total}  kept={len(rows)}", end="\r", flush=True)
    if progress_callback is None:
        print()
    return rows


def _sweep_full(raw_df: pd.DataFrame, base_cfg: dict, min_bets: int) -> list[dict]:
    """For each weight set, pre-score once then sweep all filter combos."""
    filter_combos = _build_filter_combos()
    weight_patches = _build_weight_patches()
    total_weight = len(weight_patches)
    total = total_weight * len(filter_combos)
    print(f"Full sweep: {total_weight} weight sets × {len(filter_combos)} filter combos = {total}", flush=True)

    rows = []
    done = 0
    for wi, wp in enumerate(weight_patches, 1):
        cfg_w = {**base_cfg, **wp}
        scored = _prescored_df(raw_df, cfg_w)
        for fc in filter_combos:
            cfg = {**cfg_w, **fc}
            m, fp = _apply_and_settle(scored, cfg)
            if m and m["total_bets"] >= min_bets:
                rows.append({**fc, **wp, **m, "__fp__": fp})
            done += 1
        pct = 100 * done // total
        print(f"  weight set {wi}/{total_weight}  done={done}/{total} ({pct}%)  kept={len(rows)}", end="\r", flush=True)
    print()
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid-search strategy parameters")
    p.add_argument(
        "--mode",
        choices=["filters", "weights", "full"],
        default="filters",
        help="Which parameters to sweep (default: filters only)",
    )
    p.add_argument(
        "--min-bets",
        type=int,
        default=30,
        help="Minimum bets required for a combination to appear in results (default: 30)",
    )
    p.add_argument(
        "--top",
        type=int,
        default=25,
        help="Number of top rows to display (default: 25)",
    )
    p.add_argument(
        "--sort-by",
        choices=["roi", "total_profit", "total_bets", "strike_rate"],
        default="roi",
        help="Column to rank results by (default: roi)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Optional path to write full results CSV",
    )
    p.add_argument(
        "--db",
        default=None,
        help="Path to race_reports.sqlite (overrides config)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    base_cfg = dict(CONFIG)
    if args.db:
        base_cfg["database_path"] = args.db

    print(f"Loading backtest data from: {base_cfg['database_path']}", flush=True)
    raw_df = load_race_runners(base_cfg["database_path"], "backtest", base_cfg)
    print(f"Loaded {len(raw_df):,} rows from {raw_df['race_id'].nunique():,} races\n", flush=True)

    if args.mode == "filters":
        rows = _sweep_filters(raw_df, base_cfg, args.min_bets)
    elif args.mode == "weights":
        rows = _sweep_weights(raw_df, base_cfg, args.min_bets)
    else:
        rows = _sweep_full(raw_df, base_cfg, args.min_bets)

    if not rows:
        print("No combinations produced enough bets. Try lowering --min-bets.")
        return

    total_raw = len(rows)
    df = _deduplicate(rows)
    df = df.sort_values(args.sort_by, ascending=False).reset_index(drop=True)
    unique_sets = len(df)

    display_cols = [
        "min_price", "max_price", "min_raw_edge", "min_ev", "max_model_rank",
        "min_model_probability", "min_model_vs_market_ratio",
        "min_field_size", "max_field_size",
        "weight_speed_rating", "weight_recent_form", "weight_suitability",
        "weight_connections", "weight_market_sanity", "weight_jockey", "weight_trainer",
        *MARKET_CONFIRMATION_KEYS,
        "total_bets", "wins", "roi", "total_profit", "strike_rate",
        "avg_price", "avg_edge", "duplicate_combos", "non_binding_params",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    top_df = df[display_cols].head(args.top).copy()
    for col in ["roi", "strike_rate", "avg_edge"]:
        if col in top_df.columns:
            top_df[col] = top_df[col].map(lambda x: f"{x*100:.1f}%")

    # Sample-size warning flag
    if "total_bets" in top_df.columns:
        top_df["confidence"] = top_df["total_bets"].map(
            lambda n: "LOW (<100 bets)" if n < LOW_SAMPLE_THRESHOLD else "OK"
        )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 260)
    pd.set_option("display.float_format", "{:.3f}".format)

    print(
        f"\nTop {min(args.top, unique_sets)} unique bet sets by {args.sort_by} "
        f"({total_raw} combos → {unique_sets} distinct bet sets after deduplication):\n"
    )
    print(top_df.to_string(index=True))

    low_sample = int((df["total_bets"] < LOW_SAMPLE_THRESHOLD).sum())
    if low_sample:
        print(
            f"\n⚠  {low_sample} of {unique_sets} results have <{LOW_SAMPLE_THRESHOLD} bets "
            f"— ROI figures are unreliable at this sample size."
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df[display_cols].to_csv(out_path, index=False)
        print(f"\nFull deduplicated results ({unique_sets} rows) written to: {args.output}")

    best = df.iloc[0]
    print("\n--- Best config (paste into betting/config.py) ---")
    for key in list(FILTER_GRID) + WEIGHT_KEYS + MARKET_CONFIRMATION_KEYS:
        if key in best:
            print(f'    "{key}": {best[key]},')
    nb = best.get("non_binding_params", "—")
    if nb and nb != "—":
        print(f"# Note: {nb} were non-binding (changing them didn't change the bet set)")
    print("---------------------------------------------------")


if __name__ == "__main__":
    main()
