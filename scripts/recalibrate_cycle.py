#!/usr/bin/env python3
"""
Automated full recalibration cycle.

Iterates the following loop until ROI stops improving:

  1. Sweep scoring weights  (find the best weight set)
  2. Update runtime config with best weights
  3. Re-fit calibration model (weights changed → raw_model_prob changed)
  4. Sweep filter thresholds (find best edge/price/rank cutoffs)
  5. Update runtime config with best filters
  6. Compare ROI to previous iteration — stop when gain < --min-gain

At the end, prints the final config block and optionally writes it back to
betting/config.py.

Usage
-----
# Dry-run: print best config but do not write config.py
python3 scripts/recalibrate_cycle.py

# Write final config back to betting/config.py
python3 scripts/recalibrate_cycle.py --write

# Tune convergence threshold (default 0.1% ROI gain)
python3 scripts/recalibrate_cycle.py --min-gain 0.005 --write

# Cap iterations and require bigger bet samples
python3 scripts/recalibrate_cycle.py --max-iters 5 --min-bets 50 --write
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback for minimal envs
    tqdm = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from betting.calibration import clear_model_cache
from betting.config import CONFIG
from betting.db import load_race_runners

# Re-use internals from finetune.py — avoids duplication
from scripts.finetune import (
    FILTER_GRID,
    MARKET_CONFIRMATION_KEYS,
    WEIGHT_KEYS,
    _prescored_df,
    _precompute_arrays,
    _sweep_filters,
    _sweep_weights,
    _vectorized_settle,
)

# Re-use internals from fit_calibration.py
from scripts.fit_calibration import fit_isotonic

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _best_row(rows: list[dict]) -> dict:
    """Return the row with the highest ROI (and most bets as tiebreaker)."""
    return max(rows, key=lambda r: (r["roi"], r["total_bets"]))


def _quiet_call(func, *args, **kwargs):
    """Run a noisy helper without leaking its internal stdout."""
    with redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _make_progress_callback(desc: str):
    if tqdm is None:
        def _callback(current: int, total: int, kept: int) -> None:
            if current == 1 or current == total or current % max(1, total // 10) == 0:
                print(f"    {desc}: {current}/{total}  kept={kept}", flush=True)
        return None, _callback

    bar = tqdm(total=0, desc=desc, unit="combo", leave=True, dynamic_ncols=True)

    def _callback(current: int, total: int, kept: int) -> None:
        if bar.total != total:
            bar.reset(total=total)
        delta = current - bar.n
        if delta > 0:
            bar.update(delta)
        bar.set_postfix_str(f"kept={kept}")

    return bar, _callback


def _extract_weights(row: dict) -> dict:
    return {k: row[k] for k in WEIGHT_KEYS + MARKET_CONFIRMATION_KEYS if k in row}


def _extract_filters(row: dict) -> dict:
    return {k: row[k] for k in FILTER_GRID if k in row}


def _weight_values(config: dict) -> dict:
    return {k: float(config[k]) for k in WEIGHT_KEYS if k in config}


def _validate_weight_sum(weight_values: dict) -> float:
    weight_sum = sum(weight_values.values())
    if abs(weight_sum - 1.0) > 0.005:
        raise ValueError(
            f"Weight sum sanity check failed: sum={weight_sum:.4f} (expected ~1.0). "
            f"Refusing to write config. Weights: {weight_values}"
        )
    return weight_sum


def _normalized_weight_values(config: dict) -> dict:
    weight_values = _weight_values(config)
    if not weight_values:
        return {}
    total = sum(weight_values.values())
    if total <= 0:
        raise ValueError("Cannot normalize non-positive weight total.")
    normalized = {
        key: weight_values[key] / total
        for key in WEIGHT_KEYS
        if key in weight_values
    }
    rounded = {}
    running = 0.0
    present_keys = [key for key in WEIGHT_KEYS if key in normalized]
    for key in present_keys[:-1]:
        value = round(normalized[key], 3)
        rounded[key] = value
        running += value
    rounded[present_keys[-1]] = round(1.0 - running, 3)
    return rounded


def _time_split_races(raw_df: pd.DataFrame, train_frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    races = (
        raw_df[["race_id", "start_time_iso"]]
        .drop_duplicates()
        .assign(start_time=lambda t: pd.to_datetime(t["start_time_iso"], errors="coerce", utc=True))
        .dropna(subset=["start_time"])
        .sort_values("start_time")
    )
    if races.empty:
        raise ValueError("No parseable start_time_iso values for time-based split.")
    n_train = int(len(races) * float(train_frac))
    n_train = max(1, min(n_train, len(races) - 1))
    train_races = set(races.head(n_train)["race_id"].values)
    holdout_races = set(races.tail(len(races) - n_train)["race_id"].values)
    train_df = raw_df[raw_df["race_id"].isin(train_races)].copy()
    holdout_df = raw_df[raw_df["race_id"].isin(holdout_races)].copy()
    return train_df, holdout_df


def _refit_calibration(raw_df: pd.DataFrame, config: dict) -> None:
    """Re-score training rows with current config and save a calibration model."""
    cal_path = _calibration_path(config)
    print(f"  Re-fitting calibration → {cal_path}", flush=True)
    with redirect_stdout(io.StringIO()):
        scored = _prescored_df(raw_df, config)
    ir = fit_isotonic(scored)
    Path(cal_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ir, cal_path)
    clear_model_cache()
    print("  Calibration model saved and cache cleared.", flush=True)


def _calibration_path(config: dict) -> str:
    explicit = config.get("calibration_model_path")
    if explicit:
        return explicit
    return str(Path(__file__).resolve().parent.parent / "betting" / "calibration_model.pkl")


def _evaluate_metrics(raw_df: pd.DataFrame, config: dict, min_bets: int) -> dict:
    """Run the full pipeline with the given config and return a compact metric summary."""
    with redirect_stdout(io.StringIO()):
        scored = _prescored_df(raw_df, config)
    arrays = _precompute_arrays(scored, config)
    filter_patch = {k: config[k] for k in FILTER_GRID}
    m, _ = _vectorized_settle(arrays, filter_patch, min_bets)
    if m is None:
        m = {"roi": 0.0, "total_bets": 0}
    winners = scored[pd.to_numeric(scored.get("is_winner"), errors="coerce").fillna(0).astype(int) == 1].copy()
    if winners.empty:
        winner_logloss = 0.0
        avg_winner_model_prob = 0.0
        avg_winner_market_prob = 0.0
        market_blind_winner_misses = 0
    else:
        winner_probs = pd.to_numeric(winners.get("model_prob"), errors="coerce").fillna(0.0).clip(1e-9, 1.0)
        winner_market = pd.to_numeric(winners.get("raw_market_prob"), errors="coerce").fillna(0.0)
        winner_logloss = float((-np.log(winner_probs)).mean())
        avg_winner_model_prob = float(winner_probs.mean())
        avg_winner_market_prob = float(winner_market.mean())
        market_blind_winner_misses = int(
            ((winner_market >= 0.15) & (winner_probs <= (winner_market - 0.08))).sum()
        )
    return {
        "roi": float(m["roi"]),
        "total_bets": int(m["total_bets"]),
        "winner_logloss": winner_logloss,
        "avg_winner_model_prob": avg_winner_model_prob,
        "avg_winner_market_prob": avg_winner_market_prob,
        "market_blind_winner_misses": market_blind_winner_misses,
    }


def _write_config(best_config: dict, config_path: Path) -> None:
    """
    Overwrite just the tunable values in config.py, preserving all other
    keys, comments, and formatting.  Each tunable key is matched by a simple
    regex so that only the numeric value on that line changes.
    """
    text = config_path.read_text()
    write_config = dict(best_config)
    normalized_weights = _normalized_weight_values(write_config)
    if normalized_weights:
        write_config.update(normalized_weights)
    tunable_keys = list(FILTER_GRID.keys()) + WEIGHT_KEYS + MARKET_CONFIRMATION_KEYS
    for key in tunable_keys:
        if key not in write_config:
            continue
        value = write_config[key]
        fmt = f"{value:.3f}" if isinstance(value, float) else str(value)
        # Match:  "key": <number>,
        pattern = rf'("{re.escape(key)}")\s*:\s*[0-9]+(?:\.[0-9]+)?,'
        replacement = rf'\1: {fmt},'
        new_text = re.sub(pattern, replacement, text)
        if new_text != text:
            text = new_text
    weight_values = _weight_values(write_config)
    if weight_values:
        _validate_weight_sum(weight_values)
    config_path.write_text(text)


def _print_config_block(best_config: dict) -> None:
    print_config = dict(best_config)
    normalized_weights = _normalized_weight_values(print_config)
    if normalized_weights:
        print_config.update(normalized_weights)
    print("\n--- Final config (paste into betting/config.py) ---")
    for key in list(FILTER_GRID.keys()) + WEIGHT_KEYS + MARKET_CONFIRMATION_KEYS:
        if key in print_config:
            print(f'    "{key}": {print_config[key]},')
    print("---------------------------------------------------")


# ---------------------------------------------------------------------------
# Main cycle
# ---------------------------------------------------------------------------

def run_recalibration_cycle(
    config: dict,
    raw_df: pd.DataFrame,
    min_bets: int,
    min_gain: float,
    max_iters: int,
    verbose: bool,
) -> tuple[dict, list[dict], dict]:
    """
    Run the weight → calibration → filter loop until convergence.

    Returns
    -------
    best_config : dict  — final config with best weights + filters
    history     : list[dict]  — per-iteration metrics for reporting
    summary     : dict  — baseline/best holdout metrics for write gating
    """
    current_cfg = dict(config)
    train_df, holdout_df = _time_split_races(raw_df, train_frac=0.8)
    train_metrics = _evaluate_metrics(train_df, current_cfg, min_bets)
    holdout_metrics = _evaluate_metrics(holdout_df, current_cfg, min_bets)
    baseline_holdout_roi = holdout_metrics["roi"]
    prev_holdout_roi = holdout_metrics["roi"]
    best_holdout_roi = holdout_metrics["roi"]
    best_cfg = dict(current_cfg)
    history = []
    cpu_count = os.cpu_count() or 1
    weight_workers = max(1, cpu_count)

    print(
        f"\nSplit: train {train_df['race_id'].nunique():,} races / holdout {holdout_df['race_id'].nunique():,} races"
    )
    print(
        f"Starting train ROI: {train_metrics['roi']*100:.2f}% ({train_metrics['total_bets']} bets)"
    )
    print(
        f"Starting holdout ROI: {holdout_metrics['roi']*100:.2f}% ({holdout_metrics['total_bets']} bets)\n"
    )
    print(
        "Starting holdout winner diagnostics: "
        f"avg_model={holdout_metrics['avg_winner_model_prob']*100:.1f}%  "
        f"avg_market={holdout_metrics['avg_winner_market_prob']*100:.1f}%  "
        f"logloss={holdout_metrics['winner_logloss']:.3f}  "
        f"blind_misses={holdout_metrics['market_blind_winner_misses']}"
    )
    print("=" * 65)

    for iteration in range(1, max_iters + 1):
        print(f"\n[Iteration {iteration}/{max_iters}]")

        # ── Step 1: sweep weights ─────────────────────────────────────────
        print(f"  Step 1/3 — Sweeping scoring weights/market overrides ({weight_workers} threads)…", flush=True)
        weights_bar, weights_progress = _make_progress_callback("weights")
        try:
            weight_rows = _quiet_call(
                _sweep_weights,
                train_df,
                current_cfg,
                min_bets,
                progress_callback=weights_progress,
                max_workers=weight_workers,
            )
        finally:
            if weights_bar is not None:
                weights_bar.close()
        if not weight_rows:
            print("  No weight combos produced enough bets — keeping current weights.")
        else:
            best_w = _best_row(weight_rows)
            new_weights = _extract_weights(best_w)
            w_sum = sum(new_weights[k] for k in WEIGHT_KEYS if k in new_weights)
            if abs(w_sum - 1.0) > 0.005:
                logger.warning(f"Weight sum={w_sum:.4f} — skipping weight update (sum not ~1.0)")
                best_w = None
                new_weights = {k: current_cfg.get(k) for k in WEIGHT_KEYS}

            if best_w is None:
                print("  → Weight update skipped; keeping current weights.")
            elif new_weights != {k: current_cfg.get(k) for k in WEIGHT_KEYS}:
                print(f"  → New best weights (ROI {best_w['roi']*100:.2f}%, {best_w['total_bets']} bets)")
                if verbose:
                    for k, v in new_weights.items():
                        print(f"      {k}: {v}")
            else:
                print(f"  → Weights unchanged (best ROI {best_w['roi']*100:.2f}%)")
            current_cfg = {**current_cfg, **new_weights}

        # ── Step 2: refit calibration with new weights ────────────────────
        print("  Step 2/3 — Refitting calibration…", flush=True)
        _refit_calibration(train_df, current_cfg)

        # ── Step 3: sweep filters with new calibration ────────────────────
        print("  Step 3/3 — Sweeping filter thresholds…", flush=True)
        filters_bar, filters_progress = _make_progress_callback("filters")
        try:
            filter_rows = _quiet_call(
                _sweep_filters,
                train_df,
                current_cfg,
                min_bets,
                progress_callback=filters_progress,
            )
        finally:
            if filters_bar is not None:
                filters_bar.close()
        if not filter_rows:
            print("  No filter combos produced enough bets — keeping current filters.")
            train_metrics = _evaluate_metrics(train_df, current_cfg, min_bets)
        else:
            best_f = _best_row(filter_rows)
            new_filters = _extract_filters(best_f)
            if new_filters != {k: current_cfg.get(k) for k in FILTER_GRID}:
                print(f"  → New best filters (ROI {best_f['roi']*100:.2f}%, {best_f['total_bets']} bets)")
                if verbose:
                    for k, v in new_filters.items():
                        print(f"      {k}: {v}")
            else:
                print(f"  → Filters unchanged (best ROI {best_f['roi']*100:.2f}%)")
            current_cfg = {**current_cfg, **new_filters}
            train_metrics = {
                "roi": float(best_f["roi"]),
                "total_bets": int(best_f["total_bets"]),
            }

        holdout_metrics = _evaluate_metrics(holdout_df, current_cfg, min_bets)
        holdout_gain = holdout_metrics["roi"] - prev_holdout_roi
        history.append({
            "iteration": iteration,
            "train_roi": train_metrics["roi"],
            "train_bets": train_metrics["total_bets"],
            "holdout_roi": holdout_metrics["roi"],
            "holdout_gain": holdout_gain,
            "holdout_bets": holdout_metrics["total_bets"],
            "holdout_winner_logloss": holdout_metrics["winner_logloss"],
            "holdout_avg_winner_model_prob": holdout_metrics["avg_winner_model_prob"],
            "holdout_avg_winner_market_prob": holdout_metrics["avg_winner_market_prob"],
            "holdout_market_blind_winner_misses": holdout_metrics["market_blind_winner_misses"],
        })

        print(
            f"\n  Iteration {iteration} result: "
            f"train ROI={train_metrics['roi']*100:.2f}% ({train_metrics['total_bets']} bets)  "
            f"holdout ROI={holdout_metrics['roi']*100:.2f}% ({holdout_metrics['total_bets']} bets)  "
            f"holdout gain={holdout_gain*100:+.2f}%"
        )
        print(
            "  Holdout winner diagnostics: "
            f"avg_model={holdout_metrics['avg_winner_model_prob']*100:.1f}%  "
            f"avg_market={holdout_metrics['avg_winner_market_prob']*100:.1f}%  "
            f"logloss={holdout_metrics['winner_logloss']:.3f}  "
            f"blind_misses={holdout_metrics['market_blind_winner_misses']}"
        )
        print("=" * 65)

        if holdout_metrics["roi"] > best_holdout_roi:
            best_holdout_roi = holdout_metrics["roi"]
            best_cfg = dict(current_cfg)

        if holdout_gain < min_gain:
            reason = "converged" if holdout_gain >= 0 else "holdout ROI decreased"
            print(f"\nStopping after iteration {iteration}: {reason} "
                  f"(gain {holdout_gain*100:+.3f}% < threshold {min_gain*100:.3f}%)")
            break

        prev_holdout_roi = holdout_metrics["roi"]

    return best_cfg, history, {
        "baseline_holdout_roi": baseline_holdout_roi,
        "best_holdout_roi": best_holdout_roi,
        "holdout_improved": best_holdout_roi > baseline_holdout_roi,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Automated recalibration cycle")
    p.add_argument("--db", default=None, help="Path to SQLite database")
    p.add_argument(
        "--min-gain",
        type=float,
        default=0.001,
        help="Minimum ROI improvement (as decimal) to continue cycling. Default: 0.001 (0.1%%)",
    )
    p.add_argument(
        "--max-iters",
        type=int,
        default=8,
        help="Maximum number of iterations (safety cap). Default: 8",
    )
    p.add_argument(
        "--min-bets",
        type=int,
        default=30,
        help="Minimum bets required for a combo to be considered. Default: 30",
    )
    p.add_argument(
        "--write",
        action="store_true",
        help="Write final config back to betting/config.py",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-key weight/filter changes each iteration",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config = dict(CONFIG)
    if args.db:
        config["database_path"] = args.db

    config_path = Path(__file__).resolve().parent.parent / "betting" / "config.py"

    print(f"Database : {config['database_path']}")
    print(f"Max iters: {args.max_iters}  |  Min gain: {args.min_gain*100:.3f}%  |  Min bets: {args.min_bets}")
    if args.write:
        print(f"Will write final config → {config_path}")

    print("\nLoading backtest data…", flush=True)
    raw_df = _quiet_call(load_race_runners, config["database_path"], "backtest", config)
    print(f"Loaded {len(raw_df):,} rows from {raw_df['race_id'].nunique():,} races.")
    if raw_df.empty:
        raise ValueError(
            "No eligible backtest rows loaded. Check database_path and backtest filters "
            "(status='finished', result_code in ('W','P','L'), race_number >= 6)."
        )

    best_config, history, summary = run_recalibration_cycle(
        config=config,
        raw_df=raw_df,
        min_bets=args.min_bets,
        min_gain=args.min_gain,
        max_iters=args.max_iters,
        verbose=args.verbose,
    )

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n\nCYCLE SUMMARY")
    print("-" * 45)
    if history:
        hist_df = pd.DataFrame(history)
        hist_df["train_roi_pct"] = hist_df["train_roi"].map(lambda x: f"{x*100:.2f}%")
        hist_df["holdout_roi_pct"] = hist_df["holdout_roi"].map(lambda x: f"{x*100:.2f}%")
        hist_df["holdout_gain_pct"] = hist_df["holdout_gain"].map(lambda x: f"{x*100:+.3f}%")
        hist_df["holdout_avg_winner_model_pct"] = hist_df["holdout_avg_winner_model_prob"].map(
            lambda x: f"{x*100:.1f}%"
        )
        hist_df["holdout_avg_winner_market_pct"] = hist_df["holdout_avg_winner_market_prob"].map(
            lambda x: f"{x*100:.1f}%"
        )
        print(
            hist_df[
                [
                    "iteration",
                    "train_roi_pct",
                    "train_bets",
                    "holdout_roi_pct",
                    "holdout_gain_pct",
                    "holdout_bets",
                    "holdout_avg_winner_model_pct",
                    "holdout_avg_winner_market_pct",
                    "holdout_market_blind_winner_misses",
                ]
            ].to_string(index=False)
        )
    print("-" * 45)

    _print_config_block(best_config)

    if args.write:
        if summary["holdout_improved"]:
            _write_config(best_config, config_path)
            print(
                f"\nbetting/config.py updated "
                f"(holdout ROI improved from {summary['baseline_holdout_roi']*100:.2f}% "
                f"to {summary['best_holdout_roi']*100:.2f}%)."
            )
            print("Re-run `python3 -m pytest` to confirm nothing broke.")
        else:
            print(
                f"\nNo write applied: best holdout ROI {summary['best_holdout_roi']*100:.2f}% "
                f"did not improve on baseline {summary['baseline_holdout_roi']*100:.2f}%."
            )
    else:
        print("\nRun with --write to apply these values to betting/config.py.")


if __name__ == "__main__":
    main()
