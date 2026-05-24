"""Reporting helpers for settled backtest results."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pandas as pd

from .staking import fibonacci_stakes

_BREAKDOWN_COLUMNS = [
    "breakdown_type",
    "group_value",
    "bets",
    "wins",
    "total_staked",
    "total_profit",
    "roi",
    "strike_rate",
    "avg_price",
    "avg_edge",
]

_BREAKDOWN_ORDER = [
    ("price_band", "PRICE BAND"),
    ("track", "TRACK"),
    ("distance_band", "DISTANCE BAND"),
    ("condition", "CONDITION"),
    ("field_size", "FIELD SIZE"),
    ("market_rank", "MARKET RANK"),
    ("model_rank", "MODEL RANK"),
    ("price_quality", "PRICE QUALITY"),
]

_SUMMARY_ORDER = [
    "total_bets",
    "total_staked",
    "total_profit",
    "roi",
    "strike_rate",
    "avg_price",
    "avg_edge",
    "wins",
]

_CALIBRATION_COLUMNS = [
    "model_prob_bucket",
    "runners",
    "wins",
    "actual_win_pct",
    "model_prob_pct",
    "market_prob_pct",
    "calibration_ratio",
]

_CALIBRATION_BINS = [0.0, 0.05, 0.07, 0.10, 0.15, 0.25, 1.000001]
_CALIBRATION_LABELS = ["0-5%", "5-7%", "7-10%", "10-15%", "15-25%", "25%+"]


def _empty_breakdown() -> pd.DataFrame:
    return pd.DataFrame(columns=_BREAKDOWN_COLUMNS)


def _empty_calibration() -> pd.DataFrame:
    return pd.DataFrame(columns=_CALIBRATION_COLUMNS)


def _series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _winner_series(df: pd.DataFrame) -> pd.Series:
    if "is_winner" not in df.columns:
        return pd.Series(0, index=df.index, dtype=int)
    return pd.to_numeric(df["is_winner"], errors="coerce").fillna(0).astype(int)


def _with_breakdown_type(df: pd.DataFrame, breakdown_type: str) -> pd.DataFrame:
    if df.empty:
        return _empty_breakdown()
    out = df.copy()
    out["breakdown_type"] = breakdown_type
    return out[_BREAKDOWN_COLUMNS]


def _format_metric(key: str, value: object) -> str:
    if isinstance(value, float):
        if key in {"roi", "strike_rate", "avg_edge"}:
            return f"{value:.2%}"
        return f"{value:.3f}"
    return str(value)


def _prepare_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    table = df.copy()
    for col in ["total_staked", "total_profit", "avg_price", "avg_edge", "roi", "strike_rate"]:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce")
    for col in ["bets", "wins"]:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce").fillna(0).astype(int)
    return table


def _ensure_price_band(result_df: pd.DataFrame) -> pd.DataFrame:
    if "price_band" in result_df.columns:
        return result_df
    prices = pd.to_numeric(result_df.get("live_price"), errors="coerce")
    result = result_df.copy()
    result["price_band"] = pd.Series("NO_PRICE", index=result.index, dtype=object)
    result.loc[prices < 3, "price_band"] = "<3"
    result.loc[(prices >= 3) & (prices < 6), "price_band"] = "3-6"
    result.loc[(prices >= 6) & (prices < 9), "price_band"] = "6-9"
    result.loc[(prices >= 9) & (prices < 12), "price_band"] = "9-12"
    result.loc[(prices >= 12) & (prices <= 20), "price_band"] = "12-20"
    result.loc[prices > 20, "price_band"] = ">20"
    return result


def _ensure_distance_band(result_df: pd.DataFrame) -> pd.DataFrame:
    if "distance_band" in result_df.columns:
        return result_df
    distance = pd.to_numeric(result_df.get("distance_m"), errors="coerce")
    result = result_df.copy()
    result["distance_band"] = pd.Series("UNKNOWN", index=result.index, dtype=object)
    result.loc[distance < 1400, "distance_band"] = "sprint"
    result.loc[(distance >= 1400) & (distance < 2000), "distance_band"] = "middle"
    result.loc[distance >= 2000, "distance_band"] = "staying"
    return result


def _chronological_bets(result_df: pd.DataFrame) -> pd.DataFrame:
    for column in ["race_start_time", "start_time", "race_time", "start_time_iso"]:
        if column in result_df.columns:
            return result_df.sort_values(column)
    return result_df



def _flat_staking_metrics(result_df: pd.DataFrame, config: dict) -> dict:
    if result_df.empty:
        return {"staked": 0.0, "profit": 0.0, "roi": 0.0}

    stake_unit = float(config.get("stake", 0.0))
    winners = _winner_series(result_df)
    live_price = _series(result_df, "live_price").fillna(0.0)
    flat_stake = pd.Series(stake_unit, index=result_df.index, dtype=float)
    flat_profit = pd.Series(-stake_unit, index=result_df.index, dtype=float)
    flat_profit = flat_profit.where(winners != 1, stake_unit * (live_price - 1.0))
    total_staked = float(flat_stake.sum())
    total_profit = float(flat_profit.sum())
    return {
        "staked": total_staked,
        "profit": total_profit,
        "roi": float(total_profit / total_staked) if total_staked else 0.0,
    }



def _fibonacci_staking_metrics(result_df: pd.DataFrame, config: dict) -> dict:
    if result_df.empty:
        return {
            "staked": 0.0,
            "profit": 0.0,
            "roi": 0.0,
            "max_level": 0,
            "max_stake": 0.0,
        }

    fib_df = fibonacci_stakes(_chronological_bets(result_df), config)
    fib_staked = float(pd.to_numeric(fib_df["fib_stake"], errors="coerce").fillna(0.0).sum())
    fib_profit = float(pd.to_numeric(fib_df["fib_profit"], errors="coerce").fillna(0.0).sum())
    fib_level = pd.to_numeric(fib_df["fib_level"], errors="coerce").fillna(0)
    fib_stake = pd.to_numeric(fib_df["fib_stake"], errors="coerce").fillna(0.0)
    return {
        "staked": fib_staked,
        "profit": fib_profit,
        "roi": float(fib_profit / fib_staked) if fib_staked else 0.0,
        "max_level": int(fib_level.max()) if not fib_level.empty else 0,
        "max_stake": float(fib_stake.max()) if not fib_stake.empty else 0.0,
    }



def _flat_result_for_breakdowns(result_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of result_df with flat 1-unit stakes for consistent breakdown analysis.

    Breakdown tables measure selection quality (which tracks/distances perform best).
    They should always use 1-unit flat stakes so the analysis isn't distorted by
    where the runner happens to fall in a Fibonacci sequence.
    """
    flat = result_df.copy()
    is_winner = _winner_series(flat)
    live_price = _series(flat, "live_price").fillna(0.0)
    flat["stake"] = 1.0
    flat["profit"] = is_winner * (live_price - 1.0) + (1 - is_winner) * -1.0
    return flat



def build_report(
    result_df: pd.DataFrame,
    config: dict,
    calibration_source_df: pd.DataFrame | None = None,
) -> dict:
    """Return dict with summary metrics and standard breakdown tables."""
    flat_df = _flat_result_for_breakdowns(result_df)
    calibration_df = calibration_source_df if calibration_source_df is not None else result_df
    return {
        "summary": summary_metrics(result_df),
        "staking_comparison": {
            "flat": _flat_staking_metrics(result_df, config),
            "fibonacci": _fibonacci_staking_metrics(result_df, config),
        },
        "price_band": breakdown_by_price_band(flat_df),
        "track": breakdown_by_track(flat_df),
        "distance_band": breakdown_by_distance_band(flat_df),
        "condition": breakdown_by_condition(flat_df),
        "field_size": breakdown_by_field_size(flat_df),
        "market_rank": breakdown_by_market_rank(flat_df),
        "model_rank": breakdown_by_model_rank(flat_df),
        "price_quality": breakdown_by_price_quality(flat_df),
        "model_prob_calibration": model_prob_calibration(calibration_df),
    }


def summary_metrics(result_df: pd.DataFrame) -> dict:
    """Compute top-level settled betting metrics."""
    if result_df.empty:
        return {
            "total_bets": 0,
            "total_staked": 0.0,
            "total_profit": 0.0,
            "roi": 0.0,
            "strike_rate": 0.0,
            "avg_price": 0.0,
            "avg_edge": 0.0,
            "wins": 0,
        }

    stake = _series(result_df, "stake").fillna(0.0)
    profit = _series(result_df, "profit").fillna(0.0)
    live_price = _series(result_df, "live_price")
    edge = _series(result_df, "edge")
    wins = int(_winner_series(result_df).sum())
    total_bets = int(stake.count())
    total_staked = float(stake.sum())
    total_profit = float(profit.sum())

    return {
        "total_bets": total_bets,
        "total_staked": total_staked,
        "total_profit": total_profit,
        "roi": float(total_profit / total_staked) if total_staked else 0.0,
        "strike_rate": float(wins / total_bets) if total_bets else 0.0,
        "avg_price": float(live_price.mean()) if live_price.notna().any() else 0.0,
        "avg_edge": float(edge.mean()) if edge.notna().any() else 0.0,
        "wins": wins,
    }


def breakdown_by(result_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Group settled bets by one column and compute standard metrics."""
    if group_col not in result_df.columns:
        raise ValueError(f"Column '{group_col}' not found")
    if result_df.empty:
        return _empty_breakdown()

    g = result_df.groupby(group_col, dropna=False)
    out = pd.DataFrame(
        {
            "bets": g["stake"].count(),
            "wins": g["is_winner"].sum(),
            "total_staked": g["stake"].sum(),
            "total_profit": g["profit"].sum(),
            "avg_price": g["live_price"].mean(),
            "avg_edge": g["edge"].mean(),
        }
    )
    out["roi"] = out["total_profit"] / out["total_staked"].replace(0, float("nan"))
    out["strike_rate"] = out["wins"] / out["bets"].replace(0, float("nan"))
    out = out.reset_index().rename(columns={group_col: "group_value"})
    out["group_value"] = out["group_value"].where(out["group_value"].notna(), "UNKNOWN")
    out.insert(0, "breakdown_type", group_col)
    return out.sort_values("total_profit", ascending=False).reset_index(drop=True)[_BREAKDOWN_COLUMNS]


def breakdown_by_price_band(result_df: pd.DataFrame) -> pd.DataFrame:
    return _with_breakdown_type(breakdown_by(_ensure_price_band(result_df), "price_band"), "price_band")


def breakdown_by_track(result_df: pd.DataFrame) -> pd.DataFrame:
    if "track" in result_df.columns:
        return _with_breakdown_type(breakdown_by(result_df, "track"), "track")
    if "competition_name" not in result_df.columns:
        raise ValueError("Column 'competition_name' not found")
    working = result_df.copy()
    working["track"] = working["competition_name"].fillna("UNKNOWN")
    return _with_breakdown_type(breakdown_by(working, "track"), "track")


def breakdown_by_distance_band(result_df: pd.DataFrame) -> pd.DataFrame:
    return _with_breakdown_type(breakdown_by(_ensure_distance_band(result_df), "distance_band"), "distance_band")


def breakdown_by_condition(result_df: pd.DataFrame) -> pd.DataFrame:
    if "condition" in result_df.columns:
        return _with_breakdown_type(breakdown_by(result_df, "condition"), "condition")
    if "track_status" not in result_df.columns:
        raise ValueError("Column 'track_status' not found")
    working = result_df.copy()
    working["condition"] = working["track_status"].fillna("UNKNOWN")
    return _with_breakdown_type(breakdown_by(working, "condition"), "condition")


def breakdown_by_field_size(result_df: pd.DataFrame) -> pd.DataFrame:
    return _with_breakdown_type(breakdown_by(result_df, "field_size"), "field_size")


def breakdown_by_market_rank(result_df: pd.DataFrame) -> pd.DataFrame:
    return _with_breakdown_type(breakdown_by(result_df, "market_rank"), "market_rank")


def breakdown_by_model_rank(result_df: pd.DataFrame) -> pd.DataFrame:
    return _with_breakdown_type(breakdown_by(result_df, "model_rank"), "model_rank")


def breakdown_by_price_quality(result_df: pd.DataFrame) -> pd.DataFrame:
    return _with_breakdown_type(breakdown_by(result_df, "price_quality"), "price_quality")


def model_prob_calibration(result_df: pd.DataFrame) -> pd.DataFrame:
    """Bucket model probabilities and compare to realised strike rate."""
    required = {"model_prob", "is_winner"}
    if result_df.empty or not required.issubset(result_df.columns):
        return _empty_calibration()

    clean = result_df.copy()
    clean["model_prob"] = pd.to_numeric(clean["model_prob"], errors="coerce")
    market_col = "raw_market_prob" if "raw_market_prob" in clean.columns else "market_implied_prob"
    clean["market_prob_for_cal"] = pd.to_numeric(clean[market_col], errors="coerce")
    clean["is_winner"] = pd.to_numeric(clean["is_winner"], errors="coerce")
    clean = clean[
        clean["model_prob"].notna()
        & clean["market_prob_for_cal"].notna()
        & clean["is_winner"].notna()
    ]
    if clean.empty:
        return _empty_calibration()

    clean["model_prob_bucket"] = pd.cut(
        clean["model_prob"],
        bins=_CALIBRATION_BINS,
        labels=_CALIBRATION_LABELS,
        include_lowest=True,
        right=False,
    )
    grouped = (
        clean.groupby("model_prob_bucket", observed=False)
        .agg(
            runners=("is_winner", "count"),
            wins=("is_winner", "sum"),
            model_prob=("model_prob", "mean"),
            market_prob=("market_prob_for_cal", "mean"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["runners"] > 0].copy()
    if grouped.empty:
        return _empty_calibration()

    grouped["actual_win_pct"] = (grouped["wins"] / grouped["runners"] * 100.0).round(1)
    grouped["model_prob_pct"] = (grouped["model_prob"] * 100.0).round(1)
    grouped["market_prob_pct"] = (grouped["market_prob"] * 100.0).round(1)
    grouped["calibration_ratio"] = (
        (grouped["wins"] / grouped["runners"])
        / grouped["model_prob"].replace(0.0, float("nan"))
    ).round(2)
    return grouped[_CALIBRATION_COLUMNS]


def _print_staking_comparison(comparison: dict) -> None:
    if not comparison:
        return
    flat = comparison.get("flat", {})
    fibonacci = comparison.get("fibonacci", {})
    print("\nStaking comparison:")
    print(
        "  Flat stake  ROI: "
        f"{flat.get('roi', 0.0):.1%}  profit: {flat.get('profit', 0.0):.2f}  staked: {flat.get('staked', 0.0):.2f}"
    )
    print(
        "  Fibonacci   ROI: "
        f"{fibonacci.get('roi', 0.0):.1%}  profit: {fibonacci.get('profit', 0.0):.2f}  "
        f"staked: {fibonacci.get('staked', 0.0):.2f}  max_level: {fibonacci.get('max_level', 0)}  "
        f"max_stake: {fibonacci.get('max_stake', 0.0):.2f} units"
    )



def print_report(report_dict: dict) -> None:
    """Print a human-readable summary followed by all configured breakdowns."""
    print("SUMMARY")
    summary = report_dict.get("summary", {})
    if not summary:
        print("No data")
    else:
        for key in _SUMMARY_ORDER:
            if key in summary:
                label = key.replace("_", " ").title()
                print(f"{label}: {_format_metric(key, summary[key])}")

    _print_staking_comparison(report_dict.get("staking_comparison", {}))

    for report_key, title in _BREAKDOWN_ORDER:
        print(f"\n{title}")
        df = _prepare_table(report_dict.get(report_key, _empty_breakdown()))
        if df.empty:
            print("No data")
            continue
        print(df.to_string(index=False))

    print("\nMODEL PROBABILITY CALIBRATION")
    cal_df = report_dict.get("model_prob_calibration", _empty_calibration())
    if cal_df.empty:
        print("No data")
    else:
        print(cal_df.to_string(index=False))


def export_csv(df: pd.DataFrame, output_path: str) -> None:
    """Write a DataFrame to CSV, creating parent directories as needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def export_all(result_df: pd.DataFrame, report_dict: dict, output_dir: str, timestamp: str) -> dict:
    """Export the settled results, concatenated breakdowns, and text summary."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    backtest_path = output_root / f"backtest_{timestamp}.csv"
    breakdown_path = output_root / f"breakdown_{timestamp}.csv"
    calibration_path = output_root / f"calibration_{timestamp}.csv"
    summary_path = output_root / f"summary_{timestamp}.txt"

    export_csv(result_df, str(backtest_path))

    breakdown_frames = [
        report_dict.get("price_band", _empty_breakdown()),
        report_dict.get("track", _empty_breakdown()),
        report_dict.get("distance_band", _empty_breakdown()),
        report_dict.get("condition", _empty_breakdown()),
        report_dict.get("field_size", _empty_breakdown()),
        report_dict.get("market_rank", _empty_breakdown()),
        report_dict.get("model_rank", _empty_breakdown()),
        report_dict.get("price_quality", _empty_breakdown()),
    ]
    combined_breakdowns = pd.concat(breakdown_frames, ignore_index=True) if breakdown_frames else _empty_breakdown()
    combined_breakdowns = combined_breakdowns.reindex(columns=_BREAKDOWN_COLUMNS)
    export_csv(combined_breakdowns, str(breakdown_path))
    export_csv(report_dict.get("model_prob_calibration", _empty_calibration()), str(calibration_path))

    buffer = StringIO()
    with redirect_stdout(buffer):
        print_report(report_dict)
    summary_path.write_text(buffer.getvalue(), encoding="utf-8")

    return {
        "backtest_csv": str(backtest_path),
        "breakdown_csv": str(breakdown_path),
        "calibration_csv": str(calibration_path),
        "summary_txt": str(summary_path),
    }
