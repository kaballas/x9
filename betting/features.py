"""Feature engineering for runner-level betting inputs."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def _is_wet_track(track_status: pd.Series) -> pd.Series:
    status = track_status.fillna("").str.lower()
    return status.str.contains("soft") | status.str.contains("heavy")


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce").fillna(0.0)
    denominator = pd.to_numeric(denominator, errors="coerce").fillna(0.0)
    return numerator.div(denominator.where(denominator > 0), fill_value=0.0).fillna(0.0)


def _numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    raw = df[column] if column in df.columns else pd.Series(default, index=df.index)
    if not isinstance(raw, pd.Series):
        raw = pd.Series(raw, index=df.index)
    return pd.to_numeric(raw, errors="coerce").fillna(default)


# legacy - use _smooth_rate instead
def _rate_with_prior_legacy(
    wins: pd.Series,
    starts: pd.Series,
    prior: float = 0.30,
    min_starts: int = 3,
) -> pd.Series:
    """Bayesian blend of observed win rate and a neutral prior."""
    wins = pd.to_numeric(wins, errors="coerce").fillna(0.0)
    starts = pd.to_numeric(starts, errors="coerce").fillna(0.0)
    weight = (starts / min_starts).clip(0.0, 1.0)
    observed = wins.div(starts.where(starts > 0), fill_value=0.0).fillna(0.0)
    return (observed * weight + prior * (1.0 - weight)).clip(0.0, 1.0)


def _smooth_rate(
    wins: pd.Series,
    starts: pd.Series,
    prior: float = 0.25,
    prior_starts: int = 4,
) -> pd.Series:
    """Bayesian smoothing: prior always blends in, regardless of sample size."""
    wins_n = pd.to_numeric(wins, errors="coerce").fillna(0.0)
    starts_n = pd.to_numeric(starts, errors="coerce").fillna(0.0)
    return ((wins_n + prior * prior_starts) / (starts_n + prior_starts)).clip(0.0, 1.0)


def build_features(
    df: pd.DataFrame,
    config: dict,
    draw_bias_df: pd.DataFrame | None = None,
    jockey_stats_df: pd.DataFrame | None = None,
    trainer_stats_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build all runner features."""
    df = add_condition_rating(df)
    df = add_distance_band(df)
    df = add_historical_speed_score(df)
    df = add_market_movement_features(df)
    df = add_pace_map_features(df)
    df = add_distance_change_features(df)
    df = add_class_movement_features(df)
    df = add_weight_trend_features(df)
    df = add_barrier_transition_features(df)
    df = add_recent_sp_features(df)
    df = add_jockey_continuity_features(df)
    df = add_form_string_features(df)
    df = add_travel_region_features(df)
    df = add_equipment_features(df)
    df = add_pedigree_priors(df)
    df = add_suitability_score(df)
    df = add_recent_form_score(df)
    df = add_connection_score(df)
    df = add_margin_score(df)
    df = add_freshness_score(df)
    df = add_class_score(df)
    df = add_market_rank(df)
    df = add_price_band(df, config)
    df = add_form_recency_flag(df, config)
    df = add_draw_bias_score(df, draw_bias_df)
    df = add_jockey_score(df, jockey_stats_df)
    df = add_trainer_score(df, trainer_stats_df)
    return df


def add_condition_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Choose wet or dry rating based on track status."""
    result = df.copy()
    wet_mask = _is_wet_track(result["track_status"])
    chosen = pd.Series(np.where(wet_mask, result["wet_rating"], result["dry_rating"]), index=result.index)
    fallback = pd.Series(np.where(wet_mask, result["dry_rating"], result["wet_rating"]), index=result.index)
    result["condition_rating"] = pd.to_numeric(chosen, errors="coerce").combine_first(
        pd.to_numeric(fallback, errors="coerce")
    ).fillna(0.0)
    return result


def _parse_race_time_to_seconds(value) -> float:
    """Parse a recent-run time string into seconds."""
    if value is None or pd.isna(value):
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")

    parts = re.findall(r"\d+", text)
    if not parts:
        return float("nan")
    if len(parts) >= 3:
        minutes = float(parts[-3])
        seconds = float(parts[-2])
        hundredths = float(parts[-1])
        return minutes * 60.0 + seconds + hundredths / 100.0
    if len(parts) == 2:
        seconds = float(parts[0])
        hundredths = float(parts[1])
        return seconds + hundredths / 100.0
    return float(parts[0])


def add_historical_speed_score(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate speed from recent run time and distance history."""
    result = df.copy()
    target_distance = _numeric_series(result, "distance_m", default=np.nan)
    exact_peak = pd.Series(0.0, index=result.index, dtype=float)
    exact_sum = pd.Series(0.0, index=result.index, dtype=float)
    exact_weight = pd.Series(0.0, index=result.index, dtype=float)
    near_sum = pd.Series(0.0, index=result.index, dtype=float)
    near_weight = pd.Series(0.0, index=result.index, dtype=float)
    weighted_sum = pd.Series(0.0, index=result.index, dtype=float)
    weighted_weight = pd.Series(0.0, index=result.index, dtype=float)
    relevant_speed_sum = pd.Series(0.0, index=result.index, dtype=float)
    relevant_speed_sq_sum = pd.Series(0.0, index=result.index, dtype=float)
    relevant_speed_weight = pd.Series(0.0, index=result.index, dtype=float)

    for i, weight in enumerate(_RECENCY_WEIGHTS, start=1):
        time_col = f"recent_{i}_time"
        dist_col = f"recent_{i}_distance_m"
        if time_col not in result.columns or dist_col not in result.columns:
            continue

        times = result[time_col].map(_parse_race_time_to_seconds)
        distances = pd.to_numeric(result[dist_col], errors="coerce")
        run_speed = distances.div(times.where(times > 0))
        valid = run_speed.notna() & distances.notna() & target_distance.notna()
        if not valid.any():
            continue

        gap = (distances - target_distance).abs()
        distance_weight = pd.Series(0.0, index=result.index, dtype=float)
        distance_weight = distance_weight.mask(gap <= 100, 1.00)
        distance_weight = distance_weight.mask((gap > 100) & (gap <= 200), 0.65)
        distance_weight = distance_weight.mask((gap > 200) & (gap <= 300), 0.35)
        distance_weight = distance_weight.mask((gap > 300) & (gap <= 400), 0.05)
        exact_mask = valid & (gap == 0)
        near_mask = valid & (distance_weight > 0)

        if exact_mask.any():
            exact_speed = run_speed.where(exact_mask)
            exact_peak = np.maximum(exact_peak, exact_speed.fillna(0.0))
            exact_sum += weight * exact_speed.fillna(0.0)
            exact_weight += weight * exact_mask.astype(float)

        if near_mask.any():
            near_speed = run_speed.where(near_mask)
            combined_weight = weight * distance_weight.where(near_mask, 0.0)
            weighted_sum += combined_weight * near_speed.fillna(0.0)
            weighted_weight += combined_weight
            near_sum += weight * near_speed.fillna(0.0)
            near_weight += weight * near_mask.astype(float)
            relevant_speed_sum += combined_weight * near_speed.fillna(0.0)
            relevant_speed_sq_sum += combined_weight * near_speed.fillna(0.0).pow(2)
            relevant_speed_weight += combined_weight

    speed_peak_same_distance = exact_peak.where(exact_weight > 0, 0.0)
    speed_avg_same_distance = exact_sum.div(exact_weight.where(exact_weight > 0)).fillna(0.0)
    speed_recent_weighted_distance_adjusted = weighted_sum.div(weighted_weight.where(weighted_weight > 0)).fillna(0.0)
    speed_avg_nearby_distance = near_sum.div(near_weight.where(near_weight > 0)).fillna(0.0)
    speed_sample_count_same_distance = exact_weight.fillna(0.0)
    speed_sample_count_nearby = near_weight.fillna(0.0)

    result["speed_peak_same_distance"] = speed_peak_same_distance
    result["speed_avg_same_distance"] = speed_avg_same_distance
    result["speed_recent_weighted_distance_adjusted"] = speed_recent_weighted_distance_adjusted
    result["speed_avg_nearby_distance"] = speed_avg_nearby_distance
    relevant_mean = relevant_speed_sum.div(relevant_speed_weight.where(relevant_speed_weight > 0)).fillna(0.0)
    relevant_var = (
        relevant_speed_sq_sum.div(relevant_speed_weight.where(relevant_speed_weight > 0)).fillna(0.0)
        - relevant_mean.pow(2)
    ).clip(lower=0.0)
    relevant_std = np.sqrt(relevant_var)
    speed_consistency = 1.0 - (relevant_std / 0.50).clip(0.0, 1.0)

    result["speed_consistency_std"] = relevant_std
    result["speed_consistency"] = speed_consistency.clip(0.0, 1.0)
    result["speed_sample_count_same_distance"] = speed_sample_count_same_distance
    result["speed_sample_count_nearby"] = speed_sample_count_nearby
    confidence = (
        0.50 * speed_sample_count_same_distance.clip(0.0, 1.0)
        + 0.30 * speed_sample_count_nearby.clip(0.0, 1.0)
        + 0.20 * speed_consistency.clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    result["speed_confidence"] = confidence
    result["historical_speed_score"] = speed_recent_weighted_distance_adjusted
    result["speed_feature_confidence"] = confidence
    result["speed_feature_score"] = result["historical_speed_score"]
    missing = result["speed_feature_score"] <= 0
    fallback = pd.to_numeric(result.get("condition_rating", pd.Series(0.0, index=result.index)), errors="coerce").fillna(0.0)
    result.loc[missing, "speed_feature_score"] = fallback.loc[missing] * 0.01
    result["speed_feature_source"] = np.where(
        result["historical_speed_score"] > 0, "history", "condition_rating"
    )
    return result


def _safe_log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    if isinstance(numerator, pd.Series):
        index = numerator.index
    elif isinstance(denominator, pd.Series):
        index = denominator.index
    else:
        index = pd.RangeIndex(1)
    num = pd.to_numeric(
        numerator if isinstance(numerator, pd.Series) else pd.Series(numerator, index=index),
        errors="coerce",
    )
    den = pd.to_numeric(
        denominator if isinstance(denominator, pd.Series) else pd.Series(denominator, index=index),
        errors="coerce",
    )
    valid = num.notna() & den.notna() & (num > 0) & (den > 0)
    out = pd.Series(np.nan, index=num.index, dtype=float)
    out.loc[valid] = np.log(num.loc[valid] / den.loc[valid])
    return out


def _sigmoid(x: pd.Series, scale: float = 1.0) -> pd.Series:
    scale = max(float(scale), 1e-6)
    return 1.0 / (1.0 + np.exp(-x / scale))


def _normalise_text_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "", regex=True)
    )


def add_market_movement_features(df: pd.DataFrame) -> pd.DataFrame:
    """Late-market movement and steam features."""
    result = df.copy()
    open_price = _numeric_series(result, "open_price", default=np.nan)
    fluc1 = _numeric_series(result, "fluc1", default=np.nan)
    live = _numeric_series(result, "live_price", default=np.nan)

    result["price_drift_open_to_live"] = _safe_log_ratio(live, open_price)
    result["price_drift_fluc1_to_live"] = _safe_log_ratio(live, fluc1)
    result["price_move_abs"] = result["price_drift_open_to_live"].abs()

    drift = pd.to_numeric(result["price_drift_open_to_live"], errors="coerce")
    # Negative drift = odds shorten (steam). Positive drift = drift out.
    result["market_movement_score"] = _sigmoid(-drift.fillna(0.0), scale=0.20).clip(0.0, 1.0)
    return result


def _settling_bin(value: object) -> str:
    text = str(value or "").strip().lower()
    if "lead" in text:
        return "leader"
    if "pace" in text and "off" not in text:
        return "onpace"
    if "off" in text and "pace" in text:
        return "mid"
    if "mid" in text:
        return "mid"
    if "back" in text:
        return "back"
    return "mid"


def add_pace_map_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tempo/position interaction for race-shape fit."""
    result = df.copy()
    result["settling_bin"] = result.get("expected_settling_position", pd.Series("", index=result.index)).map(_settling_bin)

    tempo = result.get("tempo", pd.Series("", index=result.index)).fillna("").astype(str).str.lower()
    fast = tempo.str.contains("fast")
    slow = tempo.str.contains("slow")
    normal = tempo.str.contains("normal")

    fit = pd.Series(0.50, index=result.index, dtype=float)
    bin_series = result["settling_bin"]

    fit = fit.mask(fast & bin_series.eq("leader"), 0.30)
    fit = fit.mask(fast & bin_series.eq("onpace"), 0.45)
    fit = fit.mask(fast & bin_series.eq("mid"), 0.62)
    fit = fit.mask(fast & bin_series.eq("back"), 0.75)

    fit = fit.mask(slow & bin_series.eq("leader"), 0.75)
    fit = fit.mask(slow & bin_series.eq("onpace"), 0.66)
    fit = fit.mask(slow & bin_series.eq("mid"), 0.52)
    fit = fit.mask(slow & bin_series.eq("back"), 0.35)

    fit = fit.mask(normal & bin_series.eq("leader"), 0.55)
    fit = fit.mask(normal & bin_series.eq("onpace"), 0.62)
    fit = fit.mask(normal & bin_series.eq("mid"), 0.58)
    fit = fit.mask(normal & bin_series.eq("back"), 0.52)

    draw = _numeric_series(result, "draw_number", default=np.nan)
    field = _numeric_series(result, "active_field_size", default=np.nan).fillna(
        _numeric_series(result, "field_size", default=np.nan)
    )
    draw_ratio = draw.div(field.where(field > 0))
    fit += np.where(bin_series.eq("leader") & (draw_ratio <= 0.33), 0.05, 0.0)
    fit += np.where(bin_series.eq("back") & (draw_ratio >= 0.66), 0.03, 0.0)
    result["tempo_position_fit"] = fit.clip(0.0, 1.0)
    return result


def add_distance_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """Distance step, trend, and volatility over recent runs."""
    result = df.copy()
    target = _numeric_series(result, "distance_m", default=np.nan)
    step_cols = []
    for i in range(1, 4):
        recent = _numeric_series(result, f"recent_{i}_distance_m", default=np.nan)
        step = target - recent
        col = f"distance_step_{i}"
        result[col] = step
        step_cols.append(col)

    steps = result[step_cols]
    result["distance_step_last"] = result["distance_step_1"]
    result["distance_step_mean_3"] = steps.mean(axis=1, skipna=True)
    result["distance_change_volatility_3"] = steps.std(axis=1, skipna=True).fillna(0.0)

    mean_abs = result["distance_step_mean_3"].abs()
    vol = result["distance_change_volatility_3"]
    score = 1.0 - 0.70 * (mean_abs / 400.0).clip(0.0, 1.0) - 0.30 * (vol / 300.0).clip(0.0, 1.0)
    no_hist = steps.notna().sum(axis=1) == 0
    result["distance_change_score"] = score.mask(no_hist, 0.50).clip(0.0, 1.0)
    return result


def _class_level_scalar(value: object) -> float:
    text = str(value or "").strip().lower()
    if not text:
        return 40.0
    if "group" in text and "1" in text:
        return 100.0
    if "group" in text and "2" in text:
        return 95.0
    if "group" in text and "3" in text:
        return 90.0
    if text in {"one", "g1"}:
        return 100.0
    if text in {"two", "g2"}:
        return 95.0
    if text in {"three", "g3"}:
        return 90.0
    if "listed" in text or text == "lr":
        return 82.0
    if "open" in text:
        return 75.0
    bm_match = re.search(r"bm\s*(\d+)", text)
    if bm_match:
        return 20.0 + float(bm_match.group(1)) * 0.75
    rst_match = re.search(r"rst\s*(\d+)", text)
    if rst_match:
        return 18.0 + float(rst_match.group(1)) * 0.70
    cls_match = re.search(r"cls\s*(\d+)", text)
    if cls_match:
        return 25.0 + float(cls_match.group(1)) * 6.0
    if "maiden" in text or text.startswith("mdn"):
        return 20.0
    if "trial" in text:
        return 15.0
    if "spec" in text:
        return 45.0
    return 45.0


def add_class_movement_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode class rise/drop from today's class versus recent classes."""
    result = df.copy()
    today_grade = result.get("grade", pd.Series("", index=result.index)).map(_class_level_scalar)
    today_class = result.get("class_name", pd.Series("", index=result.index)).map(_class_level_scalar)
    today_race_name = result.get("race_name", pd.Series("", index=result.index)).map(_class_level_scalar)
    today = pd.concat([today_grade, today_class, today_race_name], axis=1).max(axis=1)

    recent_cols = []
    for i in range(1, 7):
        col = f"recent_{i}_class"
        if col in result.columns:
            new_col = f"class_level_recent_{i}"
            result[new_col] = result[col].map(_class_level_scalar)
            recent_cols.append(new_col)

    recent_avg = result[recent_cols].mean(axis=1, skipna=True) if recent_cols else pd.Series(40.0, index=result.index)
    delta = today - recent_avg
    score = (0.5 + (-delta / 40.0)).clip(0.0, 1.0)

    result["class_level_today"] = today
    result["class_level_recent_avg"] = recent_avg
    result["class_delta_today_vs_recent"] = delta
    result["class_movement_score"] = score.fillna(0.5)
    return result


def add_weight_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Weight changes versus recent profile and field."""
    result = df.copy()
    today = _numeric_series(result, "weight_kg", default=np.nan)
    recent_weights = []
    for i in range(1, 7):
        col = f"recent_{i}_weight_kg"
        if col in result.columns:
            recent_weights.append(pd.to_numeric(result[col], errors="coerce"))
    recent_df = pd.concat(recent_weights, axis=1) if recent_weights else pd.DataFrame(index=result.index)
    recent_avg = recent_df.mean(axis=1, skipna=True) if not recent_df.empty else pd.Series(np.nan, index=result.index)
    recent_last = recent_weights[0] if recent_weights else pd.Series(np.nan, index=result.index)

    result["weight_delta_last"] = today - recent_last
    result["weight_vs_recent_avg"] = today - recent_avg
    if "race_id" in result.columns:
        field_mean = today.groupby(result["race_id"]).transform("mean")
        field_std = today.groupby(result["race_id"]).transform("std").replace(0, np.nan)
    else:
        field_mean = pd.Series(today.mean(), index=result.index)
        field_std = pd.Series(today.std(), index=result.index).replace(0, np.nan)
    result["field_relative_weight_z"] = ((today - field_mean) / field_std).fillna(0.0)

    score = (
        0.5
        + 0.6 * (-(result["weight_vs_recent_avg"]) / 4.0).clip(-1.0, 1.0)
        + 0.4 * (-(result["field_relative_weight_z"]) / 2.0).clip(-1.0, 1.0)
    )
    result["weight_trend_score"] = score.clip(0.0, 1.0).fillna(0.5)
    return result


def add_barrier_transition_features(df: pd.DataFrame) -> pd.DataFrame:
    """Runner-level barrier change and historical barrier efficiency."""
    result = df.copy()
    draw = _numeric_series(result, "draw_number", default=np.nan)
    last_barrier = _numeric_series(result, "recent_1_barrier", default=np.nan)
    result["barrier_change_last"] = draw - last_barrier

    eff_sum = pd.Series(0.0, index=result.index)
    weight_sum = pd.Series(0.0, index=result.index)
    for i, recency_weight in enumerate(_RECENCY_WEIGHTS, start=1):
        place = _numeric_series(result, f"recent_{i}_place", default=np.nan)
        total = _numeric_series(result, f"recent_{i}_total_runners", default=np.nan)
        barrier = _numeric_series(result, f"recent_{i}_barrier", default=np.nan)

        quality = pd.Series(
            [_position_quality_scalar(p, n) for p, n in zip(place, total)],
            index=result.index,
            dtype=float,
        )
        gap = (barrier - draw).abs()
        barrier_weight = recency_weight * (1.0 / (1.0 + gap.fillna(99.0)))
        eff_sum += quality * barrier_weight
        weight_sum += barrier_weight

    efficiency = eff_sum.div(weight_sum.where(weight_sum > 0)).fillna(0.5)
    change_penalty = (result["barrier_change_last"].abs() / 8.0).clip(0.0, 1.0).fillna(0.5)
    result["recent_barrier_efficiency"] = efficiency
    result["barrier_transition_score"] = (0.60 * efficiency + 0.40 * (1.0 - change_penalty)).clip(0.0, 1.0)
    return result


def add_recent_sp_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recent SP compression/expansion versus today's live price."""
    result = df.copy()
    live_price = _numeric_series(result, "live_price", default=np.nan)
    recent_avg_sp = _numeric_series(result, "recent_avg_starting_price", default=np.nan)
    result["today_price_vs_recent_sp_avg"] = _safe_log_ratio(live_price, recent_avg_sp)

    sp1 = _numeric_series(result, "recent_1_starting_price", default=np.nan)
    sp2 = _numeric_series(result, "recent_2_starting_price", default=np.nan)
    sp3 = _numeric_series(result, "recent_3_starting_price", default=np.nan)
    trend_raw = ((sp1 - sp2) + (sp2 - sp3)) / 2.0
    denom = recent_avg_sp.where(recent_avg_sp > 0)
    trend_norm = trend_raw.div(denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    result["recent_sp_trend"] = trend_norm

    drift = pd.to_numeric(result["today_price_vs_recent_sp_avg"], errors="coerce").fillna(0.0)
    score = 0.5 + 0.35 * (-drift).clip(-1.0, 1.0) + 0.15 * (-trend_norm).clip(-1.0, 1.0)
    result["recent_sp_score"] = score.clip(0.0, 1.0)
    return result


def add_jockey_continuity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Continuity with recent jockey history."""
    result = df.copy()
    today = result.get("jockey", pd.Series("", index=result.index)).fillna("").astype(str).str.strip().str.lower()
    last = result.get("recent_1_jockey", pd.Series("", index=result.index)).fillna("").astype(str).str.strip().str.lower()
    same = (today != "") & (last != "") & (today == last)
    result["same_jockey_as_last"] = same.astype(float)

    change_count = pd.Series(0.0, index=result.index)
    chain = [today]
    for i in range(1, 7):
        chain.append(
            result.get(f"recent_{i}_jockey", pd.Series("", index=result.index))
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
    for left, right in zip(chain[:-1], chain[1:]):
        valid = (left != "") & (right != "")
        change_count += (valid & (left != right)).astype(float)

    result["jockey_change_count_6"] = change_count
    continuity = 0.70 * result["same_jockey_as_last"] + 0.30 * (1.0 - (change_count / 5.0).clip(0.0, 1.0))
    no_signal = (today == "") & (last == "")
    result["jockey_continuity_score"] = continuity.mask(no_signal, 0.5).clip(0.0, 1.0)
    return result


def _parse_form_positions(value: object) -> tuple[list[int], int]:
    text = str(value or "").strip().upper()
    if not text:
        return [], 0
    positions: list[int] = []
    noise = 0
    for ch in text:
        if ch.isdigit():
            pos = int(ch)
            positions.append(10 if pos == 0 else pos)
            continue
        noise += 1
    return positions[:6], noise


def add_form_string_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse last_six/form_fig into structured form features."""
    result = df.copy()
    source = result.get("last_six", pd.Series("", index=result.index)).fillna("")
    fallback = result.get("form_fig", pd.Series("", index=result.index)).fillna("")
    merged = source.mask(source.astype(str).str.strip() == "", fallback)

    wins = []
    top3 = []
    trend = []
    noise_flags = []
    score = []
    for value in merged:
        positions, noise = _parse_form_positions(value)
        if not positions:
            wins.append(0.0)
            top3.append(0.0)
            trend.append(0.0)
            noise_flags.append(1.0 if noise > 0 else 0.0)
            score.append(0.5)
            continue
        n = float(len(positions))
        win_rate = sum(1 for p in positions if p == 1) / n
        top3_rate = sum(1 for p in positions if p <= 3) / n
        mid = max(1, len(positions) // 2)
        recent = positions[:mid]
        older = positions[mid:] if len(positions[mid:]) > 0 else positions[:mid]
        trend_val = (np.mean(older) - np.mean(recent)) / 10.0
        trend_val = float(np.clip(trend_val, -1.0, 1.0))
        wins.append(win_rate)
        top3.append(top3_rate)
        trend.append(trend_val)
        noise_flags.append(1.0 if noise > 0 else 0.0)
        score.append(float(np.clip(0.45 * win_rate + 0.35 * top3_rate + 0.20 * (0.5 + 0.5 * trend_val), 0.0, 1.0)))

    result["form_win_count_6"] = wins
    result["form_top3_count_6"] = top3
    result["form_trend_slope"] = trend
    result["form_noise_flag"] = noise_flags
    result["form_string_score"] = score
    return result


def add_travel_region_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simple travel/cross-region heuristics from trainer and race locations."""
    result = df.copy()
    trainer_norm = _normalise_text_series(result.get("trainer_location", pd.Series("", index=result.index)))
    track_norm = _normalise_text_series(result.get("competition_name", pd.Series("", index=result.index)))
    race_country = result.get("country", pd.Series("", index=result.index)).fillna("").astype(str).str.lower()
    runner_country = result.get("runner_country", pd.Series("", index=result.index)).fillna("").astype(str).str.upper()

    cross_region = (trainer_norm != "") & (track_norm != "") & (trainer_norm != track_norm)
    interstate = race_country.eq("australia") & cross_region
    international = (
        race_country.eq("australia") & ~runner_country.isin(["", "AUS", "NZ"])
    ) | (
        race_country.eq("new zealand") & ~runner_country.isin(["", "NZ", "AUS"])
    )

    result["cross_region_flag"] = cross_region.astype(float)
    result["interstate_travel_flag"] = interstate.astype(float)
    result["international_runner_flag"] = international.astype(float)
    result["travel_score"] = (
        1.0
        - 0.35 * result["cross_region_flag"]
        - 0.20 * result["interstate_travel_flag"]
        - 0.15 * result["international_runner_flag"]
    ).clip(0.0, 1.0)
    return result


def add_equipment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Blinkers state features."""
    result = df.copy()
    raw = result.get("blinkers", pd.Series("", index=result.index)).fillna("").astype(str).str.strip().str.lower()
    on = raw.isin(["1", "true", "yes", "on", "t"])
    change = raw.str.contains("first") | raw.str.contains("onoff") | raw.str.contains("offon")
    result["blinkers_on_flag"] = on.astype(float)
    result["blinkers_change_flag"] = change.astype(float)
    result["equipment_score"] = (0.55 + 0.20 * result["blinkers_on_flag"] + 0.25 * result["blinkers_change_flag"]).clip(0.0, 1.0)
    return result


def add_pedigree_priors(df: pd.DataFrame) -> pd.DataFrame:
    """Sire/dam smoothed priors by distance band and track condition."""
    result = df.copy()
    sire = result.get("sire", pd.Series("", index=result.index)).fillna("").astype(str).str.strip().str.lower()
    dam = result.get("dam", pd.Series("", index=result.index)).fillna("").astype(str).str.strip().str.lower()
    winner_raw = _numeric_series(result, "is_winner", default=np.nan)
    known = winner_raw.notna()
    winner = winner_raw.fillna(0.0)

    prior = float(winner[known].mean()) if known.any() else 0.10
    prior = min(max(prior, 0.05), 0.20)
    prior_starts = 20.0

    stats = pd.DataFrame({"sire": sire, "dam": dam, "is_winner": winner, "known": known})
    sire_known = stats[stats["known"]]
    dam_known = stats[stats["known"]]

    sire_tot = sire_known.groupby("sire")["is_winner"].agg(["sum", "count"]) if not sire_known.empty else pd.DataFrame(columns=["sum", "count"])
    dam_tot = dam_known.groupby("dam")["is_winner"].agg(["sum", "count"]) if not dam_known.empty else pd.DataFrame(columns=["sum", "count"])

    dist = result.get("distance_band", pd.Series("UNKNOWN", index=result.index)).fillna("UNKNOWN").astype(str)
    cond = result.get("track_status", pd.Series("UNKNOWN", index=result.index)).fillna("UNKNOWN").astype(str)
    sire_dist_key = sire + "|" + dist
    sire_cond_key = sire + "|" + cond
    known_df = pd.DataFrame(
        {
            "sire_dist_key": sire_dist_key,
            "sire_cond_key": sire_cond_key,
            "is_winner": winner,
            "known": known,
        }
    )
    known_df = known_df[known_df["known"]]

    if known_df.empty:
        sire_dist_stats = pd.DataFrame(columns=["sum", "count"])
        sire_cond_stats = pd.DataFrame(columns=["sum", "count"])
    else:
        sire_dist_stats = known_df.groupby("sire_dist_key")["is_winner"].agg(["sum", "count"])
        sire_cond_stats = known_df.groupby("sire_cond_key")["is_winner"].agg(["sum", "count"])

    def _smoothed(sum_series: pd.Series, count_series: pd.Series) -> pd.Series:
        return (sum_series + prior * prior_starts) / (count_series + prior_starts)

    sire_rate_map = _smoothed(sire_tot.get("sum", pd.Series(dtype=float)), sire_tot.get("count", pd.Series(dtype=float)))
    dam_rate_map = _smoothed(dam_tot.get("sum", pd.Series(dtype=float)), dam_tot.get("count", pd.Series(dtype=float)))
    sire_dist_rate_map = _smoothed(
        sire_dist_stats.get("sum", pd.Series(dtype=float)),
        sire_dist_stats.get("count", pd.Series(dtype=float)),
    )
    sire_cond_rate_map = _smoothed(
        sire_cond_stats.get("sum", pd.Series(dtype=float)),
        sire_cond_stats.get("count", pd.Series(dtype=float)),
    )

    result["sire_win_rate_by_distance_band"] = sire_dist_key.map(sire_dist_rate_map).fillna(sire.map(sire_rate_map)).fillna(prior)
    result["sire_win_rate_by_track_condition"] = sire_cond_key.map(sire_cond_rate_map).fillna(sire.map(sire_rate_map)).fillna(prior)
    result["dam_win_rate"] = dam.map(dam_rate_map).fillna(prior)
    result["pedigree_score"] = (
        0.45 * result["sire_win_rate_by_distance_band"]
        + 0.35 * result["sire_win_rate_by_track_condition"]
        + 0.20 * result["dam_win_rate"]
    ).clip(0.0, 1.0)
    return result


def add_suitability_score(df: pd.DataFrame) -> pd.DataFrame:
    """Combine distance, track, and condition win rates."""
    result = df.copy()
    wet_status = result["track_status"].fillna("").str.lower()
    distance_rate = _smooth_rate(
        result["distance_wins"], result["distance_starts"], prior=0.30, prior_starts=4
    )
    track_rate = _smooth_rate(
        result["track_wins"], result["track_starts"], prior=0.30, prior_starts=4
    )
    good_rate = _safe_rate(result["good_wins"], result["good_starts"])
    soft_rate = _safe_rate(result["soft_wins"], result["soft_starts"])
    heavy_rate = _safe_rate(result["heavy_wins"], result["heavy_starts"])

    condition_rate = np.where(
        wet_status.str.contains("heavy"),
        heavy_rate,
        np.where(wet_status.str.contains("soft"), soft_rate, good_rate),
    )
    result["suitability_score"] = (
        pd.Series(distance_rate, index=result.index)
        + pd.Series(track_rate, index=result.index)
        + pd.Series(condition_rate, index=result.index)
    ) / 3.0
    result["suitability_score"] = pd.to_numeric(result["suitability_score"], errors="coerce").fillna(0.0)
    return result

# Recency weights for recent_1 (most recent) → recent_6 (oldest). Sum = 1.0.
_RECENCY_WEIGHTS = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
_PLACE_QUALITY_PRIOR = 0.30   # neutral score when position is unknown
_PLACE_QUALITY_CAP = 12       # fallback field-size cap when total_runners is missing


def _position_quality_scalar(place, total_runners) -> float:
    """Convert a single finish position to a 0–1 quality score.

    Uses field-size normalisation when total_runners is available so that
    finishing 2nd in a 15-runner race scores better than 2nd in a 5-runner
    race.  Falls back to a fixed cap of 12 when total_runners is missing.
    Returns the neutral prior (0.30) for missing/invalid positions.
    """
    try:
        p = int(place)
    except (TypeError, ValueError):
        return _PLACE_QUALITY_PRIOR
    if p <= 0:
        return _PLACE_QUALITY_PRIOR

    try:
        n = int(total_runners)
        if n >= 2:
            return float(np.clip((n - p) / (n - 1), 0.0, 1.0))
    except (TypeError, ValueError):
        pass

    # fallback: fixed cap
    return max(0.0, 1.0 - (p - 1) / (_PLACE_QUALITY_CAP - 1))


def add_recent_place_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """Recency-weighted position quality score from the last 6 individual runs.

    For each run slot (1 = most recent … 6 = oldest):
      - If place is known: score = (total_runners - place) / (total_runners - 1)
        normalised within the field, clipped [0, 1].
      - If place is unknown: score = neutral prior (0.30) — not treated as bad.
      - If recent class is available: score is scaled by a simple class factor
        relative to today's class (better runs in stronger class count more).

    Final score = weighted average using ``_RECENCY_WEIGHTS``.
    A runner with no form history scores 0.30 (neutral).
    A runner that won all 6 last starts scores 1.0.
    """
    result = df.copy()
    today_grade = result.get("grade", pd.Series("", index=result.index)).map(_class_level_scalar)
    today_class = result.get("class_name", pd.Series("", index=result.index)).map(_class_level_scalar)
    today_race_name = result.get("race_name", pd.Series("", index=result.index)).map(_class_level_scalar)
    today_level = pd.concat([today_grade, today_class, today_race_name], axis=1).max(axis=1)
    scores = pd.Series(0.0, index=result.index)
    for i, weight in enumerate(_RECENCY_WEIGHTS, start=1):
        place_col  = f"recent_{i}_place"
        runners_col = f"recent_{i}_total_runners"
        places   = pd.to_numeric(result.get(place_col,  pd.Series(np.nan, index=result.index)), errors="coerce")
        runners  = pd.to_numeric(result.get(runners_col, pd.Series(np.nan, index=result.index)), errors="coerce")
        slot_score = pd.Series(
            [_position_quality_scalar(p, r) for p, r in zip(places, runners)],
            index=result.index,
        )
        class_col = f"recent_{i}_class"
        if class_col in result.columns:
            recent_level = result[class_col].map(_class_level_scalar)
            class_factor = (1.0 + (recent_level - today_level) / 100.0).clip(0.7, 1.3)
            slot_score = (slot_score * class_factor).clip(0.0, 1.0)
        scores += weight * slot_score
    result["recent_place_quality_score"] = scores.clip(0.0, 1.0)
    return result


def add_recent_form_score(df: pd.DataFrame) -> pd.DataFrame:
    """Blended recent-form score (0–1).

    Combines two complementary signals at 50/50:
      - Aggregate stats (win_rate_5, top3_rate_5, avg_place): long-run consistency.
      - ``recent_place_quality_score``: field-normalised quality of each of the
        last 6 individual runs, recency-weighted.
    """
    result = add_recent_place_quality_score(df.copy())

    recent_win_rate_5  = pd.to_numeric(result["recent_win_rate_5"],  errors="coerce")
    recent_top3_rate_5 = pd.to_numeric(result["recent_top3_rate_5"], errors="coerce")
    recent_avg_place   = pd.to_numeric(result["recent_avg_place"],   errors="coerce")
    recent_avg_place_3 = pd.to_numeric(result["recent_avg_place_3"], errors="coerce")

    agg_score = (
        0.4  * recent_win_rate_5.fillna(0.0)
        + 0.3  * recent_top3_rate_5.fillna(0.0)
        + 0.15 * (1.0 / (1.0 + recent_avg_place.fillna(10.0)))
        + 0.15 * (1.0 / (1.0 + recent_avg_place_3.fillna(10.0)))
    )
    all_null = (
        recent_win_rate_5.isna()
        & recent_top3_rate_5.isna()
        & recent_avg_place.isna()
        & recent_avg_place_3.isna()
    )
    agg_score = agg_score.mask(all_null, 0.0)

    pos_score = result["recent_place_quality_score"]
    blended   = 0.5 * agg_score + 0.5 * pos_score
    result["recent_form_score"] = pd.to_numeric(blended, errors="coerce").fillna(0.0)
    return result



def add_connection_score(df: pd.DataFrame) -> pd.DataFrame:
    """Build the version-1 horse-jockey connection score."""
    result = df.copy()
    result["connection_score"] = _smooth_rate(
        result["horse_jockey_wins"], result["horse_jockey_starts"], prior=0.15, prior_starts=4
    )
    return result


def add_margin_score(df: pd.DataFrame) -> pd.DataFrame:
    """Score based on recent beaten margins. Smaller margin = ran closer = better score.

    avg_3 cap: 8 lengths (beyond this = maximum penalty)
    best cap:  4 lengths (a 4L+ best run gets no credit)
    Weighting: 80% avg_3 (consistency), 20% best (ceiling of ability)
    """
    result = df.copy()
    avg_3 = pd.to_numeric(result["recent_avg_margin_3"], errors="coerce").fillna(8.0)
    best = pd.to_numeric(result["recent_best_margin"], errors="coerce").fillna(4.0)

    avg_3_score = 1.0 - (avg_3.clip(0.0, 8.0) / 8.0)
    best_score = 1.0 - (best.clip(0.0, 4.0) / 4.0)

    result["margin_score"] = (0.80 * avg_3_score + 0.20 * best_score).clip(0.0, 1.0)
    return result


def add_freshness_score(df: pd.DataFrame) -> pd.DataFrame:
    """Fitness/freshness score based on days since last run and first/second-up rates."""
    result = df.copy()
    days = pd.to_numeric(result["recent_days_since_last_run"], errors="coerce").fillna(21.0)
    day_score = np.exp(-0.5 * ((days - 21.0) / 20.0) ** 2)
    first_up_rate = _smooth_rate(
        result["first_up_wins"], result["first_up_starts"], prior=0.20, prior_starts=4
    )
    second_up_rate = _smooth_rate(
        result["second_up_wins"], result["second_up_starts"], prior=0.20, prior_starts=4
    )
    freshness = 0.60 * day_score + 0.25 * first_up_rate + 0.15 * second_up_rate
    result["freshness_score"] = (
        pd.to_numeric(freshness, errors="coerce").fillna(0.5).clip(0.0, 1.0)
    )
    return result


def add_class_score(df: pd.DataFrame) -> pd.DataFrame:
    """Class quality score from career earnings, place rate, and win rate."""
    result = df.copy()
    career_win_rate = _safe_rate(result["career_wins"], result["career_starts"])
    place_rate = pd.to_numeric(result["place_percentage"], errors="coerce").fillna(0.0).clip(0, 100) / 100.0
    prize = pd.to_numeric(result["prize_money"], errors="coerce").fillna(0.0).clip(lower=0)
    prize_log = np.log1p(prize)
    prize_min = prize_log.min()
    prize_max = prize_log.max()
    if prize_max > prize_min:
        prize_norm = (prize_log - prize_min) / (prize_max - prize_min)
    else:
        prize_norm = pd.Series(0.0, index=result.index)
    class_score = 0.40 * career_win_rate + 0.35 * place_rate + 0.25 * prize_norm
    result["class_score"] = pd.to_numeric(class_score, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return result


def add_market_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Rank runners within each race by live price."""
    result = df.copy()
    result["market_rank"] = result.groupby("race_id")["live_price"].rank(
        method="min", ascending=True
    )
    return result


def add_price_band(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Bucket live prices into fixed reporting bands."""
    del config
    result = df.copy()
    prices = pd.to_numeric(result["live_price"], errors="coerce")
    result["price_band"] = np.select(
        [
            prices.isna() | (prices <= 0),
            prices < 3,
            (prices >= 3) & (prices < 6),
            (prices >= 6) & (prices < 9),
            (prices >= 9) & (prices < 12),
            (prices >= 12) & (prices <= 20),
            prices > 20,
        ],
        ["NO_PRICE", "<3", "3-6", "6-9", "9-12", "12-20", ">20"],
        default="NO_PRICE",
    )
    return result


def add_distance_band(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket races into sprint, middle, staying, or unknown."""
    result = df.copy()
    distance = pd.to_numeric(result["distance_m"], errors="coerce")
    result["distance_band"] = np.select(
        [distance.isna(), distance < 1400, (distance >= 1400) & (distance < 2000), distance >= 2000],
        ["UNKNOWN", "sprint", "middle", "staying"],
        default="UNKNOWN",
    )
    return result


def add_form_recency_flag(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Flag runners with insufficient recent-runs evidence."""
    result = df.copy()
    recent_runs_count = pd.to_numeric(result["recent_runs_count"], errors="coerce")
    result["has_sparse_recent_form"] = recent_runs_count.isna() | (
        recent_runs_count < config["min_recent_form_count"]
    )
    return result


def add_draw_bias_score(df: pd.DataFrame, draw_bias_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Score runner by historical win rate for this draw/barrier at track × distance × condition.

    Uses Bayesian smoothing so draws with few starts regress to the neutral prior.
    A neutral prior of 0.25 is used (unknown draw = no edge over field average).
    When draw_bias_df is None or the runner has no matching draw record, returns 0.25.
    """
    result = df.copy()
    if draw_bias_df is None or draw_bias_df.empty:
        result["draw_bias_score"] = 0.25
        result["draw_bias_starts"] = 0
        result["draw_bias_win_rate"] = float("nan")
        return result

    lookup = draw_bias_df.copy()
    lookup["track_name"] = lookup["track_name"].astype(str).str.strip()
    lookup["track_condition"] = lookup["track_condition"].astype(str).str.strip()
    lookup["distance_m"] = pd.to_numeric(lookup["distance_m"], errors="coerce")
    lookup["draw_number"] = pd.to_numeric(lookup["draw_number"], errors="coerce")

    join_df = result[["competition_name", "distance_m", "track_status", "draw_number"]].copy()
    join_df["_track_name"] = join_df["competition_name"].fillna("").str.strip().replace("", "(blank)")
    join_df["_track_condition"] = join_df["track_status"].fillna("").str.strip().replace("", "(blank)")
    join_df["_distance_m"] = pd.to_numeric(join_df["distance_m"], errors="coerce")
    join_df["_draw_number"] = pd.to_numeric(join_df["draw_number"], errors="coerce")

    merged = join_df.merge(
        lookup.rename(
            columns={
                "track_name": "_track_name",
                "distance_m": "_distance_m",
                "track_condition": "_track_condition",
                "draw_number": "_draw_number",
            }
        ),
        on=["_track_name", "_distance_m", "_track_condition", "_draw_number"],
        how="left",
    )

    wins = pd.to_numeric(merged["wins"], errors="coerce").fillna(0.0)
    starts = pd.to_numeric(merged["starts"], errors="coerce").fillna(0.0)

    result["draw_bias_score"] = _smooth_rate(wins, starts, prior=0.25, prior_starts=20).values
    result["draw_bias_starts"] = starts.values
    result["draw_bias_win_rate"] = pd.to_numeric(merged["win_rate_pct"], errors="coerce").values
    return result


def add_jockey_score(
    df: pd.DataFrame,
    jockey_stats_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score runner by jockey's overall historical win rate across all rides.

    Uses Bayesian smoothing (prior=0.15, prior_starts=8) so low-sample jockeys
    regress toward an average overall jockey win rate. Unknown jockeys return
    the prior.
    """
    result = df.copy()
    if jockey_stats_df is None or jockey_stats_df.empty:
        result["jockey_score"] = 0.15
        result["jockey_starts"] = 0
        result["jockey_win_rate"] = float("nan")
        return result

    lookup = jockey_stats_df.copy()
    lookup["jockey_name"] = lookup["jockey_name"].astype(str).str.strip()
    lookup["wins"] = pd.to_numeric(lookup["wins"], errors="coerce").fillna(0.0)
    lookup["starts"] = pd.to_numeric(lookup["starts"], errors="coerce").fillna(0.0)

    join_df = result[["jockey"]].copy()
    join_df["_jockey_name"] = join_df["jockey"].fillna("").str.strip().replace("", "(blank)")

    merged = join_df.merge(
        lookup.rename(columns={"jockey_name": "_jockey_name"}),
        on="_jockey_name",
        how="left",
    )

    wins = pd.to_numeric(merged["wins"], errors="coerce").fillna(0.0)
    starts = pd.to_numeric(merged["starts"], errors="coerce").fillna(0.0)

    result["jockey_score"] = _smooth_rate(wins, starts, prior=0.15, prior_starts=8).values
    result["jockey_starts"] = starts.values
    result["jockey_win_rate"] = pd.to_numeric(merged["win_rate_pct"], errors="coerce").values
    return result



def add_trainer_score(
    df: pd.DataFrame,
    trainer_stats_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score runner by trainer's overall historical win rate across all runners.

    Uses Bayesian smoothing (prior=0.15, prior_starts=8) so low-sample trainers
    regress toward an average overall trainer win rate. Unknown trainers return
    the prior.
    """
    result = df.copy()
    if trainer_stats_df is None or trainer_stats_df.empty:
        result["trainer_score"] = 0.15
        result["trainer_starts"] = 0
        result["trainer_win_rate"] = float("nan")
        return result

    lookup = trainer_stats_df.copy()
    lookup["trainer_name"] = lookup["trainer_name"].astype(str).str.strip()
    lookup["wins"] = pd.to_numeric(lookup["wins"], errors="coerce").fillna(0.0)
    lookup["starts"] = pd.to_numeric(lookup["starts"], errors="coerce").fillna(0.0)

    join_df = result[["trainer"]].copy()
    join_df["_trainer_name"] = join_df["trainer"].fillna("").str.strip().replace("", "(blank)")

    merged = join_df.merge(
        lookup.rename(columns={"trainer_name": "_trainer_name"}),
        on="_trainer_name",
        how="left",
    )

    wins = pd.to_numeric(merged["wins"], errors="coerce").fillna(0.0)
    starts = pd.to_numeric(merged["starts"], errors="coerce").fillna(0.0)

    result["trainer_score"] = _smooth_rate(wins, starts, prior=0.15, prior_starts=8).values
    result["trainer_starts"] = starts.values
    result["trainer_win_rate"] = pd.to_numeric(merged["win_rate_pct"], errors="coerce").values
    return result
