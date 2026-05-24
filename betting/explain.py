"""Score explanation and rich display for inspect_race."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .scoring import get_effective_weights
from .filters import candidate_mask


def _fmt(val) -> str:
    """Format a rating value — show the number or 'NULL' if missing."""
    try:
        return f"{float(val):.1f}"
    except (TypeError, ValueError):
        return "NULL"


def _get_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    """Read a float from a row, using *default* only when the value is None/NaN.

    Unlike ``row.get(key, default) or default``, this does NOT treat 0.0 as
    missing — which would silently replace a genuine zero score with the default.
    """
    val = row.get(key)
    if val is None:
        return default
    try:
        f = float(val)
        return default if pd.isna(f) else f
    except (TypeError, ValueError):
        return default


def _fmt_count(val: float) -> str:
    return f"{val:g}"


def _fmt_record(wins: float, starts: float, prior: float | None = None) -> str:
    record = f"{_fmt_count(wins)}/{_fmt_count(starts)}"
    if starts == 0 and prior is not None:
        return f"{record}(prior={prior:.2f})"
    return record


def _parse_time_to_seconds(value) -> float:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return float("nan")
    parts = [piece for piece in text.replace(":", " ").replace(".", " ").split() if piece.isdigit()]
    if len(parts) >= 3:
        return float(parts[-3]) * 60.0 + float(parts[-2]) + float(parts[-1]) / 100.0
    if len(parts) == 2:
        return float(parts[0]) + float(parts[1]) / 100.0
    if len(parts) == 1:
        return float(parts[0])
    return float("nan")


def _speed_rank_norm(rank: float, field_size: float) -> float:
    if pd.isna(rank) or pd.isna(field_size) or field_size <= 1:
        return 0.0
    return 1.0 - ((rank - 1.0) / (field_size - 1.0))


# (weight_key, score_col, short_label)
COMPONENT_DEFS = [
    ("weight_speed_rating", "speed_score_norm", "Speed"),
    ("weight_recent_form", "recent_form_score", "Form"),
    ("weight_suitability", "suitability_score", "Suit"),
    ("weight_connections", "connection_score", "Conn"),
    ("weight_market_sanity", "market_sanity_score", "Mkt"),
    ("weight_steam", "market_movement_score", "Steam"),
    ("weight_margin", "margin_score", "Margin"),
    ("weight_freshness", "freshness_score", "Fresh"),
    ("weight_class", "class_score", "Class"),
    ("weight_draw_bias", "draw_bias_score", "Draw"),
    ("weight_jockey", "jockey_score", "Jockey"),
    ("weight_trainer", "trainer_score", "Trainer"),
]


def transparent_component_score(row: pd.Series, config: dict) -> float:
    """Return the weighted transparent score before any meta-model blend."""
    eff_weights, _ = get_effective_weights(config)
    total = 0.0
    for wkey, scol, _ in COMPONENT_DEFS:
        score = _get_float(row, scol, 0.0)
        total += eff_weights.get(wkey, 0.0) * score
    return total


def _component_contributions(row: pd.Series, config: dict) -> list[tuple[str, float, float]]:
    eff_weights, _ = get_effective_weights(config)
    values: list[tuple[str, float, float]] = []
    for wkey, scol, label in COMPONENT_DEFS:
        score = _get_float(row, scol, 0.0)
        contribution = eff_weights.get(wkey, 0.0) * score
        values.append((label, contribution, score))
    return values


def _format_driver_summary(row: pd.Series, config: dict) -> tuple[str, str]:
    contributions = _component_contributions(row, config)
    top = sorted(contributions, key=lambda item: item[1], reverse=True)[:3]
    low = sorted(contributions, key=lambda item: item[1])[:3]
    top_text = ", ".join(f"{label} {contrib:.4f} (score {score:.3f})" for label, contrib, score in top)
    low_text = ", ".join(f"{label} {contrib:.4f} (score {score:.3f})" for label, contrib, score in low)
    return top_text, low_text


def _candidate_rejection_reason(row: pd.Series, config: dict) -> str:
    reasons: list[str] = []
    raw_edge = _get_float(row, "raw_edge", 0.0)
    min_raw_edge = float(config.get("min_raw_edge", 0.0))
    if raw_edge < min_raw_edge:
        reasons.append(f"raw edge {raw_edge:.1%} < {min_raw_edge:.1%}")

    ev = _get_float(row, "ev", 0.0)
    min_ev = float(config.get("min_ev", 0.0))
    if ev < min_ev:
        reasons.append(f"EV {ev:.1%} < {min_ev:.1%}")

    price = _get_float(row, "live_price", 0.0)
    min_price = float(config.get("min_price", 0.0))
    max_price = float(config.get("max_price", float("inf")))
    if not (min_price <= price <= max_price):
        reasons.append(f"price {price:.2f} outside {min_price:.2f}-{max_price:.2f}")

    rank = _get_float(row, "model_rank", 9999.0)
    max_rank = float(config.get("max_model_rank", 9999.0))
    if rank > max_rank:
        reasons.append(f"rank {int(rank)} > {int(max_rank)}")

    model_prob = _get_float(row, "model_prob", 0.0)
    min_model_prob = float(config.get("min_model_probability", 0.0))
    if model_prob < min_model_prob:
        reasons.append(f"model {model_prob:.1%} < {min_model_prob:.1%}")

    score_gap = _get_float(row, "model_score_gap_to_next", 0.0)
    min_score_gap = float(config.get("min_score_gap_to_next", 0.0))
    if score_gap < min_score_gap:
        reasons.append(f"gap {score_gap:.3f} < {min_score_gap:.3f}")

    if not reasons:
        return "QUALIFIED"
    return "; ".join(reasons)


def build_ranking_table(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Return a display DataFrame sorted by model_rank with columns:
    Rank, Runner, Price, weighted contribution per component, Score, ModelP%, FairMktP%, FairEdge%, Bet?
    Each weighted column = weight * component_score for the transparent base model.
    `Score` is the final live score after any meta-model blend.
    """
    from .filters import candidate_mask

    eff_weights, _ = get_effective_weights(config)
    rows = []
    qualifies_mask = candidate_mask(df, config)
    for _, r in df.sort_values("model_rank").iterrows():
        fp = r.get("finish_place")
        fp_str = str(int(fp)) if fp is not None and pd.notna(fp) else "-"
        row = {
            "Rank": int(r["model_rank"]),
            "Runner": f"#{int(r['runner_number'])} {r['runner_name']}",
            "Price": r["live_price"],
            "Place": fp_str,
        }
        score_check = 0.0
        for wkey, scol, label in COMPONENT_DEFS:
            w = eff_weights.get(wkey, 0.0)
            s = _get_float(r, scol, 0.0)
            contribution = round(w * s, 4)
            row[label] = contribution
            score_check += contribution
        row["Base"] = round(score_check, 4)
        row["Meta"] = round(_get_float(r, "meta_model_score", 0.0), 4)
        row["Score"] = round(float(r.get("model_score", score_check)), 4)
        model_p = float(r.get("model_prob", r.get("raw_model_prob", 0))) * 100
        fair_mkt_p = float(r.get("fair_market_prob", r.get("market_implied_prob", 0))) * 100
        raw_mkt_p = float(r.get("raw_market_prob", 0)) * 100
        fair_edge = float(r.get("fair_edge", r.get("edge", 0))) * 100
        raw_edge = float(r.get("raw_edge", 0)) * 100
        ev = float(r.get("ev", 0)) * 100
        row["ModelP%"] = round(model_p, 1)
        row["RawMktP%"] = round(raw_mkt_p, 1)
        row["FairMktP%"] = round(fair_mkt_p, 1)
        row["RawEdge%"] = round(raw_edge, 1)
        row["FairEdge%"] = round(fair_edge, 1)
        row["EV%"] = round(ev, 1)
        qualifies = bool(qualifies_mask.get(r.name, False))
        row["Bet?"] = "✓ BET" if qualifies else "✗"
        rows.append(row)
    return pd.DataFrame(rows)


def format_weight_header(config: dict) -> str:
    """Return a one-line weight summary string."""
    eff_weights, raw_total = get_effective_weights(config)
    normalized = raw_total > 1.0
    parts = []
    for wkey, _, label in COMPONENT_DEFS:
        parts.append(f"{label}={eff_weights.get(wkey, 0.0):.4f}")
    mode = "yes" if normalized else "no"
    return f"  weights(effective, raw_total={raw_total:.4f}, normalized={mode}): " + "  ".join(parts)


def format_runner_verbose(row: pd.Series, config: dict, field_df: "pd.DataFrame | None" = None) -> str:
    """Return a multi-line breakdown for one runner (--verbose mode).

    Args:
        row: Single runner Series from the scored pipeline DataFrame.
        config: Betting config dict (weights, filters).
        field_df: Full scored field DataFrame — used to show field-level
                  speed normalisation context (min/max/fallback counts).
    """
    lines = []
    eff_weights, _ = get_effective_weights(config)
    name = f"#{int(row['runner_number'])} {row['runner_name']}"
    lines.append(f"\n  ── {name} ──────────────────────────────────────────────")
    top_drivers, low_drivers = _format_driver_summary(row, config)
    qualifies = bool(candidate_mask(pd.DataFrame([row]), config).iloc[0])
    verdict = "QUALIFIED" if qualifies else _candidate_rejection_reason(row, config)
    model_prob_now = _get_float(row, "model_prob", _get_float(row, "raw_model_prob", 0.0))
    market_prob_now = _get_float(row, "raw_market_prob", 0.0)
    fair_market_prob_now = _get_float(row, "fair_market_prob", _get_float(row, "market_implied_prob", 0.0))
    fair_odds = 1.0 / model_prob_now if model_prob_now > 0 else float("inf")
    market_odds = _get_float(row, "live_price", 0.0)
    market_fair_odds = 1.0 / fair_market_prob_now if fair_market_prob_now > 0 else float("inf")
    leader_gap = None
    if field_df is not None and "model_score" in field_df.columns:
        leader_score = pd.to_numeric(field_df["model_score"], errors="coerce").max()
        leader_gap = leader_score - _get_float(row, "model_score", 0.0)
    gap_to_next = _get_float(row, "model_score_gap_to_next", 0.0)
    lines.append(f"  Why here: strongest drivers = {top_drivers}")
    lines.append(f"            weakest support = {low_drivers}")
    if leader_gap is not None:
        lines.append(
            f"            rank context: model_rank={int(_get_float(row, 'model_rank', 0))}"
            f"  gap_to_next={gap_to_next:.4f}  behind_leader={leader_gap:.4f}"
        )
    lines.append(
        f"            odds view: model {model_prob_now:.1%} -> fair odds {fair_odds:.2f}"
        f"  | market {market_odds:.2f} -> raw {market_prob_now:.1%} / fair {fair_market_prob_now:.1%}"
        f" ({market_fair_odds:.2f})"
    )
    lines.append(f"            bet verdict: {verdict}")

    # ── Speed ──────────────────────────────────────────────────────────────
    track_status = str(row.get("track_status", "Unknown"))
    track_lower = track_status.lower()
    is_wet = "soft" in track_lower or "heavy" in track_lower
    preferred_col = "wet_rating" if is_wet else "dry_rating"
    fallback_col  = "dry_rating" if is_wet else "wet_rating"

    raw_preferred = row.get(preferred_col)
    raw_fallback  = row.get(fallback_col)
    raw_cr        = _get_float(row, "condition_rating", 0.0)
    historical_speed = _get_float(row, "historical_speed_score", 0.0)
    speed_feature = _get_float(row, "speed_feature_score", raw_cr)
    speed_source = str(row.get("speed_feature_source", "condition_rating") or "condition_rating")
    norm_cr       = _get_float(row, "speed_score_norm", 0.0)
    speed_confidence = _get_float(row, "speed_feature_confidence", 1.0)

    # Which column was actually used?
    try:
        pref_val = float(raw_preferred)
        used_col = preferred_col
        used_val = pref_val
        used_fallback = False
    except (TypeError, ValueError):
        used_col = fallback_col
        used_val = float(raw_fallback) if raw_fallback is not None else 0.0
        used_fallback = True

    fallback_note = f"  ⚠ {preferred_col} missing, fell back to {fallback_col}" if used_fallback else ""

    # Field-level rank context for normalisation
    field_note = ""
    if field_df is not None and "speed_feature_score" in field_df.columns:
        import pandas as _pd
        speed_series = _pd.to_numeric(field_df["speed_feature_score"], errors="coerce")
        n_field = int(speed_series.notna().sum())
        if n_field > 1:
            field_mean = float(speed_series.mean())
            base_speed_component = 1.0 / (1.0 + np.exp(-(speed_feature - field_mean) / 0.60))
            speed_component = base_speed_component * speed_confidence + 0.5 * (1.0 - speed_confidence)
            field_note = (
                f"\n         field_size={n_field}"
                f"  field_mean={field_mean:.2f}"
                f"  →  sigmoid(({speed_feature:.2f} - {field_mean:.2f}) / 0.60)"
                f"  =  {base_speed_component:.3f}"
                f"  →  confidence={speed_confidence:.2f}  final={speed_component:.3f}"
            )
            if is_wet:
                n_wet_null = field_df["wet_rating"].isna().sum() if "wet_rating" in field_df.columns else 0
                if n_wet_null:
                    field_note += f"\n         ⚠ {n_wet_null}/{len(field_df)} runners missing wet_rating (fell back to dry_rating)"
        else:
            field_note = f"\n         field_size={n_field}  (single runner → score 0.0)"

    lines.append(
        f"  Speed  [{track_status}]  hist={historical_speed:.2f}  source={speed_source}"
        f"  {preferred_col}={_fmt(raw_preferred)}  {fallback_col}={_fmt(raw_fallback)}"
        f"  →  using {used_col}={used_val:.1f}{fallback_note}"
    )
    lines.append(
        f"         speed_feature={speed_feature:.1f}"
        f"  confidence={speed_confidence:.2f}"
        f"  →  speed_score_norm={norm_cr:.3f}  ×  {eff_weights['weight_speed_rating']:.4f}"
        f"  =  {eff_weights['weight_speed_rating'] * norm_cr:.4f}"
        f"{field_note}"
    )
    speed_parts = []
    for i in range(1, 7):
        t = row.get(f"recent_{i}_time")
        d = _get_float(row, f"recent_{i}_distance_m", 0.0)
        secs = _parse_time_to_seconds(t)
        if d > 0 and secs > 0:
            pace = d / secs
            speed_parts.append(f"r{i}:{t}@{int(d)}m={pace:.2f}m/s")
        elif t not in (None, "", "nan") or d > 0:
            speed_parts.append(f"r{i}:{t or '-'}@{int(d) if d else '-'}m")
    if speed_parts:
        lines.append(f"         speed_history: {'  '.join(speed_parts)}")
        same_dist = _get_float(row, "speed_peak_same_distance", 0.0)
        avg_same = _get_float(row, "speed_avg_same_distance", 0.0)
        near_adj = _get_float(row, "speed_recent_weighted_distance_adjusted", 0.0)
        nearby_avg = _get_float(row, "speed_avg_nearby_distance", 0.0)
        same_count = _get_float(row, "speed_sample_count_same_distance", 0.0)
        nearby_count = _get_float(row, "speed_sample_count_nearby", 0.0)
        consistency = _get_float(row, "speed_consistency", 0.0)
        consistency_std = _get_float(row, "speed_consistency_std", 0.0)
        lines.append(
            f"         speed_calc: same_dist_peak={same_dist:.2f}  same_dist_avg={avg_same:.2f}"
            f"  nearby_avg={nearby_avg:.2f}  adjusted={near_adj:.2f}"
            f"  consistency_std={consistency_std:.3f}  consistency={consistency:.2f}"
            f"  samples_same={same_count:.1f}  samples_near={nearby_count:.1f}"
        )

    # New engineered signals
    pace_fit = _get_float(row, "tempo_position_fit", 0.5)
    settle_bin = str(row.get("settling_bin", "mid"))
    lines.append(
        f"  Pace   tempo={row.get('tempo','?')}  settle={settle_bin}"
        f"  →  tempo_position_fit={pace_fit:.3f}"
    )
    step_last = _get_float(row, "distance_step_last", 0.0)
    step_mean = _get_float(row, "distance_step_mean_3", 0.0)
    step_vol = _get_float(row, "distance_change_volatility_3", 0.0)
    dist_score = _get_float(row, "distance_change_score", 0.5)
    lines.append(
        f"  Dist   step_last={step_last:+.0f}m  step_mean_3={step_mean:+.0f}m  vol_3={step_vol:.1f}"
        f"  →  score={dist_score:.3f}"
    )
    drift_ol = _get_float(row, "price_drift_open_to_live", 0.0)
    drift_f1 = _get_float(row, "price_drift_fluc1_to_live", 0.0)
    move_score = _get_float(row, "market_movement_score", 0.5)
    steam_contribution = move_score * eff_weights["weight_steam"]
    lines.append(
        f"  Steam  ln(live/open)={drift_ol:+.3f}  ln(live/fluc1)={drift_f1:+.3f}"
        f"  →  score={move_score:.3f}"
        f"  ×  {eff_weights['weight_steam']:.4f}  =  {steam_contribution:.4f}"
    )

    # Form
    wr5 = _get_float(row, "recent_win_rate_5", 0.0)
    t3r5 = _get_float(row, "recent_top3_rate_5", 0.0)
    ap = _get_float(row, "recent_avg_place", 10.0)
    ap3 = _get_float(row, "recent_avg_place_3", 10.0)
    fs = _get_float(row, "recent_form_score", 0.0)
    pqs = _get_float(row, "recent_place_quality_score", 0.0)

    # Build individual run position string: e.g. "1(12) 3(10) - - - -"
    run_parts = []
    for i in range(1, 7):
        p  = row.get(f"recent_{i}_place")
        nr = row.get(f"recent_{i}_total_runners")
        try:
            p_int = int(p)
            try:
                n_int = int(nr)
                run_parts.append(f"{p_int}/{n_int}")
            except (TypeError, ValueError):
                run_parts.append(str(p_int))
        except (TypeError, ValueError):
            run_parts.append("-")
    runs_str = "  ".join(run_parts)

    lines.append(
        f"  Form   win_rate_5={wr5:.2f}  top3_rate_5={t3r5:.2f}  avg_place={ap:.1f}  avg_place_3={ap3:.1f}"
    )
    lines.append(
        f"         pos_quality={pqs:.3f}  runs[r1→r6]: {runs_str}"
        f"  →  blended={fs:.3f}  ×  {eff_weights['weight_recent_form']:.4f}"
        f"  =  {eff_weights['weight_recent_form']*fs:.4f}"
    )
    form_str = _get_float(row, "form_string_score", 0.5)
    form_wins = _get_float(row, "form_win_count_6", 0.0)
    form_t3 = _get_float(row, "form_top3_count_6", 0.0)
    form_trend = _get_float(row, "form_trend_slope", 0.0)
    form_noise = _get_float(row, "form_noise_flag", 0.0)
    lines.append(
        f"         form_str wins={form_wins:.2f} top3={form_t3:.2f} trend={form_trend:+.2f} noise={form_noise:.0f}"
        f"  →  form_string_score={form_str:.3f}"
    )
    sp_drift = _get_float(row, "today_price_vs_recent_sp_avg", 0.0)
    sp_trend = _get_float(row, "recent_sp_trend", 0.0)
    sp_score = _get_float(row, "recent_sp_score", 0.5)
    lines.append(
        f"         price_vs_hist ln(live/recent_sp)={sp_drift:+.3f}  recent_sp_trend={sp_trend:+.3f}"
        f"  →  recent_sp_score={sp_score:.3f}"
    )


    # Suitability
    ss = _get_float(row, "suitability_score", 0.0)
    dw = _get_float(row, "distance_wins", 0.0)
    ds = _get_float(row, "distance_starts", 0.0)
    tw = _get_float(row, "track_wins", 0.0)
    ts = _get_float(row, "track_starts", 0.0)
    if "heavy" in track_lower:
        cond_label = "Heavy"
        cw = _get_float(row, "heavy_wins", 0.0)
        cs = _get_float(row, "heavy_starts", 0.0)
        cond_prior = 0.0
    elif "soft" in track_lower:
        cond_label = "Soft"
        cw = _get_float(row, "soft_wins", 0.0)
        cs = _get_float(row, "soft_starts", 0.0)
        cond_prior = 0.0
    else:
        cond_label = "Good"
        cw = _get_float(row, "good_wins", 0.0)
        cs = _get_float(row, "good_starts", 0.0)
        cond_prior = 0.0
    lines.append(
        f"  Suit   dist={_fmt_record(dw, ds, prior=0.30)}"
        f"  track={_fmt_record(tw, ts, prior=0.30)}"
        f"  cond({cond_label})={_fmt_record(cw, cs, prior=cond_prior)}"
        f"  →  score={ss:.3f}  ×  {eff_weights['weight_suitability']:.4f}"
        f"  =  {eff_weights['weight_suitability']*float(ss):.4f}"
    )

    # Connection
    hjw = _get_float(row, "horse_jockey_wins", 0.0)
    hjs = _get_float(row, "horse_jockey_starts", 0.0)
    connection_score = _get_float(row, "connection_score", 0.0)
    lines.append(
        f"  Conn   jockey={_fmt_record(hjw, hjs, prior=0.15)}"
        f"  →  score={connection_score:.3f}  ×  {eff_weights['weight_connections']:.4f}"
        f"  =  {eff_weights['weight_connections']*float(connection_score):.4f}"
    )

    # Draw bias
    draw_num = row.get("draw_number", None)
    draw_starts = float(row.get("draw_bias_starts", 0) or 0)
    draw_win_rate = row.get("draw_bias_win_rate", None)
    draw_score = float(row.get("draw_bias_score", 0.25) or 0.25)
    w_draw = eff_weights.get("weight_draw_bias", 0.0)

    if draw_num is None or pd.isna(draw_num):
        draw_str = "no draw"
    elif draw_starts == 0:
        draw_str = f"draw={int(draw_num)}  no history → prior=0.25"
    else:
        rate_str = f"{draw_win_rate:.1f}%" if draw_win_rate is not None and not pd.isna(draw_win_rate) else "?"
        draw_str = f"draw={int(draw_num)}  starts={int(draw_starts)}  win_rate={rate_str}"

    lines.append(
        f"  Draw   {draw_str}"
        f"  →  score={draw_score:.3f}  ×  {w_draw:.4f}"
        f"  =  {w_draw * draw_score:.4f}"
    )
    barrier_change = _get_float(row, "barrier_change_last", 0.0)
    barrier_eff = _get_float(row, "recent_barrier_efficiency", 0.5)
    barrier_score = _get_float(row, "barrier_transition_score", 0.5)
    lines.append(
        f"         barrier_change_last={barrier_change:+.0f}  barrier_eff={barrier_eff:.3f}"
        f"  →  barrier_transition_score={barrier_score:.3f}"
    )

    # Jockey
    jockey_name = str(row.get("jockey", "") or "")
    jockey_starts = float(row.get("jockey_starts", 0) or 0)
    jockey_win_rate = row.get("jockey_win_rate", None)
    jockey_score = float(row.get("jockey_score", 0.15) or 0.15)
    w_jockey = eff_weights.get("weight_jockey", 0.0)

    if not jockey_name:
        jockey_str = "no jockey"
    elif jockey_starts == 0:
        jockey_str = f"jockey={jockey_name!r}  no history → prior=0.15"
    else:
        rate_str = f"{jockey_win_rate:.1f}%" if jockey_win_rate is not None and not pd.isna(jockey_win_rate) else "?"
        jockey_str = f"jockey={jockey_name!r}  starts={int(jockey_starts)}  win_rate={rate_str}"

    lines.append(
        f"  Jockey {jockey_str}"
        f"  →  score={jockey_score:.3f}  ×  {w_jockey:.4f}"
        f"  =  {w_jockey * jockey_score:.4f}"
    )
    same_jock = _get_float(row, "same_jockey_as_last", 0.0)
    jock_changes = _get_float(row, "jockey_change_count_6", 0.0)
    jock_cont = _get_float(row, "jockey_continuity_score", 0.5)
    lines.append(
        f"         same_as_last={same_jock:.0f}  jockey_change_count_6={jock_changes:.0f}"
        f"  →  continuity={jock_cont:.3f}"
    )

    # Trainer
    trainer_name = str(row.get("trainer", "") or "")
    trainer_starts = float(row.get("trainer_starts", 0) or 0)
    trainer_win_rate = row.get("trainer_win_rate", None)
    trainer_score = float(row.get("trainer_score", 0.15) or 0.15)
    w_trainer = eff_weights.get("weight_trainer", 0.0)

    if not trainer_name:
        trainer_str = "no trainer"
    elif trainer_starts == 0:
        trainer_str = f"trainer={trainer_name!r}  no history → prior=0.15"
    else:
        rate_str = f"{trainer_win_rate:.1f}%" if trainer_win_rate is not None and not pd.isna(trainer_win_rate) else "?"
        trainer_str = f"trainer={trainer_name!r}  starts={int(trainer_starts)}  win_rate={rate_str}"

    lines.append(
        f"  Trainer {trainer_str}"
        f"  →  score={trainer_score:.3f}  ×  {w_trainer:.4f}"
        f"  =  {w_trainer * trainer_score:.4f}"
    )
    cross_region = _get_float(row, "cross_region_flag", 0.0)
    interstate = _get_float(row, "interstate_travel_flag", 0.0)
    travel_score = _get_float(row, "travel_score", 0.5)
    lines.append(
        f"         travel cross_region={cross_region:.0f} interstate={interstate:.0f}"
        f"  →  travel_score={travel_score:.3f}"
    )

    # Market sanity
    ms = _get_float(row, "market_sanity_score", 0.0)
    raw_price = row.get("live_price")
    price_str = f"{float(raw_price):.2f}" if raw_price is not None and str(raw_price).lower() not in ("", "nan", "none") else "None"
    lines.append(
        f"  Mkt    live_price={price_str}  →  norm_inv_price={ms:.3f}"
        f"  ×  {eff_weights['weight_market_sanity']:.4f}  =  {eff_weights['weight_market_sanity']*float(ms):.4f}"
    )

    # Margin
    am3 = _get_float(row, "recent_avg_margin_3", 8.0)
    bm = _get_float(row, "recent_best_margin", 4.0)
    mgs = _get_float(row, "margin_score", 0.5)
    lines.append(
        f"  Margin avg_margin_3={am3:+.2f}  best_margin={bm:+.2f}  →  score={mgs:.3f}"
        f"  ×  {eff_weights['weight_margin']:.4f}  =  {eff_weights['weight_margin']*float(mgs):.4f}"
    )

    # Freshness
    days = _get_float(row, "recent_days_since_last_run", 21.0)
    fuw = _get_float(row, "first_up_wins", 0.0)
    fus = _get_float(row, "first_up_starts", 0.0)
    frs = _get_float(row, "freshness_score", 0.5)
    lines.append(
        f"  Fresh  days_since={days:.0f}  first_up={fuw}/{fus}  →  score={frs:.3f}"
        f"  ×  {eff_weights['weight_freshness']:.4f}  =  {eff_weights['weight_freshness']*float(frs):.4f}"
    )

    # Class
    pw = _get_float(row, "prize_money", 0.0)
    pp = _get_float(row, "place_percentage", 0.0)
    cw = _get_float(row, "career_wins", 0.0)
    cst = _get_float(row, "career_starts", 0.0)
    cls = _get_float(row, "class_score", 0.0)
    lines.append(
        f"  Class  prize=${pw:,.0f}  place_pct={pp:.1f}%  career={cw}/{cst}  →  score={cls:.3f}"
        f"  ×  {eff_weights['weight_class']:.4f}  =  {eff_weights['weight_class']*float(cls):.4f}"
    )
    class_today = _get_float(row, "class_level_today", 0.0)
    class_recent = _get_float(row, "class_level_recent_avg", 0.0)
    class_delta = _get_float(row, "class_delta_today_vs_recent", 0.0)
    class_move = _get_float(row, "class_movement_score", 0.5)
    weight_delta = _get_float(row, "weight_delta_last", 0.0)
    weight_rel = _get_float(row, "field_relative_weight_z", 0.0)
    weight_score = _get_float(row, "weight_trend_score", 0.5)
    pedigree = _get_float(row, "pedigree_score", 0.5)
    blinkers_on = _get_float(row, "blinkers_on_flag", 0.0)
    blinkers_change = _get_float(row, "blinkers_change_flag", 0.0)
    lines.append(
        f"         class_lvl today={class_today:.1f} recent={class_recent:.1f} delta={class_delta:+.1f}"
        f"  →  class_move={class_move:.3f}"
    )
    lines.append(
        f"         weight delta_last={weight_delta:+.1f}kg  rel_z={weight_rel:+.2f}"
        f"  →  weight_trend_score={weight_score:.3f}"
    )
    lines.append(
        f"         pedigree={pedigree:.3f}  blinkers_on={blinkers_on:.0f} blinkers_change={blinkers_change:.0f}"
    )

    # Model score total
    base_model_score = transparent_component_score(row, config)
    meta_model_score = _get_float(row, "meta_model_score", 0.0)
    model_score = _get_float(row, "model_score", 0.0)
    model_p = model_prob_now * 100
    raw_mkt_p = market_prob_now * 100
    fair_mkt_p = fair_market_prob_now * 100
    raw_edge = _get_float(row, "raw_edge", 0.0) * 100
    fair_edge = _get_float(row, "fair_edge", _get_float(row, "edge", 0.0)) * 100
    ev = _get_float(row, "ev", 0.0) * 100
    fp = row.get("finish_place")
    place_str = f"   Place={int(fp)}" if fp is not None and pd.notna(fp) else ""
    lines.append(f"  ─────────────────────────────────────────────────────────")
    if meta_model_score > 0:
        lines.append(
            f"  SCORE BLEND = 0.20 * base({base_model_score:.4f}) + 0.80 * meta({meta_model_score:.4f})"
            f" = {model_score:.4f}"
        )
    else:
        lines.append(f"  SCORE TOTAL = base({base_model_score:.4f}) = {model_score:.4f}")
    lines.append(
        f"  MODEL SCORE = {model_score:.4f}   ModelP={model_p:.1f}%"
        f"   RawMktP={raw_mkt_p:.1f}%   FairMktP={fair_mkt_p:.1f}%"
        f"   RawEdge={raw_edge:+.1f}%   FairEdge={fair_edge:+.1f}%   EV={ev:+.1f}%{place_str}"
    )
    return "\n".join(lines)
