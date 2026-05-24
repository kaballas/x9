# Python Implementation Plan: Betting Edge Framework for `race_reports.sqlite`

## Context gathered before planning

This plan is based on `plan.md` plus direct inspection of `/home/theo/perplex/x7/x9/database/race_reports.sqlite`.

Observed database facts:

- Base table: `race_runners`
- Database also contains many aggregate or derived views; examples inspected: `race_runners_active_races`, `race_runners_value_candidate_rows`, `race_runners_price_band_stats`
- Shape of `race_runners`:
  - `runner_rows = 45,645`
  - `races = 3,595`
  - `competitions = 169`
  - `first_start = 2026-02-14T03:55:00+00:00`
  - `last_start = 2026-05-20T08:40:00+00:00`
- Price availability:
  - `fluc2 > 0`: `41,874`
  - `fluc1 > 0`: `41,378`
  - `open_price > 0`: `42,200`
  - `sp_starting_price > 0`: `13,355`
  - total rows: `45,645`
- Result distribution:
  - `finished/L = 19,155`
  - `finished/P = 10,013`
  - `finished/W = 3,339`
  - `late_scratched/V = 11,819`
  - `no_result/- = 683`
  - `no_result/L = 630`
- Important implication: `status` is a safer live/backtest discriminator than `result_code`, because some `no_result` rows already have non-blank `result_code` values.

Planning principles used here:

1. `plan.md` is the source of truth.
2. Version 1 stays deliberately narrow.
3. The core pipeline is shared by historical backtesting and live candidate selection.
4. `sp_starting_price` is reference-only and never part of live candidate selection.
5. The base table should drive the pipeline; views are useful for inspection but are unsafe for v1 features unless explicitly proven leakage-safe.

---

## Section 1: Key decisions summary

1. **Primary betting target**: version 1 targets **win-only bets** where the model-estimated win probability exceeds the market-implied probability.
2. **Critical live price rule**: `live_price = COALESCE(fluc2, fluc1, open_price)`. This is the only price used for candidate selection and the default price used for backtest settlement.
3. **SP rule**: `sp_starting_price` is loaded only as a **reference field** for SP comparison, market movement review, and optional future analysis. It must **never** be used for live candidate selection or for computing `live_price`.
4. **Shared pipeline**: historical backtesting and live candidate selection must use the **same pipeline** from validation through staking. The only difference is the input query and the extra settlement step in backtest mode.
5. **One-row-one-runner invariant**: every module must preserve the grain of the data so that **one DataFrame row always represents exactly one runner**.
6. **Version 1 constraints**: win-only, flat `1.0` unit staking, one bet per race by default, no each-way betting, no place betting, no Kelly staking, no manual overrides.
7. **Edge rule**: only bet when `edge = model_prob - market_implied_prob` meets or exceeds the configured minimum edge threshold. V1 default is `min_edge = 0.05` (5 percentage points).
8. **Price range rule**: version 1 only considers runners with `live_price` inside the tested range, default `2.0 <= live_price <= 12.0`.
9. **Field size rule**: version 1 only considers races with declared `field_size` between `6` and `14` inclusive.
10. **Model rank rule**: version 1 limits bets to high-confidence runners, default `model_rank <= 2`.
11. **Price availability and quality rules**: runners without valid `live_price` are excluded, and race-level price coverage is checked so poorly priced races can be discarded.
12. **Backtest settlement rule**: for settled historical bets, profit is `(live_price - 1) * stake` when `is_winner = 1`, otherwise `-stake`.
13. **Query safety rule**: backtest mode reads only `status = 'finished'` races; live mode reads only `status = 'no_result'` races. This avoids cross-contamination between resolved and unresolved races.
14. **Late-scratch rule**: runners with `result_code = 'V'` are always excluded from both backtests and live candidate lists.
15. **What is deferred to v2**: probability calibration beyond passthrough, place/each-way logic, Kelly or variable staking, manual exclusions/overrides, view-based derived signals, and SP-mode selection experiments.

---

## Section 2: Module list and folder structure

Exact target structure:

```text
x9/
├── database/
│   └── race_reports.sqlite
├── betting/
│   ├── __init__.py
│   ├── config.py
│   ├── db.py
│   ├── schema.py
│   ├── validation.py
│   ├── features.py
│   ├── scoring.py
│   ├── probabilities.py
│   ├── calibration.py
│   ├── edge.py
│   ├── filters.py
│   ├── staking.py
│   ├── settlement.py
│   ├── backtest.py
│   ├── reporting.py
│   └── live_candidates.py
├── scripts/
│   ├── run_backtest.py
│   ├── inspect_race.py
│   ├── list_live_bets.py
│   └── export_backtest_report.py
├── tests/
│   ├── test_prices.py
│   ├── test_probabilities.py
│   ├── test_edge.py
│   ├── test_settlement.py
│   └── test_no_leakage.py
└── outputs/
    ├── backtests/
    ├── live/
    └── reports/
```

### Exact responsibility of each file

#### `betting/__init__.py`
Package marker for the `betting` package. Optionally exports the most-used entry points and constants so scripts can import from `betting` cleanly.

#### `betting/config.py`
Defines the single source of runtime configuration through one `CONFIG` dictionary. Holds database paths, thresholds, flags, price-column naming, staking defaults, and score weights.

#### `betting/db.py`
Owns all SQLite reads from `race_runners`. Produces canonical DataFrames with derived `live_price` and `price_quality`, and keeps `sp_starting_price` only as a reference column.

#### `betting/schema.py`
Declares required column groups for identity, price, features, and results. Provides a common schema validator that other modules call before computation.

#### `betting/validation.py`
Performs data-integrity checks before feature generation. Detects missing columns, duplicate runners, invalid prices, out-of-range race sizes, and missing result labels in backtest mode.

#### `betting/features.py`
Builds deterministic pre-race runner features while preserving one-row-one-runner granularity. Adds condition-aware ratings, suitability features, form features, market rank, and categorical bands.

#### `betting/scoring.py`
Converts engineered features into a numeric `model_score` and race-relative `model_rank`. Keeps the v1 model intentionally transparent and weighted-sum based.

#### `betting/probabilities.py`
Transforms raw `model_score` values into within-race probabilities and computes market-implied probabilities from `live_price`. This is where ranking becomes pricing logic.

#### `betting/calibration.py`
Provides a v1 passthrough calibration layer so the architecture is future-proof. In v2 this file becomes the home for probability calibration techniques.

#### `betting/edge.py`
Compares model probability to market probability and produces edge metrics. Also derives SP comparison fields strictly for reporting and diagnostics.

#### `betting/filters.py`
Applies the strategy’s gating logic: price quality, price range, field size, form sufficiency, minimum edge, model rank, price coverage, and one-bet-per-race enforcement.

#### `betting/staking.py`
Assigns stake sizes to candidate bets. Version 1 is flat staking only.

#### `betting/settlement.py`
Settles historical bets using `live_price` and result labels. Computes bet-level profit and aggregate ROI helpers.

#### `betting/backtest.py`
Runs the full backtest flow from database load through settlement. Also owns the shared `run_pipeline` function used by both historical and live flows.

#### `betting/reporting.py`
Builds summary metrics and grouped breakdowns from settled results. Prints human-readable reports and exports CSV outputs.

#### `betting/live_candidates.py`
Runs the shared pipeline on unresolved races only. Returns live candidate rows sorted by edge descending.

#### `scripts/run_backtest.py`
Primary CLI for historical backtesting. Accepts argument overrides, runs the pipeline, prints a summary, and exports outputs.

#### `scripts/inspect_race.py`
Diagnostic CLI for a single race. Loads one race, runs the shared pipeline, and prints ranking, probabilities, edges, and prices for manual inspection.

#### `scripts/list_live_bets.py`
CLI that lists current live candidates using unresolved races only. Prints a compact candidate table and can export CSV output.

#### `scripts/export_backtest_report.py`
CLI that reruns the backtest specifically for export/report generation workflows. Emits the full breakdown suite and writes files into the configured output folders.

#### `tests/test_prices.py`
Unit tests for live-price derivation, price fallback order, and price-quality classification.

#### `tests/test_probabilities.py`
Unit tests for probability math, especially softmax normalization and market-implied probability derivation.

#### `tests/test_edge.py`
Unit tests for edge computation and presentation fields like `edge_pct`.

#### `tests/test_settlement.py`
Unit tests for flat-stake settlement using `live_price`, never `sp_starting_price`.

#### `tests/test_no_leakage.py`
Safety tests ensuring finished and live queries stay separated and SP data never leaks into selection logic.

---

## Section 3: Module responsibilities and function specifications

### `betting/__init__.py`

**File purpose**: Marks `betting` as an importable package and exposes the minimal public API. The file should stay intentionally small to avoid circular imports and hidden side effects.

**Functions / objects**:

- `__all__: list[str]`
  - **Purpose**: Declares the public package surface for `CONFIG`, `run_backtest`, and `run_live_candidates` if the package chooses to expose them.
  - **Notes**: No computation should run at import time.
  - **Edge cases**: None; keep stable and explicit.

### `betting/config.py`

**File purpose**: Defines the single configuration object for the entire framework. The module should not contain dynamic logic; it should only contain declarative settings and inline comments describing each key.

**Functions / objects**:

- `CONFIG: dict[str, object]`
  - **Purpose**: Centralizes all thresholds, paths, price rules, flags, and scoring weights.
  - **Notes**: Scripts may clone and override selected keys at runtime, but this module itself stays static.
  - **Edge cases**: Values should be chosen so that importing the dict alone cannot fail.

### `betting/db.py`

**File purpose**: Encapsulates all SQLite reads and ensures the rest of the code works with a canonical runner-level DataFrame. This is the only module that should know the exact SQL used to load base rows.

#### `load_race_runners(db_path: str, query_mode: str, config: dict[str, object]) -> pd.DataFrame`
- **What it does**: Executes the canonical SQL query against `race_runners` for either backtest or live mode. It derives `live_price` using `COALESCE(fluc2, fluc1, open_price)`, derives `price_quality`, excludes late scratches (`result_code = 'V'`), and returns a DataFrame with consistent column names and sort order.
- **Key logic / algorithm notes**:
  - `query_mode` must be either `'backtest'` or `'live'`.
  - Backtest mode filters `status = 'finished'` and keeps result columns.
  - Live mode filters `status = 'no_result'` and does not require result columns to be populated.
  - `sp_starting_price` is selected but not used to derive `live_price`.
- **Edge cases**:
  - If `query_mode` is invalid, raise `ValueError`.
  - If the query returns zero rows, return an empty DataFrame with the expected columns, not `None`.
  - If some live rows have non-empty `result_code`, rely on `status`, not `result_code`, to decide mode.

#### `get_race_ids(db_path: str, query_mode: str) -> list[str]`
- **What it does**: Returns the distinct ordered `race_id` values for either backtest or live mode. This is useful for iteration, diagnostics, and CLI tools.
- **Key logic / algorithm notes**:
  - Uses the same mode predicates as `load_race_runners`.
  - Excludes `result_code = 'V'`.
  - Sorts by `start_time_iso` and then `race_id` where possible for stable CLI output.
- **Edge cases**:
  - Returns an empty list if no matching races exist.
  - Raises `ValueError` on unsupported `query_mode`.

#### `load_single_race(db_path: str, race_id: str) -> pd.DataFrame`
- **What it does**: Loads all rows for a single `race_id` from `race_runners`, including prices, result columns, and race context. It also derives `live_price` and `price_quality` so the result is immediately usable by `inspect_race.py`.
- **Key logic / algorithm notes**:
  - Does not apply finished/live filtering because the CLI may inspect either kind of race.
  - Excludes `result_code = 'V'` by default to keep the race view runner-active.
  - Sorts by `runner_number`.
- **Edge cases**:
  - If `race_id` is missing or unknown, return an empty DataFrame with canonical columns.
  - If mixed statuses exist for the same race, surface them as loaded; do not silently coerce status.

### `betting/schema.py`

**File purpose**: Provides the column contracts for all downstream modules. The module prevents accidental drift by defining clear required column lists.

#### `REQUIRED_FEATURE_COLUMNS: list[str]`
- **Proposed contents**:
  - `['race_id', 'selection_id', 'runner_name', 'competition_name', 'race_number', 'start_time_iso', 'distance_m', 'track_status', 'field_size', 'runner_number', 'draw_number', 'tempo', 'speed_rating', 'dry_rating', 'wet_rating', 'good_starts', 'good_wins', 'soft_starts', 'soft_wins', 'heavy_starts', 'heavy_wins', 'distance_starts', 'distance_wins', 'track_starts', 'track_wins', 'horse_jockey_starts', 'horse_jockey_wins', 'recent_runs_count', 'recent_avg_place', 'recent_avg_place_3', 'recent_avg_place_5', 'recent_win_rate_5', 'recent_top3_rate_5', 'open_price', 'fluc1', 'fluc2', 'live_price', 'price_quality']`
- **What it does**: Enumerates every base or derived input column needed by `features.py`.
- **Key logic / algorithm notes**: This list is intentionally limited to base-table inputs that are pre-race safe.
- **Edge cases**: Must stay aligned with feature code; missing updates here would allow silent runtime failures later.

#### `REQUIRED_RESULT_COLUMNS: list[str]`
- **Proposed contents**:
  - `['finish_place', 'result_code', 'status', 'is_winner', 'top3_mask', 'live_price']`
- **What it does**: Lists columns needed for historical settlement and reporting, such as `is_winner`, `finish_place`, `result_code`, and `status`.
- **Key logic / algorithm notes**: Used only in backtest mode validation.
- **Edge cases**: Should not be enforced in live mode.

#### `REQUIRED_PRICE_COLUMNS: list[str]`
- **Proposed contents**:
  - `['open_price', 'fluc1', 'fluc2', 'sp_starting_price', 'live_price', 'price_quality']`
- **What it does**: Lists `open_price`, `fluc1`, `fluc2`, `sp_starting_price`, and derived `live_price` when appropriate.
- **Key logic / algorithm notes**: Keeps price handling explicit and testable.
- **Edge cases**: `sp_starting_price` is required only as a reference column, not as a required non-null value.

#### `IDENTITY_COLUMNS: list[str]`
- **Proposed contents**:
  - `['race_id', 'selection_id', 'race_number', 'race_name', 'competition_id', 'competition_name', 'country', 'start_time_iso', 'runner_number', 'runner_name', 'draw_number']`
- **What it does**: Lists columns that preserve runner identity and race provenance, such as `race_id`, `selection_id`, `runner_name`, `competition_name`, and `start_time_iso`.
- **Key logic / algorithm notes**: These columns should survive every transformation.
- **Edge cases**: Missing identity columns should fail validation immediately.

#### `validate_schema(df: pd.DataFrame, required_cols: list[str]) -> None`
- **What it does**: Checks that all required columns exist on the DataFrame and raises `ValueError` if any are missing.
- **Key logic / algorithm notes**:
  - Compare `required_cols` against `df.columns`.
  - Raise a message listing all missing columns in sorted order.
  - Return `None` on success.
- **Edge cases**:
  - Empty DataFrames with correct columns should pass.
  - `required_cols = []` should be treated as a no-op.

### `betting/validation.py`

**File purpose**: Ensures the raw input is structurally valid before features or modeling are attempted. The module should be deterministic and easy to reason about.

#### `validate_input(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame`
- **What it does**: Runs the full set of validation checks in order and returns a cleaned DataFrame with invalid rows flagged or dropped according to config. It is the entry point used by the shared pipeline.
- **Key logic / algorithm notes**:
  - Call `check_required_columns` first.
  - Add validation flags for duplicate runners, missing/invalid live price, bad field size, and bad result labels.
  - Respect config flags such as `exclude_runner_if_no_live_price`.
  - Preserve one-row-one-runner structure.
- **Edge cases**:
  - If all rows are filtered out, return an empty DataFrame with the same columns plus validation flags.
  - If DataFrame is empty but schema is valid, return it unchanged except for any expected flag columns.

#### `check_required_columns(df: pd.DataFrame, config: dict[str, object]) -> None`
- **What it does**: Verifies that the DataFrame contains all identity, price, and feature columns required for the current mode.
- **Key logic / algorithm notes**:
  - Use `validate_schema` from `schema.py`.
  - Include result columns only in backtest mode.
- **Edge cases**:
  - Raise `ValueError` immediately on missing columns.

#### `check_duplicate_runners(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame`
- **What it does**: Detects duplicate `(race_id, selection_id)` pairs and marks them with a boolean flag such as `is_duplicate_runner`.
- **Key logic / algorithm notes**:
  - Use `duplicated(subset=[race_id_col, runner_id_col], keep=False)`.
  - Duplicates should normally be dropped because they violate the one-row-one-runner invariant.
- **Edge cases**:
  - If `selection_id` is null for some rows, treat those rows as invalid duplicates or invalid identity rows.
  - Empty DataFrames should return immediately.

#### `check_live_price(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame`
- **What it does**: Flags rows where `live_price` is null or less than or equal to zero. Depending on config, it can drop these rows.
- **Key logic / algorithm notes**:
  - Add boolean flag `has_valid_live_price`.
  - Never substitute `sp_starting_price` when live price is missing.
- **Edge cases**:
  - Zero or negative prices are treated as invalid.
  - Rows with only SP present remain invalid for selection.

#### `check_field_size(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame`
- **What it does**: Flags races whose declared `field_size` falls outside the configured limits.
- **Key logic / algorithm notes**:
  - Apply the rule at the race level, then map it back onto runner rows.
  - Add flag `is_valid_field_size`.
- **Edge cases**:
  - Null `field_size` should fail validation.
  - Use `field_size`, not `active_field_size`, for v1 filters.

#### `check_result_labels(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame`
- **What it does**: In backtest mode, flags rows missing settlement-critical labels like `is_winner` or `finish_place`.
- **Key logic / algorithm notes**:
  - Add flag `has_valid_result_labels`.
  - Ignore this check in live mode.
- **Edge cases**:
  - A finished row with null `is_winner` should be treated as invalid for settlement.
  - Live rows with null results should not be flagged as errors.

### `betting/features.py`

**File purpose**: Builds deterministic pre-race features directly from `race_runners` columns while avoiding leakage. Every function should return a DataFrame so the pipeline remains composable.

#### `build_features(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame`
- **What it does**: Orchestrates all feature builders and returns an enriched runner-level DataFrame. It is the only feature entry point used by the pipeline.
- **Key logic / algorithm notes**:
  - Call feature functions in a stable order.
  - Ensure no aggregation changes the row count.
  - Preserve identity and price columns.
- **Edge cases**:
  - Empty DataFrames should pass through with new columns added where practical.

#### `add_condition_rating(df: pd.DataFrame) -> pd.DataFrame`
- **What it does**: Creates `condition_rating` by selecting `wet_rating` when `track_status` indicates soft/heavy conditions, otherwise `dry_rating`.
- **Key logic / algorithm notes**:
  - Normalize `track_status` to lowercase before comparison.
  - Treat strings containing `'soft'` or `'heavy'` as wet.
- **Edge cases**:
  - If both source ratings are missing, `condition_rating` remains null.
  - If `track_status` is missing, default to `dry_rating` for v1 and document that assumption.

#### `add_suitability_score(df: pd.DataFrame) -> pd.DataFrame`
- **What it does**: Combines distance, track, and condition suitability into a single numeric feature. It converts win/start pairs into safe rates and averages them.
- **Key logic / algorithm notes**:
  - Use `distance_wins / distance_starts`, `track_wins / track_starts`, and the appropriate condition win rate.
  - Zero starts must yield `0.0`, not `NaN` or divide-by-zero.
  - Condition branch uses `good_*` for dry tracks and `soft_*` or `heavy_*` for wet tracks.
- **Edge cases**:
  - Missing starts are treated as zero.
  - Mixed or unusual track-status text should fall back to the dry branch unless it clearly contains `soft` or `heavy`.

#### `add_recent_form_score(df: pd.DataFrame) -> pd.DataFrame`
- **What it does**: Produces a compact recent-form feature from `recent_win_rate_5`, `recent_top3_rate_5`, `recent_avg_place`, and `recent_avg_place_3`.
- **Key logic / algorithm notes**:
  - Reward higher win/top-3 rates.
  - Reward lower average placing values by inverting or rescaling them.
  - Keep the formula simple and monotonic in v1.
- **Edge cases**:
  - Missing recent fields should degrade gracefully to partial information.
  - If all recent metrics are missing, return `0.0`.

#### `add_connection_score(df: pd.DataFrame) -> pd.DataFrame`
- **What it does**: Builds a connection-based feature from `horse_jockey_wins / horse_jockey_starts`.
- **Key logic / algorithm notes**:
  - Safe-divide wins by starts.
  - Keep the feature narrow in v1; do not reach into trainer/jockey views yet.
- **Edge cases**:
  - Zero starts produce `0.0`.
  - Missing wins or starts are treated as zero.

#### `add_market_rank(df: pd.DataFrame) -> pd.DataFrame`
- **What it does**: Adds `market_rank` by ranking runners within each race by `live_price` ascending. Lowest price is favorite and receives rank `1`.
- **Key logic / algorithm notes**:
  - Use groupby on `race_id`.
  - Ties should use a stable dense ranking method so equal prices share the same rank.
- **Edge cases**:
  - Null `live_price` should get null `market_rank`.
  - Single-runner races should still produce rank `1` if they survive validation.

#### `add_price_band(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame`
- **What it does**: Buckets `live_price` into fixed bands: `<3`, `3-6`, `6-9`, `9-12`, `12-20`, `>20`.
- **Key logic / algorithm notes**:
  - Use inclusive lower bounds and exclusive upper bounds except the final bucket.
  - Store as categorical text for reporting.
- **Edge cases**:
  - Null or invalid prices map to `'NO_PRICE'` or null, depending on preferred reporting convention.

#### `add_distance_band(df: pd.DataFrame) -> pd.DataFrame`
- **What it does**: Buckets races into distance bands such as `sprint`, `middle`, and `staying`.
- **Key logic / algorithm notes**:
  - Suggested thresholds: sprint `< 1400`, middle `1400-1999`, staying `>= 2000`.
  - Use the same thresholds consistently in reporting.
- **Edge cases**:
  - Null `distance_m` yields null or `'UNKNOWN_DISTANCE'`.

#### `add_form_recency_flag(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame`
- **What it does**: Flags runners whose `recent_runs_count` is below `config['min_recent_form_count']`.
- **Key logic / algorithm notes**:
  - Add boolean `has_sparse_recent_form`.
  - Downstream filters may drop these rows.
- **Edge cases**:
  - Missing `recent_runs_count` is treated as sparse form.

### `betting/scoring.py`

**File purpose**: Converts engineered features into model scores and ranks. The module should stay transparent enough that every score component can be explained.

#### `score_runners(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame`
- **What it does**: Applies the weighted scoring formula to each runner and then adds within-race `model_rank`.
- **Key logic / algorithm notes**:
  - Call `compute_model_score` first.
  - Then call `add_model_rank`.
  - Preserve all prior feature columns for debugging and reporting.
- **Edge cases**:
  - Empty DataFrames pass through unchanged except for added empty columns.

#### `compute_model_score(df: pd.DataFrame, config: dict[str, object]) -> pd.Series`
- **What it does**: Produces the v1 weighted sum of component scores.
- **Key logic / algorithm notes**:
  - Components and starting weights:
    - speed/condition rating: `0.35`
    - recent form score: `0.30`
    - suitability score: `0.20`
    - connection score: `0.10`
    - market sanity: `0.05`
  - `market_sanity` should be a simple stabilizer, not a hidden market-copying feature; for example, an inverse-price or market-rank-based normalization.
  - Weights are starting assumptions, not calibrated truths.
- **Edge cases**:
  - Missing component columns should have been caught by validation.
  - Individual missing component values should default to `0.0` in the weighted sum.

#### `add_model_rank(df: pd.DataFrame) -> pd.DataFrame`
- **What it does**: Ranks runners within each race by `model_score` descending. Highest score receives rank `1`.
- **Key logic / algorithm notes**:
  - Use dense ranking within `race_id`.
  - Stable sort should preserve deterministic ordering for tied scores.
- **Edge cases**:
  - Null `model_score` rows should fall to the bottom or remain null-ranked, depending on implementation preference; v1 should prefer null rank for clarity.

### `betting/probabilities.py`

**File purpose**: Converts ranking scores into within-race probabilities and market probabilities. This module is responsible for making edges mathematically comparable.

#### `assign_probabilities(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame`
- **What it does**: Adds `raw_model_prob` and `market_implied_prob` to the DataFrame. It performs softmax within each race and computes reciprocal market probabilities from `live_price`.
- **Key logic / algorithm notes**:
  - Group by `race_id` when applying softmax.
  - `market_implied_prob = 1 / live_price`.
  - No overround normalization is applied in v1.
- **Edge cases**:
  - Null or invalid `live_price` should yield market probability `0.0`.
  - Empty groups should not occur after grouping, but the code should still be safe.

#### `softmax_within_race(scores: pd.Series) -> pd.Series`
- **What it does**: Converts a Series of model scores for one race into probabilities summing to `1.0`.
- **Key logic / algorithm notes**:
  - Use numerically stable softmax by subtracting `scores.max()` before exponentiation.
  - Return probabilities in the same index order as input.
- **Edge cases**:
  - If all scores are null, return equal or zero probabilities; v1 should document the chosen fallback, with equal weights being the cleanest assumption.
  - If all scores are identical, softmax yields equal probabilities.

#### `compute_market_implied_prob(live_price: pd.Series) -> pd.Series`
- **What it does**: Computes reciprocal market probability from `live_price`.
- **Key logic / algorithm notes**:
  - Formula is `1 / live_price`.
  - Invalid prices return `0.0` rather than raising.
- **Edge cases**:
  - Null, zero, or negative prices all map to `0.0`.

### `betting/calibration.py`

**File purpose**: Maintains an explicit calibration stage even though v1 does not calibrate. This avoids rewiring the pipeline later.

#### `calibrate_probabilities(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame`
- **What it does**: Adds `model_prob` to the DataFrame. In v1 it simply copies `raw_model_prob` through unchanged.
- **Key logic / algorithm notes**:
  - Call `passthrough_calibration` in v1.
  - Document the v2 hook for isotonic regression or Platt scaling.
- **Edge cases**:
  - Empty DataFrames should pass through without error.

#### `passthrough_calibration(df: pd.DataFrame) -> pd.DataFrame`
- **What it does**: Assigns `model_prob = raw_model_prob` and returns the DataFrame.
- **Key logic / algorithm notes**:
  - No fitting, no buckets, no learned calibration.
- **Edge cases**:
  - If `raw_model_prob` is missing, validation should already have failed upstream.

### `betting/edge.py`

**File purpose**: Generates the core value metric used for betting decisions. This module makes the model-versus-market comparison explicit and auditable.

#### `calculate_edges(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame`
- **What it does**: Adds `edge`, `edge_pct`, and SP comparison fields to the DataFrame. It should also derive a reference field such as `price_vs_sp` or `price_drift` from `live_price` and `sp_starting_price` purely for reporting.
- **Key logic / algorithm notes**:
  - `edge = model_prob - market_implied_prob`
  - `edge_pct = edge * 100`
  - SP comparison must never feed back into candidate filters or score calculations.
- **Edge cases**:
  - If `sp_starting_price` is null, SP comparison fields should remain null.
  - Negative edge values are valid and should be preserved until filters run.

#### `compute_edge(model_prob: float, market_implied_prob: float) -> float`
- **What it does**: Returns the scalar difference between model probability and market-implied probability.
- **Key logic / algorithm notes**:
  - Simple subtraction with no clipping.
- **Edge cases**:
  - Null inputs should be handled before calling or converted to `0.0` explicitly.

### `betting/filters.py`

**File purpose**: Applies the strategy rules that decide whether a runner becomes a bet candidate. This module should make every exclusion reason inspectable.

#### `apply_filters(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame`
- **What it does**: Orchestrates all filters and logs how many rows or races are removed at each stage. It returns the final candidate DataFrame.
- **Key logic / algorithm notes**:
  - Apply filters in a deterministic order.
  - Keep row-level or race-level drop counts for debugging.
  - Preserve the same column schema.
- **Edge cases**:
  - Empty input should return empty output with no errors.

#### `filter_no_live_price(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame`
- **What it does**: Removes rows where `live_price` is null or invalid.
- **Key logic / algorithm notes**:
  - Respect `exclude_runner_if_no_live_price`.
  - Never substitute SP.
- **Edge cases**:
  - If the flag is false, retain rows but they will almost certainly fail later filters.

#### `filter_price_range(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame`
- **What it does**: Keeps only runners whose `live_price` is between `min_price` and `max_price` inclusive.
- **Key logic / algorithm notes**:
  - Use config thresholds directly.
- **Edge cases**:
  - Null prices should already be gone, but if present they fail the filter.

#### `filter_field_size(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame`
- **What it does**: Removes rows from races whose `field_size` falls outside the configured limits.
- **Key logic / algorithm notes**:
  - This is a race-level exclusion applied back to runners.
- **Edge cases**:
  - Missing `field_size` means exclusion.

#### `filter_min_edge(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame`
- **What it does**: Keeps only runners whose `edge` meets or exceeds `min_edge`.
- **Key logic / algorithm notes**:
  - Use raw decimal edge, not percent string.
- **Edge cases**:
  - Null edges fail the filter.

#### `filter_model_rank(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame`
- **What it does**: Keeps only runners whose `model_rank` is less than or equal to `max_model_rank`.
- **Key logic / algorithm notes**:
  - This narrows the strategy to top-ranked model runners.
- **Edge cases**:
  - Null `model_rank` fails.

#### `filter_one_per_race(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame`
- **What it does**: If `allow_multiple_bets_per_race` is `False`, keeps only the highest-edge runner in each race.
- **Key logic / algorithm notes**:
  - Sort by `edge` descending and then by `model_rank` ascending as a tie-breaker.
  - Return one row per `race_id`.
- **Edge cases**:
  - Exact ties should break deterministically using model rank, then market rank, then runner number.

#### `filter_sparse_form(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame`
- **What it does**: Removes runners flagged by `add_form_recency_flag`.
- **Key logic / algorithm notes**:
  - Uses `has_sparse_recent_form`.
- **Edge cases**:
  - Missing flag column should be considered a programming error, not silently ignored.

#### `filter_price_coverage(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame`
- **What it does**: Drops races where the fraction of runners with valid `live_price` is below `exclude_race_if_price_coverage_below`.
- **Key logic / algorithm notes**:
  - Compute coverage by `race_id` before dropping null-price runners if possible.
  - Coverage formula is `priced_runners / total_runners`.
- **Edge cases**:
  - Empty races cannot occur but should safely result in exclusion.
  - Coverage exactly equal to threshold should pass.

### `betting/staking.py`

**File purpose**: Assigns stake sizes after all selection filters have run. V1 deliberately keeps this simple so the backtest measures selection quality, not stake-sizing tricks.

#### `assign_stakes(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame`
- **What it does**: Adds a `stake` column to every candidate row. In v1, this delegates to `flat_stake`.
- **Key logic / algorithm notes**:
  - Keep the function as the stable orchestration hook for future staking modes.
- **Edge cases**:
  - Empty DataFrames should still return with a `stake` column.

#### `flat_stake(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame`
- **What it does**: Sets `stake = config['stake']` for all rows.
- **Key logic / algorithm notes**:
  - No bankroll logic, Kelly scaling, or edge-based stake multipliers.
- **Edge cases**:
  - If `stake <= 0`, validation of config should fail early in scripts; do not silently continue.

### `betting/settlement.py`

**File purpose**: Settles historical bets at the configured settlement price and computes profit metrics. This module should never be called from the live candidate flow.

#### `settle_bets(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame`
- **What it does**: Adds `profit` and `profit_net` columns for each settled bet row using `live_price` and `stake`.
- **Key logic / algorithm notes**:
  - Winner: `(live_price - 1.0) * stake`
  - Loser: `-1.0 * stake`
  - `profit_net` may equal `profit` in v1 unless a gross/net distinction is later introduced.
- **Edge cases**:
  - Missing `is_winner` is a hard failure in backtest mode.
  - `sp_starting_price` must never be consulted.

#### `compute_profit(row: pd.Series) -> float`
- **What it does**: Computes the scalar profit for a single settled bet row.
- **Key logic / algorithm notes**:
  - Reads `row[settlement_price_column]` and `row['stake']`.
  - Uses `is_winner` as the settlement truth.
- **Edge cases**:
  - Invalid settlement price should raise or be blocked by upstream validation.

#### `compute_roi(profit_series: pd.Series, stake_series: pd.Series) -> float`
- **What it does**: Computes aggregate ROI as `sum(profit) / sum(stake)`.
- **Key logic / algorithm notes**:
  - Return decimal ROI, not percent.
- **Edge cases**:
  - If total stake is zero, return `0.0` to avoid divide-by-zero.

### `betting/backtest.py`

**File purpose**: Runs the historical version of the framework and hosts the shared core pipeline. This is the module that most clearly enforces the “same logic for backtest and live” rule.

#### `run_backtest(config: dict[str, object]) -> pd.DataFrame`
- **What it does**: Loads resulted races from the database, runs the shared pipeline, settles bets, and returns the settled result DataFrame.
- **Key logic / algorithm notes**:
  - Call `db.load_race_runners(..., query_mode='backtest', ...)`.
  - Then call `run_pipeline`.
  - Then call `settle_bets`.
- **Edge cases**:
  - If no qualifying bets remain after filters, return an empty settled DataFrame that reporting can still consume.

#### `run_pipeline(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame`
- **What it does**: Implements the shared pipeline used by both backtest and live candidate flows.
- **Key logic / algorithm notes**:
  - Call, in order: `validate_input` → `build_features` → `score_runners` → `assign_probabilities` → `calibrate_probabilities` → `calculate_edges` → `apply_filters` → `assign_stakes`.
  - Must not perform settlement.
- **Edge cases**:
  - Empty input should remain safe all the way through.

### `betting/live_candidates.py`

**File purpose**: Applies the exact same pipeline to unresolved races and returns only the candidate rows. This module exists so live selection cannot drift away from historical testing logic.

#### `run_live_candidates(config: dict[str, object]) -> pd.DataFrame`
- **What it does**: Loads unresolved races, runs the shared pipeline, sorts candidates by `edge` descending, and returns the candidate DataFrame.
- **Key logic / algorithm notes**:
  - Uses `query_mode='live'`.
  - Never calls settlement.
  - Does not use `sp_starting_price` in any selection step.
- **Edge cases**:
  - If no candidates survive filtering, return an empty DataFrame with the candidate schema.

### `betting/reporting.py`

**File purpose**: Turns settled backtest results into summaries, grouped breakdowns, console output, and CSV exports. Reporting should stay downstream-only and never influence selection logic.

#### `build_report(result_df: pd.DataFrame, config: dict[str, object]) -> dict[str, object]`
- **What it does**: Builds a dictionary containing overall summary metrics plus all standard grouped breakdowns.
- **Key logic / algorithm notes**:
  - Call `summary_metrics` once.
  - Call all breakdown helpers and store them under predictable keys.
- **Edge cases**:
  - Empty `result_df` should produce zero-valued metrics and empty breakdown tables.

#### `summary_metrics(result_df: pd.DataFrame) -> dict[str, float]`
- **What it does**: Computes headline metrics: total bets, total staked, total profit, ROI, strike rate, average price, and average edge.
- **Key logic / algorithm notes**:
  - `strike_rate = wins / total_bets`
  - `avg_price` from `live_price`
  - `avg_edge` from `edge`
- **Edge cases**:
  - Zero bets should yield zeros, not NaN or exceptions.

#### `breakdown_by(result_df: pd.DataFrame, group_col: str) -> pd.DataFrame`
- **What it does**: Produces a grouped table with bets, wins, total staked, total profit, ROI, strike rate, average price, and average edge for a single grouping column.
- **Key logic / algorithm notes**:
  - Standardize the returned column order so all specialized wrappers are consistent.
- **Edge cases**:
  - Missing `group_col` should raise `ValueError`.

#### `breakdown_by_price_band(result_df: pd.DataFrame) -> pd.DataFrame`
- **What it does**: Returns grouped metrics by `price_band`.
- **Key logic / algorithm notes**: Thin wrapper over `breakdown_by`.
- **Edge cases**: Empty results return empty grouped DataFrame.

#### `breakdown_by_track(result_df: pd.DataFrame) -> pd.DataFrame`
- **What it does**: Returns grouped metrics by `competition_name` or track identifier.
- **Key logic / algorithm notes**: Use the same track field consistently across CLI output.
- **Edge cases**: Missing track values should be grouped under `UNKNOWN_TRACK` if needed.

#### `breakdown_by_distance_band(result_df: pd.DataFrame) -> pd.DataFrame`
- **What it does**: Returns grouped metrics by `distance_band`.
- **Key logic / algorithm notes**: Wrapper over `breakdown_by`.
- **Edge cases**: Null band values may be grouped separately.

#### `breakdown_by_condition(result_df: pd.DataFrame) -> pd.DataFrame`
- **What it does**: Returns grouped metrics by `track_status`.
- **Key logic / algorithm notes**: Keep raw `track_status` values rather than over-normalizing in reporting.
- **Edge cases**: Null values grouped separately if present.

#### `breakdown_by_field_size(result_df: pd.DataFrame) -> pd.DataFrame`
- **What it does**: Returns grouped metrics by `field_size` or a derived field-size band.
- **Key logic / algorithm notes**: V1 may group by exact `field_size`; if a band is desired, define it once and use consistently.
- **Edge cases**: Missing field size values grouped separately or excluded.

#### `breakdown_by_market_rank(result_df: pd.DataFrame) -> pd.DataFrame`
- **What it does**: Returns grouped metrics by `market_rank`.
- **Key logic / algorithm notes**: Useful to measure favorite versus non-favorite performance.
- **Edge cases**: Null ranks grouped separately.

#### `breakdown_by_model_rank(result_df: pd.DataFrame) -> pd.DataFrame`
- **What it does**: Returns grouped metrics by `model_rank`.
- **Key logic / algorithm notes**: Measures whether rank 1 or rank 2 candidates are driving profitability.
- **Edge cases**: Null ranks grouped separately.

#### `breakdown_by_price_quality(result_df: pd.DataFrame) -> pd.DataFrame`
- **What it does**: Returns grouped metrics by `price_quality`.
- **Key logic / algorithm notes**: This validates whether `FLUC2`, `FLUC1`, or `OPEN_ONLY` rows behave differently.
- **Edge cases**: `NO_PRICE` rows should normally never appear after filters.

#### `print_report(report_dict: dict[str, object]) -> None`
- **What it does**: Prints the summary metrics and standard breakdown sections in a predictable CLI format.
- **Key logic / algorithm notes**:
  - Print overall summary first.
  - Then print grouped tables in the required order.
- **Edge cases**:
  - Empty breakdown tables should still print section headers and a no-data message.

#### `export_csv(result_df: pd.DataFrame, output_path: str) -> None`
- **What it does**: Writes a DataFrame to CSV at the specified path.
- **Key logic / algorithm notes**:
  - Ensure parent directories exist.
  - Use `index=False`.
- **Edge cases**:
  - Empty DataFrames should still export with headers.

### `scripts/run_backtest.py`

**File purpose**: CLI entry point for the standard historical backtest. It should be the easiest way to reproduce the full v1 workflow.

#### `parse_args() -> argparse.Namespace`
- **What it does**: Parses CLI arguments for database path, output directory, and key threshold overrides.
- **Key logic / algorithm notes**:
  - Required overrides: `--db`, `--output-dir`, `--min-edge`, `--min-price`, `--max-price`.
  - All overrides should be optional and fall back to `CONFIG` defaults.
- **Edge cases**:
  - Invalid numeric inputs should trigger argparse errors.

#### `build_runtime_config(args: argparse.Namespace) -> dict[str, object]`
- **What it does**: Creates a runtime config dict by copying `CONFIG` and applying CLI overrides.
- **Key logic / algorithm notes**:
  - Never mutate the imported module-level `CONFIG` in place.
- **Edge cases**:
  - Missing override values leave defaults unchanged.

#### `main() -> None`
- **What it does**: Orchestrates the CLI flow: load config, run backtest, build report, print report, and export outputs.
- **Key logic / algorithm notes**:
  - Output files should go to `outputs/backtests/` by default.
  - Timestamp filenames for reproducibility.
- **Edge cases**:
  - No-bet backtests should still print zeros and export empty CSVs.

### `scripts/inspect_race.py`

**File purpose**: CLI for race-level diagnostics. It is a debugging tool, not a betting or reporting entry point.

#### `parse_args() -> argparse.Namespace`
- **What it does**: Parses `race_id` plus optional `--db` and override flags.
- **Key logic / algorithm notes**: `race_id` should be required.
- **Edge cases**: Missing `race_id` should fail at argument parsing.

#### `build_runtime_config(args: argparse.Namespace) -> dict[str, object]`
- **What it does**: Clones and overrides `CONFIG` for the inspect workflow.
- **Key logic / algorithm notes**: Same pattern as other scripts.
- **Edge cases**: Same as other scripts.

#### `main() -> None`
- **What it does**: Loads one race using `load_single_race`, runs `run_pipeline`, and prints ranked runners with prices, scores, probabilities, and edges.
- **Key logic / algorithm notes**:
  - Should show all runners in the race, not only final candidates, when used for debugging.
  - Candidate rows can be highlighted separately.
- **Edge cases**:
  - Unknown race ID should print a no-data message and exit cleanly.

### `scripts/list_live_bets.py`

**File purpose**: CLI for generating the current live-candidate slate. It is the operational entry point for unresolved races.

#### `parse_args() -> argparse.Namespace`
- **What it does**: Parses `--db`, optional output path, and threshold overrides.
- **Key logic / algorithm notes**: Keep the interface aligned with `run_backtest.py` where possible.
- **Edge cases**: Invalid overrides should fail at parse time.

#### `build_runtime_config(args: argparse.Namespace) -> dict[str, object]`
- **What it does**: Creates the runtime config for the live flow.
- **Key logic / algorithm notes**: Copy from `CONFIG`, then override.
- **Edge cases**: Same as above.

#### `main() -> None`
- **What it does**: Runs `run_live_candidates`, sorts by edge descending, prints the candidate table, and optionally exports CSV.
- **Key logic / algorithm notes**:
  - Required printed columns: `race_id`, `race_number`, `competition_name`, `runner_name`, `live_price`, `price_quality`, `model_score`, `model_rank`, `model_prob`, `market_implied_prob`, `edge`.
- **Edge cases**:
  - If no candidates survive, print an empty-state message rather than failing.

### `scripts/export_backtest_report.py`

**File purpose**: CLI specifically for full export/report workflows. It wraps the same backtest logic but focuses on writing files.

#### `parse_args() -> argparse.Namespace`
- **What it does**: Parses export-oriented options such as `--db`, `--output-dir`, and threshold overrides.
- **Key logic / algorithm notes**: Same basic arguments as `run_backtest.py`.
- **Edge cases**: Same as other scripts.

#### `build_runtime_config(args: argparse.Namespace) -> dict[str, object]`
- **What it does**: Creates the runtime config for this export workflow.
- **Key logic / algorithm notes**: Avoid in-place mutation of imported config.
- **Edge cases**: Same as other scripts.

#### `main() -> None`
- **What it does**: Runs the backtest, builds the report, exports the full bet-level CSV plus grouped breakdown CSVs, and prints the report summary.
- **Key logic / algorithm notes**:
  - Should share as much code as possible with `run_backtest.py`.
- **Edge cases**:
  - Empty backtest results still export correctly.

### `tests/test_prices.py`

**File purpose**: Verifies price derivation and classification rules. These tests protect the most critical source-of-truth rule in the plan.

#### `sample_price_rows() -> pd.DataFrame`
- **What it does**: Fixture returning minimal runner rows with combinations of `open_price`, `fluc1`, `fluc2`, and `sp_starting_price`.
- **Key logic / algorithm notes**: Include both valid and missing price cases.
- **Edge cases**: Include rows where only SP is populated to prove SP is ignored.

#### `test_live_price_prefers_fluc2() -> None`
- **What it does**: Asserts that `live_price` uses `fluc2` when all three live inputs exist.
- **Key logic / algorithm notes**: Compare expected value directly.
- **Edge cases**: None beyond standard fallback order.

#### `test_live_price_falls_back_to_fluc1_when_fluc2_null() -> None`
- **What it does**: Asserts that `fluc1` becomes `live_price` when `fluc2` is null.
- **Edge cases**: `open_price` may still be present but should not override `fluc1`.

#### `test_live_price_falls_back_to_open_price() -> None`
- **What it does**: Asserts that `open_price` becomes `live_price` when both `fluc2` and `fluc1` are null.
- **Edge cases**: None.

#### `test_live_price_is_null_when_all_prices_null() -> None`
- **What it does**: Asserts that `live_price` remains null when all live price fields are null.
- **Edge cases**: SP may be present and must still be ignored.

#### `test_sp_not_used_in_live_price() -> None`
- **What it does**: Asserts that `sp_starting_price` never affects `live_price`.
- **Edge cases**: SP-only row should still yield null `live_price`.

#### `test_price_quality_classification() -> None`
- **What it does**: Asserts that `price_quality` maps correctly to `FLUC2`, `FLUC1`, `OPEN_ONLY`, and `NO_PRICE`.
- **Edge cases**: Zero or invalid prices should not be misclassified as available.

### `tests/test_probabilities.py`

**File purpose**: Verifies probability normalization logic. These tests ensure scores become mathematically coherent race-level probabilities.

#### `sample_score_race() -> pd.DataFrame`
- **What it does**: Fixture returning a minimal single-race DataFrame with multiple runners and known model scores/prices.
- **Edge cases**: Include one row with null `live_price` if needed.

#### `test_softmax_sums_to_one_within_race() -> None`
- **What it does**: Asserts softmax probabilities sum to `1.0` within a race.

#### `test_softmax_highest_score_gets_highest_prob() -> None`
- **What it does**: Asserts the highest `model_score` yields the highest `raw_model_prob`.

#### `test_market_implied_prob_is_reciprocal_of_price() -> None`
- **What it does**: Asserts `1 / live_price` calculation for valid prices.

#### `test_market_implied_prob_zero_for_null_price() -> None`
- **What it does**: Asserts null or invalid prices map to `0.0` market probability.

### `tests/test_edge.py`

**File purpose**: Verifies edge calculations. These tests protect the final selection metric.

#### `sample_probability_rows() -> pd.DataFrame`
- **What it does**: Fixture returning minimal rows with `model_prob`, `market_implied_prob`, `live_price`, and `sp_starting_price`.

#### `test_positive_edge_when_model_prob_exceeds_market() -> None`
- **What it does**: Asserts edge is positive when model probability is greater than market probability.

#### `test_negative_edge_when_market_prob_exceeds_model() -> None`
- **What it does**: Asserts edge is negative when market-implied probability is greater.

#### `test_edge_is_zero_when_equal() -> None`
- **What it does**: Asserts exact equality yields zero edge.

#### `test_edge_pct_is_100x_edge() -> None`
- **What it does**: Asserts `edge_pct` equals `edge * 100`.

### `tests/test_settlement.py`

**File purpose**: Verifies historical bet settlement under flat staking. These tests ensure the backtest settles at `live_price`, not SP.

#### `sample_settlement_rows() -> pd.DataFrame`
- **What it does**: Fixture returning winner and loser rows with `live_price`, `sp_starting_price`, `is_winner`, and `stake`.

#### `test_winner_profit_is_price_minus_one() -> None`
- **What it does**: Asserts winner profit is `(live_price - 1) * stake`.

#### `test_loser_profit_is_minus_one() -> None`
- **What it does**: Asserts loser profit is `-stake`.

#### `test_roi_calculation() -> None`
- **What it does**: Asserts `compute_roi` returns `sum(profit) / sum(stake)`.

#### `test_profit_uses_live_price_not_sp() -> None`
- **What it does**: Asserts a differing `sp_starting_price` does not affect profit.

### `tests/test_no_leakage.py`

**File purpose**: Verifies that historical and live modes stay separated and that SP remains reference-only. These are policy tests as much as math tests.

#### `sample_query_frames() -> tuple[pd.DataFrame, pd.DataFrame]`
- **What it does**: Fixture returning one mock finished DataFrame and one mock live DataFrame for pipeline-level safety tests.

#### `test_backtest_query_excludes_no_result_races() -> None`
- **What it does**: Asserts the backtest loader predicate excludes `status = 'no_result'` rows.

#### `test_live_query_excludes_finished_races() -> None`
- **What it does**: Asserts the live loader predicate excludes `status = 'finished'` rows.

#### `test_settlement_does_not_use_sp_starting_price() -> None`
- **What it does**: Asserts settlement logic references `live_price` only.

#### `test_pipeline_does_not_use_sp_for_edge_calculation() -> None`
- **What it does**: Asserts `edge` derives from `model_prob` and `live_price`, not SP.

#### `test_sp_column_exists_but_is_reference_only() -> None`
- **What it does**: Asserts `sp_starting_price` is present in loaded data yet has no impact on candidate selection fields.

---

## Section 4: Required database columns

Only the base table `race_runners` should feed the v1 pipeline. Views are inspection-only unless later proven leakage-safe.

### Identity columns (must always be present)

| Column | Type | Used in module(s) | Live or result-only |
|---|---|---|---|
| `race_id` | integer | `db.py`, `schema.py`, `validation.py`, `features.py`, `scoring.py`, `probabilities.py`, `filters.py`, `reporting.py` | both |
| `selection_id` | integer | `db.py`, `schema.py`, `validation.py`, `reporting.py` | both |
| `race_number` | integer | `db.py`, `reporting.py`, CLI scripts | both |
| `race_name` | text | `db.py`, CLI scripts, exports | both |
| `runner_number` | integer | `db.py`, tie-breaking in `filters.py`, CLI scripts | both |
| `runner_name` | text | all CLI/report/export layers | both |
| `draw_number` | integer | `db.py`, `features.py` | both |

### Race context columns

| Column | Type | Used in module(s) | Live or result-only |
|---|---|---|---|
| `competition_id` | integer | `db.py`, exports | both |
| `competition_name` | text | `db.py`, `reporting.py`, CLI scripts | both |
| `country` | text | `db.py`, exports/reporting | both |
| `class_name` | text | `db.py`, exports/reporting | both |
| `grade` | text | `db.py`, exports/reporting | both |
| `tempo` | text | `db.py`, `features.py`, exports | both |
| `distance_m` | integer | `features.py`, `reporting.py`, CLI scripts | both |
| `track_status` | text | `features.py`, `reporting.py` | both |
| `start_time_iso` | text | `db.py`, CLI scripts, exports | both |
| `field_size` | integer | `validation.py`, `filters.py`, `reporting.py` | both |
| `active_field_size` | integer | `db.py`, exports/diagnostics only | both |
| `status` | text | `db.py`, `validation.py`, `tests/test_no_leakage.py` | both |

### Runner quality columns

| Column | Type | Used in module(s) | Live or result-only |
|---|---|---|---|
| `jockey` | text | `db.py`, `features.py`, exports | both |
| `trainer` | text | `db.py`, `features.py`, exports | both |
| `trainer_location` | text | `db.py`, exports/reporting | both |
| `weight_kg` | real | `db.py`, exports | both |
| `age` | integer | `db.py`, exports | both |
| `sex` | text | `db.py`, exports | both |
| `speed_rating` | real | `features.py`, `scoring.py` | both |
| `dry_rating` | real | `features.py`, `scoring.py` | both |
| `wet_rating` | real | `features.py`, `scoring.py` | both |
| `win_percentage` | real | `db.py`, optional exports | both |
| `place_percentage` | real | `db.py`, optional exports | both |
| `prize_money` | real | `db.py`, optional exports | both |
| `expected_settling_position` | text | `features.py`, exports | both |
| `last_six` | text | `db.py`, exports | both |
| `form_fig` | text | `db.py`, exports | both |

### Form columns

| Column | Type | Used in module(s) | Live or result-only |
|---|---|---|---|
| `recent_runs_count` | integer | `features.py`, `filters.py` | both |
| `recent_wins` | integer | `db.py`, exports | both |
| `recent_places` | integer | `db.py`, exports | both |
| `recent_avg_place` | real | `features.py` | both |
| `recent_best_place` | integer | `db.py`, exports | both |
| `recent_avg_place_3` | real | `features.py` | both |
| `recent_avg_place_5` | real | `features.py` | both |
| `recent_win_rate_5` | real | `features.py` | both |
| `recent_top3_rate_5` | real | `features.py` | both |
| `recent_avg_margin` | real | `db.py`, exports | both |
| `recent_best_margin` | real | `db.py`, exports | both |
| `recent_avg_margin_3` | real | `db.py`, exports | both |
| `recent_avg_starting_price` | real | `db.py`, exports | both |
| `recent_same_distance_runs` | integer | `db.py`, exports | both |
| `recent_same_track_runs` | integer | `db.py`, exports | both |
| `recent_same_condition_runs` | integer | `db.py`, exports | both |
| `recent_days_since_last_run` | integer | `db.py`, exports | both |

### Suitability columns

| Column | Type | Used in module(s) | Live or result-only |
|---|---|---|---|
| `career_starts` | integer | `db.py`, exports | both |
| `career_wins` | integer | `db.py`, exports | both |
| `career_seconds` | integer | `db.py`, exports | both |
| `career_thirds` | integer | `db.py`, exports | both |
| `good_starts` | integer | `features.py` | both |
| `good_wins` | integer | `features.py` | both |
| `soft_starts` | integer | `features.py` | both |
| `soft_wins` | integer | `features.py` | both |
| `heavy_starts` | integer | `features.py` | both |
| `heavy_wins` | integer | `features.py` | both |
| `distance_starts` | integer | `features.py` | both |
| `distance_wins` | integer | `features.py` | both |
| `track_starts` | integer | `features.py` | both |
| `track_wins` | integer | `features.py` | both |
| `first_up_starts` | integer | `db.py`, exports | both |
| `first_up_wins` | integer | `db.py`, exports | both |
| `second_up_starts` | integer | `db.py`, exports | both |
| `second_up_wins` | integer | `db.py`, exports | both |

### Connections columns

| Column | Type | Used in module(s) | Live or result-only |
|---|---|---|---|
| `horse_jockey_starts` | integer | `features.py` | both |
| `horse_jockey_wins` | integer | `features.py` | both |

### Price columns

| Column | Type | Used in module(s) | Live or result-only |
|---|---|---|---|
| `open_price` | real | `db.py`, `schema.py`, CLI scripts, exports | live input |
| `fluc1` | real | `db.py`, `schema.py`, CLI scripts, exports | live input |
| `fluc2` | real | `db.py`, `schema.py`, CLI scripts, exports | live input |
| `live_price` (derived) | real | `db.py`, `validation.py`, `features.py`, `probabilities.py`, `filters.py`, `settlement.py`, `reporting.py` | live input + backtest settlement |
| `price_quality` (derived) | text | `db.py`, `filters.py`, `reporting.py`, CLI scripts | diagnostic/live quality |
| `sp_starting_price` | real | `db.py`, `schema.py`, `edge.py`, `reporting.py`, tests | reference/result-only |

### Result columns (backtest-only)

| Column | Type | Used in module(s) | Live or result-only |
|---|---|---|---|
| `finish_place` | integer | `db.py`, `validation.py`, `settlement.py`, exports | backtest-only |
| `result_code` | text | `db.py`, `validation.py`, query filters | backtest-only for settlement, also scratch filter |
| `is_winner` | integer | `db.py`, `validation.py`, `settlement.py`, `reporting.py` | backtest-only |
| `top3_mask` | integer | `db.py`, optional exports/reporting | backtest-only |

---

## Section 5: SQL queries

### 5.1 Backtest query

```sql
SELECT
  race_id, race_number, race_name, competition_id, competition_name, country,
  class_name, grade, tempo, distance_m, track_status, start_time_iso, field_size, active_field_size,
  selection_id, runner_number, runner_name, draw_number,
  jockey, trainer, trainer_location, weight_kg, age, sex,
  speed_rating, dry_rating, wet_rating,
  win_percentage, place_percentage, prize_money,
  career_starts, career_wins, career_seconds, career_thirds,
  good_starts, good_wins, soft_starts, soft_wins, heavy_starts, heavy_wins,
  distance_starts, distance_wins, track_starts, track_wins,
  first_up_starts, first_up_wins, second_up_starts, second_up_wins,
  horse_jockey_starts, horse_jockey_wins,
  last_six, form_fig, expected_settling_position,
  recent_runs_count, recent_wins, recent_places, recent_avg_place, recent_best_place,
  recent_avg_place_3, recent_avg_place_5, recent_win_rate_5, recent_top3_rate_5,
  recent_avg_margin, recent_best_margin, recent_avg_margin_3,
  recent_avg_starting_price, recent_same_distance_runs, recent_same_track_runs,
  recent_same_condition_runs, recent_days_since_last_run,
  open_price, fluc1, fluc2,
  sp_starting_price,  -- reference only, not used for live_price
  COALESCE(fluc2, fluc1, open_price) AS live_price,
  CASE
    WHEN fluc2 IS NOT NULL AND fluc2 > 0 THEN 'FLUC2'
    WHEN fluc1 IS NOT NULL AND fluc1 > 0 THEN 'FLUC1'
    WHEN open_price IS NOT NULL AND open_price > 0 THEN 'OPEN_ONLY'
    ELSE 'NO_PRICE'
  END AS price_quality,
  finish_place, result_code, status, is_winner, top3_mask
FROM race_runners
WHERE status = 'finished'
  AND result_code IN ('W', 'P', 'L')
  AND result_code != 'V'  -- exclude late scratched
ORDER BY start_time_iso, race_id, runner_number;
```

### 5.2 Live candidates query

```sql
SELECT
  race_id, race_number, race_name, competition_id, competition_name, country,
  class_name, grade, tempo, distance_m, track_status, start_time_iso, field_size, active_field_size,
  selection_id, runner_number, runner_name, draw_number,
  jockey, trainer, trainer_location, weight_kg, age, sex,
  speed_rating, dry_rating, wet_rating,
  win_percentage, place_percentage, prize_money,
  career_starts, career_wins, career_seconds, career_thirds,
  good_starts, good_wins, soft_starts, soft_wins, heavy_starts, heavy_wins,
  distance_starts, distance_wins, track_starts, track_wins,
  first_up_starts, first_up_wins, second_up_starts, second_up_wins,
  horse_jockey_starts, horse_jockey_wins,
  last_six, form_fig, expected_settling_position,
  recent_runs_count, recent_wins, recent_places, recent_avg_place, recent_best_place,
  recent_avg_place_3, recent_avg_place_5, recent_win_rate_5, recent_top3_rate_5,
  recent_avg_margin, recent_best_margin, recent_avg_margin_3,
  recent_avg_starting_price, recent_same_distance_runs, recent_same_track_runs,
  recent_same_condition_runs, recent_days_since_last_run,
  open_price, fluc1, fluc2,
  sp_starting_price,
  COALESCE(fluc2, fluc1, open_price) AS live_price,
  CASE
    WHEN fluc2 IS NOT NULL AND fluc2 > 0 THEN 'FLUC2'
    WHEN fluc1 IS NOT NULL AND fluc1 > 0 THEN 'FLUC1'
    WHEN open_price IS NOT NULL AND open_price > 0 THEN 'OPEN_ONLY'
    ELSE 'NO_PRICE'
  END AS price_quality,
  finish_place, result_code, status, is_winner, top3_mask
FROM race_runners
WHERE status = 'no_result'
  AND result_code != 'V'
  AND COALESCE(fluc2, fluc1, open_price) IS NOT NULL
ORDER BY start_time_iso, race_id, runner_number;
```

### 5.3 Price coverage query

```sql
SELECT
  race_id,
  COUNT(*) AS total_runners,
  SUM(CASE WHEN COALESCE(fluc2, fluc1, open_price) IS NOT NULL THEN 1 ELSE 0 END) AS priced_runners,
  CAST(SUM(CASE WHEN COALESCE(fluc2, fluc1, open_price) IS NOT NULL THEN 1 ELSE 0 END) AS REAL)
    / COUNT(*) AS price_coverage
FROM race_runners
WHERE status IN ('finished', 'no_result')
  AND result_code != 'V'
GROUP BY race_id;
```

### 5.4 SP comparison query (reference only, not selection)

```sql
SELECT
  race_id, selection_id, runner_name,
  COALESCE(fluc2, fluc1, open_price) AS live_price,
  sp_starting_price,
  sp_starting_price - COALESCE(fluc2, fluc1, open_price) AS price_drift
FROM race_runners
WHERE status = 'finished'
  AND result_code IN ('W', 'P', 'L')
ORDER BY start_time_iso, race_id, runner_number;
```

### 5.5 Backtest summary validation query

```sql
SELECT
  COUNT(*) AS total_bets,
  SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) AS wins,
  CAST(SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) AS strike_rate,
  SUM(CASE WHEN is_winner = 1 THEN COALESCE(fluc2, fluc1, open_price) - 1 ELSE -1 END) AS flat_profit
FROM race_runners
WHERE status = 'finished'
  AND result_code IN ('W', 'P', 'L')
  AND COALESCE(fluc2, fluc1, open_price) BETWEEN 2.0 AND 12.0
  AND field_size BETWEEN 6 AND 14;
```

---

## Section 6: Config structure (complete)

```python
CONFIG = {
    # Database
    "database_path": "database/race_reports.sqlite",   # str: path to SQLite DB

    # Column identifiers
    "race_id_col": "race_id",
    "runner_id_col": "selection_id",
    "result_col": "finish_place",
    "winner_col": "is_winner",

    # Price columns - live_price is derived, sp is reference only
    "live_price_columns": ["fluc2", "fluc1", "open_price"],  # ordered by priority
    "live_price_column": "live_price",            # derived column name
    "settlement_price_column": "live_price",      # used for profit calc in backtest
    "sp_reference_column": "sp_starting_price",   # reference only, never for selection

    # Backtest mode
    "backtest_price_mode": "latest_pre_race",     # always use live_price

    # Strategy v1 thresholds
    "min_price": 2.0,
    "max_price": 12.0,
    "min_field_size": 6,
    "max_field_size": 14,
    "max_model_rank": 2,
    "min_edge": 0.05,

    # Staking
    "stake": 1.0,                                  # flat staking, v1 only
    "allow_multiple_bets_per_race": False,

    # Price-quality filters
    "exclude_runner_if_no_live_price": True,
    "exclude_race_if_price_coverage_below": 0.80,

    # Form filters
    "min_recent_form_count": 2,

    # Calibration
    "min_races_for_bucket": 30,

    # Scoring weights (v1 defaults)
    "weight_speed_rating": 0.35,
    "weight_recent_form": 0.30,
    "weight_suitability": 0.20,
    "weight_connections": 0.10,
    "weight_market_sanity": 0.05,
}
```

### Key-by-key descriptions

- `database_path: str` — relative or absolute SQLite path used by all loaders.
- `race_id_col: str` — canonical race key.
- `runner_id_col: str` — canonical runner/selection key.
- `result_col: str` — finish-position column used in settlement validation and exports.
- `winner_col: str` — ground-truth winner indicator used in settlement.
- `live_price_columns: list[str]` — ordered live price fallback priority.
- `live_price_column: str` — derived selected live price column name.
- `settlement_price_column: str` — exact column used by historical settlement.
- `sp_reference_column: str` — SP reference column for diagnostics only.
- `backtest_price_mode: str` — documents that backtests use latest pre-race price, not SP.
- `min_price: float` — lower allowed live price bound.
- `max_price: float` — upper allowed live price bound.
- `min_field_size: int` — lower allowed declared field size.
- `max_field_size: int` — upper allowed declared field size.
- `max_model_rank: int` — highest allowed model rank for eligible bets.
- `min_edge: float` — minimum probability edge in decimal form.
- `stake: float` — flat stake per selected bet.
- `allow_multiple_bets_per_race: bool` — if false, keep only the highest-edge runner per race.
- `exclude_runner_if_no_live_price: bool` — if true, null or invalid live-price rows are dropped.
- `exclude_race_if_price_coverage_below: float` — race-level minimum share of runners with valid live prices.
- `min_recent_form_count: int` — minimum recent sample size to avoid sparse-form runners.
- `min_races_for_bucket: int` — reserved for later calibration or bucketed reporting safeguards.
- `weight_speed_rating: float` — score weight for condition-aware ability rating.
- `weight_recent_form: float` — score weight for recent-form component.
- `weight_suitability: float` — score weight for distance/track/condition fit.
- `weight_connections: float` — score weight for horse-jockey combination.
- `weight_market_sanity: float` — small stabilizing score weight from market context.

---

## Section 7: Shared pipeline flow

```text
Input: raw_df (one row per runner, loaded from race_runners)

Step 1: validate_input(raw_df, config)
  → check required columns
  → check for duplicate runners
  → check live_price is present and valid
  → check field_size within range
  → in backtest mode: check result labels present
  → output: validated_df

Step 2: build_features(validated_df, config)
  → add condition_rating (wet vs dry)
  → add suitability_score (distance/track/condition win rates)
  → add recent_form_score
  → add connection_score
  → add market_rank (rank by live_price within race)
  → add price_band
  → add distance_band
  → add form_recency_flag
  → output: features_df

Step 3: score_runners(features_df, config)
  → compute weighted model_score from components
  → add model_rank within race
  → output: scored_df

Step 4: assign_probabilities(scored_df, config)
  → softmax model_score within race → raw_model_prob
  → 1/live_price → market_implied_prob
  → output: probability_df

Step 5: calibrate_probabilities(probability_df, config)
  → v1: passthrough, model_prob = raw_model_prob
  → output: calibrated_df

Step 6: calculate_edges(calibrated_df, config)
  → edge = model_prob - market_implied_prob
  → add edge_pct, price_vs_sp (reference)
  → output: edge_df

Step 7: apply_filters(edge_df, config)
  → filter_no_live_price
  → filter_price_range
  → filter_field_size
  → filter_min_edge
  → filter_model_rank
  → filter_sparse_form
  → filter_price_coverage
  → filter_one_per_race (if allow_multiple_bets_per_race=False)
  → output: candidate_df

Step 8: assign_stakes(candidate_df, config)
  → v1: flat stake = 1.0
  → output: bet_df

Backtest only - Step 9: settle_bets(bet_df, config)
  → profit = (live_price - 1) * stake if is_winner=1 else -stake
  → output: result_df

Backtest only - Step 10: build_report(result_df, config)
  → summary metrics + breakdowns
  → output: report_dict
```

### Flow guarantees

- The same transformation order is used in backtest and live modes.
- No step may change row grain away from one-row-one-runner.
- SP-derived fields are downstream diagnostics only.
- Settlement is the only backtest-only transformation.

---

## Section 8: Backtest flow

Detailed steps for `scripts/run_backtest.py`:

1. Parse CLI args.
2. Load `CONFIG` and apply any runtime overrides.
3. Call `db.load_race_runners(db_path, query_mode="backtest", config)`.
   - Returns a DataFrame with `is_winner`, `finish_place`, `result_code`, and `status` populated.
   - `live_price = COALESCE(fluc2, fluc1, open_price)` is already computed in SQL.
4. Call `run_pipeline(df, config)`.
5. Call `settle_bets(bet_df, config)`.
6. Call `build_report(result_df, config)`.
7. Print the summary.
8. Export CSVs to `outputs/backtests/`.

### What to print

- Total bets
- Total staked
- Total profit
- ROI %
- Strike rate
- Average price
- Average edge
- Breakdown by price band (`bets`, `profit`, `roi`, `strike_rate`)
- Breakdown by track / competition
- Breakdown by distance band
- Breakdown by `track_status`
- Breakdown by field size band or exact field size
- Breakdown by `market_rank`
- Breakdown by `model_rank`
- Breakdown by `price_quality`

### Backtest invariants

- Settlement uses `live_price`, never `sp_starting_price`.
- Only `status = 'finished'` races are loaded.
- Late scratches are excluded before the pipeline begins.
- The report must still run even when zero bets survive filters.

---

## Section 9: Live candidate flow

Detailed steps for `scripts/list_live_bets.py`:

1. Parse CLI args.
2. Load config.
3. Call `db.load_race_runners(db_path, query_mode="live", config)`.
   - Returns only `status = 'no_result'` runners.
   - `live_price = COALESCE(fluc2, fluc1, open_price)`.
   - Result columns (`finish_place`, `is_winner`) may be null or empty; that is expected.
4. Call `run_pipeline(df, config)`.
5. Sort by `edge` descending.
6. Print candidate table with columns:
   - `start_time_iso`, `competition_name`, `race_number`, `runner_name`
   - `live_price`, `price_quality`, `open_price`, `fluc1`, `fluc2`
   - `model_score`, `model_rank`, `market_rank`
   - `model_prob`, `market_implied_prob`, `edge (%)`
   - `stake`

### Live-flow notes

- `sp_starting_price` is not used in this flow at all.
- The same `run_pipeline` function is used as backtest mode.
- The only difference from backtest mode is the SQL predicate and the absence of settlement/reporting.
- Because observed database rows include some `no_result` records with non-blank `result_code`, `status` remains the authoritative live selector.

---

## Section 10: Reporting outputs

### Output files

1. `outputs/backtests/backtest_YYYYMMDD_HHMMSS.csv` — full bet-level historical results
2. `outputs/backtests/backtest_summary_YYYYMMDD_HHMMSS.txt` — printed summary captured to text
3. `outputs/live/candidates_YYYYMMDD_HHMMSS.csv` — live candidates
4. `outputs/reports/breakdown_YYYYMMDD_HHMMSS.csv` — grouped breakdowns

### 1. `backtest_YYYYMMDD_HHMMSS.csv` columns

Recommended columns, in order:

- Identity and context:
  - `race_id`
  - `race_number`
  - `race_name`
  - `competition_id`
  - `competition_name`
  - `country`
  - `class_name`
  - `grade`
  - `tempo`
  - `distance_m`
  - `distance_band`
  - `track_status`
  - `start_time_iso`
  - `field_size`
  - `active_field_size`
  - `selection_id`
  - `runner_number`
  - `runner_name`
  - `draw_number`
  - `jockey`
  - `trainer`
  - `trainer_location`
- Base prices:
  - `open_price`
  - `fluc1`
  - `fluc2`
  - `live_price`
  - `price_quality`
  - `sp_starting_price`
  - `price_vs_sp`
- Derived model fields:
  - `condition_rating`
  - `suitability_score`
  - `recent_form_score`
  - `connection_score`
  - `market_rank`
  - `model_score`
  - `model_rank`
  - `raw_model_prob`
  - `model_prob`
  - `market_implied_prob`
  - `edge`
  - `edge_pct`
- Filter and stake fields:
  - `has_sparse_recent_form`
  - `stake`
- Result and settlement fields:
  - `finish_place`
  - `result_code`
  - `status`
  - `is_winner`
  - `top3_mask`
  - `profit`
  - `profit_net`

### 2. `backtest_summary_YYYYMMDD_HHMMSS.txt` sections

- Run timestamp
- Effective config values
- Summary metrics block
- Price-band breakdown
- Track / competition breakdown
- Distance-band breakdown
- Track-status breakdown
- Field-size breakdown
- Market-rank breakdown
- Model-rank breakdown
- Price-quality breakdown
- Notes on exclusions / counts after filters

### 3. `candidates_YYYYMMDD_HHMMSS.csv` columns

- `start_time_iso`
- `competition_name`
- `race_id`
- `race_number`
- `race_name`
- `runner_number`
- `runner_name`
- `draw_number`
- `distance_m`
- `track_status`
- `field_size`
- `open_price`
- `fluc1`
- `fluc2`
- `live_price`
- `price_quality`
- `market_rank`
- `condition_rating`
- `suitability_score`
- `recent_form_score`
- `connection_score`
- `model_score`
- `model_rank`
- `raw_model_prob`
- `model_prob`
- `market_implied_prob`
- `edge`
- `edge_pct`
- `stake`

### 4. `breakdown_YYYYMMDD_HHMMSS.csv` columns

This should be a long-format export combining all breakdowns.

- `breakdown_type` — e.g. `price_band`, `competition_name`, `distance_band`, `track_status`, `field_size`, `market_rank`, `model_rank`, `price_quality`
- `group_value`
- `bets`
- `wins`
- `total_staked`
- `total_profit`
- `roi`
- `strike_rate`
- `avg_price`
- `avg_edge`

---

## Section 11: Tests to write

### Shared fixtures to include

- `base_config` — a minimal config fixture copied from `CONFIG`
- `sample_price_rows` — price fallback cases
- `sample_score_race` — multi-runner race with known scores/prices
- `sample_probability_rows` — edge cases for model and market probabilities
- `sample_settlement_rows` — winner/loser settlement rows
- `sample_query_frames` — minimal finished/live mock frames for leakage tests

### `tests/test_prices.py`

**What it tests**: live-price derivation priority and price-quality classification.

**Test functions**:

- `test_live_price_prefers_fluc2` — asserts `fluc2` wins over `fluc1` and `open_price`
- `test_live_price_falls_back_to_fluc1_when_fluc2_null` — asserts `fluc1` is used when `fluc2` is missing
- `test_live_price_falls_back_to_open_price` — asserts `open_price` is used when both fluctuation prices are missing
- `test_live_price_is_null_when_all_prices_null` — asserts null `live_price` when no live inputs exist
- `test_sp_not_used_in_live_price` — asserts `sp_starting_price` never changes `live_price`
- `test_price_quality_classification` — asserts `FLUC2` / `FLUC1` / `OPEN_ONLY` / `NO_PRICE` mapping

### `tests/test_probabilities.py`

**What it tests**: within-race softmax logic and reciprocal market probability logic.

**Test functions**:

- `test_softmax_sums_to_one_within_race` — total probability per race equals `1.0`
- `test_softmax_highest_score_gets_highest_prob` — best score gets highest probability
- `test_market_implied_prob_is_reciprocal_of_price` — checks `1 / live_price`
- `test_market_implied_prob_zero_for_null_price` — null/invalid price returns `0.0`

### `tests/test_edge.py`

**What it tests**: core edge arithmetic.

**Test functions**:

- `test_positive_edge_when_model_prob_exceeds_market` — positive edge case
- `test_negative_edge_when_market_prob_exceeds_model` — negative edge case
- `test_edge_is_zero_when_equal` — zero edge case
- `test_edge_pct_is_100x_edge` — `edge_pct` scaling

### `tests/test_settlement.py`

**What it tests**: profit and ROI under flat-stake win-only settlement.

**Test functions**:

- `test_winner_profit_is_price_minus_one` — checks winner settlement
- `test_loser_profit_is_minus_one` — checks loser settlement
- `test_roi_calculation` — checks aggregate ROI formula
- `test_profit_uses_live_price_not_sp` — proves settlement ignores SP

### `tests/test_no_leakage.py`

**What it tests**: separation of backtest/live queries and reference-only SP handling.

**Test functions**:

- `test_backtest_query_excludes_no_result_races` — loader/query predicate safety
- `test_live_query_excludes_finished_races` — loader/query predicate safety
- `test_settlement_does_not_use_sp_starting_price` — settlement safety
- `test_pipeline_does_not_use_sp_for_edge_calculation` — selection safety
- `test_sp_column_exists_but_is_reference_only` — presence without selection impact

---

## Section 12: Implementation order

1. `betting/config.py` — config dict, all thresholds
2. `betting/schema.py` — required column lists
3. `betting/db.py` — `load_race_runners` with live-price SQL
4. `betting/validation.py` — all `check_` functions, `validate_input`
5. `betting/features.py` — all feature derivation functions, `build_features`
6. `betting/scoring.py` — `model_score`, `model_rank`
7. `betting/probabilities.py` — softmax, market-implied probability
8. `betting/calibration.py` — passthrough for v1
9. `betting/edge.py` — edge calculation
10. `betting/filters.py` — all filter functions
11. `betting/staking.py` — flat stake
12. `betting/settlement.py` — profit calculation, ROI
13. `betting/backtest.py` — `run_pipeline`, `run_backtest`
14. `betting/reporting.py` — summary metrics, breakdowns
15. `betting/live_candidates.py` — `run_live_candidates`
16. `scripts/run_backtest.py` — CLI entry point
17. `scripts/list_live_bets.py` — CLI entry point
18. `scripts/inspect_race.py` — CLI entry point
19. `scripts/export_backtest_report.py` — CLI entry point
20. `tests/` — all test files

---

## Section 13: Acceptance criteria per stage

### Stage 1 — `betting/config.py`
**Done means**:
- `CONFIG` dict exists.
- All required keys from Section 6 are present.
- Key types match their intended types.
- Importing `CONFIG` raises no error.

### Stage 2 — `betting/schema.py`
**Done means**:
- `REQUIRED_FEATURE_COLUMNS`, `REQUIRED_RESULT_COLUMNS`, `REQUIRED_PRICE_COLUMNS`, and `IDENTITY_COLUMNS` are defined.
- Lists cover every column used by their downstream modules.
- `validate_schema` raises `ValueError` with missing column names when input is incomplete.

### Stage 3 — `betting/db.py`
**Done means**:
- `load_race_runners` supports `query_mode='backtest'` and `query_mode='live'`.
- Returned DataFrame contains derived `live_price` and `price_quality` columns.
- `sp_starting_price` is present but not used in live-price derivation.
- Rows with `result_code='V'` are excluded.
- Loader ordering is stable by `start_time_iso`, `race_id`, `runner_number`.

### Stage 4 — `betting/validation.py`
**Done means**:
- `validate_input` calls all expected sub-checks.
- Duplicate `(race_id, selection_id)` rows are detectable.
- Invalid or missing `live_price` rows are flagged and handled per config.
- Out-of-range `field_size` races are flagged.
- Backtest mode flags missing result labels.

### Stage 5 — `betting/features.py`
**Done means**:
- All feature functions from Section 3 exist.
- `condition_rating`, `suitability_score`, `recent_form_score`, `connection_score`, `market_rank`, `price_band`, `distance_band`, and `has_sparse_recent_form` are added.
- One-row-one-runner grain is preserved.
- Zero-start suitability rates produce `0.0`, not `NaN`.

### Stage 6 — `betting/scoring.py`
**Done means**:
- `model_score` is computed from the configured weights.
- `model_rank` is added within each race.
- Highest `model_score` gets rank `1`.
- Missing component values do not crash the score calculation.

### Stage 7 — `betting/probabilities.py`
**Done means**:
- `raw_model_prob` is added and sums to `1.0` within each race.
- `market_implied_prob` equals `1 / live_price` for valid prices.
- Invalid prices map to `0.0` market probability.

### Stage 8 — `betting/calibration.py`
**Done means**:
- `calibrate_probabilities` adds `model_prob`.
- In v1, `model_prob == raw_model_prob` for all rows.
- The calibration hook is documented for later extension.

### Stage 9 — `betting/edge.py`
**Done means**:
- `edge` and `edge_pct` are added.
- `price_vs_sp` or equivalent SP comparison field is added.
- SP comparison fields are reference-only and not reused upstream.

### Stage 10 — `betting/filters.py`
**Done means**:
- All listed filters exist and run in the planned order.
- Price-range, field-size, min-edge, model-rank, sparse-form, and price-coverage logic are implemented.
- When `allow_multiple_bets_per_race=False`, at most one row per `race_id` remains.
- Filter drop counts are inspectable or logged.

### Stage 11 — `betting/staking.py`
**Done means**:
- `stake` column is added.
- Every surviving bet gets `stake == config['stake']`.
- No edge-based or Kelly logic exists in v1.

### Stage 12 — `betting/settlement.py`
**Done means**:
- `settle_bets` adds `profit` and `profit_net`.
- Winners settle at `(live_price - 1) * stake`.
- Losers settle at `-stake`.
- `compute_roi` returns `sum(profit) / sum(stake)`.
- `sp_starting_price` is not used.

### Stage 13 — `betting/backtest.py`
**Done means**:
- `run_pipeline` matches the shared flow exactly.
- `run_backtest` loads backtest data, runs the shared pipeline, then settles bets.
- Empty input or zero-bet outcomes still return valid DataFrames.

### Stage 14 — `betting/reporting.py`
**Done means**:
- `build_report` returns summary metrics plus all required breakdown tables.
- Summary metrics include total bets, total staked, total profit, ROI, strike rate, average price, and average edge.
- All breakdown helpers return a consistent schema.
- `print_report` and `export_csv` work on empty and non-empty data.

### Stage 15 — `betting/live_candidates.py`
**Done means**:
- `run_live_candidates` loads only live rows.
- It calls the exact same `run_pipeline` used by backtest mode.
- It returns candidates sorted by edge descending.
- It never calls settlement.

### Stage 16 — `scripts/run_backtest.py`
**Done means**:
- CLI parses expected overrides.
- Runtime config is built without mutating base `CONFIG`.
- Backtest runs end-to-end.
- Summary is printed and CSV/text outputs are written.

### Stage 17 — `scripts/list_live_bets.py`
**Done means**:
- CLI loads live candidates only.
- Printed table contains the required columns.
- Output is sorted by edge descending.
- Empty candidate slate is handled gracefully.

### Stage 18 — `scripts/inspect_race.py`
**Done means**:
- CLI accepts a `race_id`.
- It loads a single race and runs the shared pipeline.
- It prints runner-level ranking, scores, probabilities, and edges.
- Unknown race IDs exit cleanly with a no-data message.

### Stage 19 — `scripts/export_backtest_report.py`
**Done means**:
- CLI runs the backtest and exports full result CSV plus breakdown CSVs.
- Printed report matches `run_backtest.py` summary structure.
- Export paths are timestamped and created automatically.

### Stage 20 — `tests/`
**Done means**:
- All five test files exist.
- Every test listed in Section 11 is implemented.
- Tests explicitly prove live-price fallback order, probability math, edge math, settlement math, and no-leakage rules.
- Tests can run without needing the full production database by using minimal fixtures.

---

## Section 14: Assumptions and risks

### Assumptions

1. `live_price = COALESCE(fluc2, fluc1, open_price)`; if all three are null, the runner is excluded.
2. `sp_starting_price` is generally null for future races and only populated after result resolution.
3. `is_winner = 1` is the ground truth for settlement.
4. `status = 'no_result'` is the correct signal for live/unresolved races.
5. `result_code = 'V'` means late scratched and those rows are always excluded.
6. `field_size` is the pre-race declared field count and is the v1 filter column, not `active_field_size`.
7. Database views may contain aggregates built from historical outcomes, so using them in features risks data leakage; v1 therefore uses only base-table columns in the pipeline.
8. Softmax probability in v1 is only a rough probability estimate; a future calibration step is needed.
9. Scoring weights (`0.35 / 0.30 / 0.20 / 0.10 / 0.05`) are starting guesses and are not yet empirically validated.
10. Because direct inspection showed `no_result` rows with `result_code = 'L'`, query mode must trust `status` more than `result_code`.
11. One row in the DataFrame always corresponds to one runner; no function may aggregate away runner granularity.
12. The first implementation should avoid using inspected views like `race_runners_value_candidate_rows` or `race_runners_price_band_stats`, because they are likely post-result or aggregate products.

### Risks

1. **Price availability risk**: if many races only have `open_price`, the model may be forced to use stale prices rather than the latest market.
2. **Data leakage risk**: if any view or derived source embeds final-result knowledge, using it in scoring would overfit the backtest.
3. **Overfitting risk**: v1 score weights are hand-set, so good historical results may not generalize live.
4. **Sample-size risk**: `3,595` races over roughly three months is useful but may still be too small for strong confidence once sliced into many subgroups.
5. **Scratch/field-size risk**: removing `result_code='V'` rows means actual active runner counts can differ from the stored `field_size` used for filtering.
6. **Status/result inconsistency risk**: the observed `no_result/L` rows suggest some columns may update asynchronously, so SQL predicates must stay conservative.
7. **SP sparsity risk**: only `13,355` of `45,645` rows currently have positive SP values, which reinforces that SP comparisons will be incomplete and strictly diagnostic.
8. **Market-efficiency risk**: the strongest historical edges may collapse in live use once prices update closer to jump time.
9. **Operational drift risk**: if live scripts diverge from the shared `run_pipeline`, historical and live performance will no longer be comparable.

---

## Recommended implementation notes beyond the required sections

- Keep all feature engineering purely pre-race and base-table derived.
- Prefer explicit intermediate columns over opaque nested expressions; this makes inspection easier.
- Preserve `open_price`, `fluc1`, `fluc2`, `live_price`, and `price_quality` in all exported candidate/result files so price provenance stays visible.
- Include filter-stage counts in logs or in the summary text so the operator can see where most rows are being discarded.
- Start with deterministic pandas logic and avoid hidden state or fitting steps in v1.

---

## Deliverables completed

- Full source-of-truth-aligned implementation plan created.
- Plan grounded in actual `race_reports.sqlite` schema and data-shape inspection.
- All required 14 sections included.
- Exact module structure, functions, SQL, config, pipeline, flows, tests, acceptance criteria, assumptions, and risks documented.

**TODO STATUS: DONE**
