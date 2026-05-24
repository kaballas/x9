# Plan: Using `race_reports.sqlite` to Bet on Future Races

This plan assumes the goal is not to predict every race, but to find repeatable betting edges from the data already stored in `database/race_reports.sqlite`.

## 1. Define the betting target

Start by being explicit about the market we are trying to beat.

- Primary target: win bets on runners whose true chance is higher than the market price implies.
- Secondary target: place or each-way bets only when the field structure and odds support it.
- Skip races where the data is thin, the market is efficient, or the model edge is small.

Success means the process produces a positive expected value over many races, not a high strike rate in a small sample.

## 2. Use the database as the source of truth

The base table is `race_runners`, which gives us:

- race context: `competition_name`, `country`, `class_name`, `grade`, `tempo`, `distance_m`, `track_status`, `start_time_iso`, `field_size`
- runner quality: `speed_rating`, `dry_rating`, `wet_rating`, `win_percentage`, `place_percentage`, `prize_money`
- form: `last_six`, `form_fig`, `recent_*` fields
- connections: `jockey`, `trainer`, `trainer_location`, `horse_jockey_*`, `trainer_*`
- market: `open_price`, `fluc1`, `fluc2`, `sp_starting_price`
- results labels: `finish_place`, `result_code`, `is_winner`, `top3_mask`

The views are useful for precomputed signals such as favourite performance, market rank, draw stats, trainer/jockey stats, and track-specific profiles.

## 3. Build a simple pre-race ranking

Start with a score that ranks runners before the race starts.

Initial score inputs:

- recent form: `recent_wins`, `recent_places`, `recent_avg_place`, `recent_win_rate_5`, `recent_top3_rate_5`
- ability ratings: `speed_rating`, `dry_rating`, `wet_rating`
- suitability: `distance_starts`, `distance_wins`, `track_starts`, `track_wins`, `good/soft/heavy_*`
- connections: `jockey`, `trainer`, and the related stats views
- market sanity check: `open_price`, `fluc1`, `fluc2`, and market-rank views

Keep the first version boring:

- rank runners within a race
- compare that rank to the market rank
- bet only when the model rank is materially better than the market

## 4. Focus on a few repeatable edge types

Do not bet every race. Concentrate on patterns the data can support.

Good candidate edge buckets:

- strong favourite with weak track/condition fit
- mid-price runner with strong recent form and solid track/distance stats
- runner improving on second-up or first-up profiles
- horses with strong jockey/trainer combinations in the relevant conditions
- runners that the market underrates in fields with useful draw or tempo setups

The analytic views should help answer these questions quickly:

- Are favourites reliable at this track?
- Do certain draw bands outperform at this track/distance/condition?
- Does tempo favor front-runners or closers?
- Are there trainer/jockey or distance specialists worth upgrading?

## 5. Convert ratings into a price threshold

The model should not just rank runners. It should tell us when a bet is worth placing.

For each runner:

1. Estimate an implied win probability.
2. Convert the market price into implied probability.
3. Bet only when model probability exceeds market probability by a minimum margin.

Practical rule:

- require a gap before betting
- avoid tiny edges
- add a stronger threshold for bigger fields or noisy races

## 6. Backtest on historical races

Use past races in `race_runners` to test the plan.

Backtest checks:

- win ROI by price band
- place ROI if we consider place or each-way bets
- strike rate by track, distance, class, and field size
- performance against favourites and mid-price runners
- whether the edge survives after removing obvious overfit fields

Useful questions:

- Which track/condition combinations are profitable?
- Which runner profiles lose money even when they look good on paper?
- Does the model only work on short-priced runners?
- Does it still work when the market price moves late?

## 7. Separate candidate selection from staking

Do not let staking hide a weak selection process.

Selection logic:

- choose the top runners by model edge
- require minimum liquidity or confidence rules if needed
- ignore races where multiple runners cluster too closely

Staking logic:

- use flat staking first
- only move to fractional Kelly or similar after the edge is stable
- cap daily exposure
- cap per-race exposure

## 8. Add discipline around race filters

Before betting, filter out low-quality opportunities.

Typical filters:

- no bet if the field is too small or too chaotic
- no bet if the model and market disagree only slightly
- no bet if recent form data is sparse
- no bet if the runner has poor track/distance evidence and no compensating edge
- no bet if the race is outside the profile the model has been tested on

## 9. Track live performance

Every bet should be logged with:

- race ID
- runner ID
- model rank
- market price
- estimated probability
- stake size
- result
- profit/loss

Then review performance by:

- track
- distance
- class
- price band
- favourite vs non-favourite
- early price vs starting price

## 10. Keep an override list of failure cases

The data will eventually show runners or race types that look good but keep failing.

Track and exclude:

- bad favourite types
- overbet trainer/jockey combinations
- tracks where draw stats are misleading
- price bands that are consistently unprofitable
- race shapes where the model is overconfident

## 11. Suggested implementation order

1. Build a historical ranking script using `race_runners` and the main stats views.
2. Backtest the script on past races and score ROI by race type and price band.
3. Add a minimum edge threshold and rerun the backtest.
4. Add staking rules only after selection quality is stable.
5. Use the same scoring logic on future races and log every bet.

## 12. Implementation Shape

BHere’s what I think you meant.

Replace the plan with this version. Main changes from your current draft: `sp_starting_price` is no longer the live price, `live_price` is now `COALESCE(fluc2, fluc1, open_price)`, and the architecture adds price-quality, validation, calibration, and settlement controls. 

````markdown
# Plan: Using `race_reports.sqlite` to Bet on Future Races

This plan assumes the goal is not to predict every race, but to find repeatable betting edges from the data already stored in `database/race_reports.sqlite`.

The system should be built as a Python research and live-candidate framework. The same core pipeline should be used for historical backtesting and future race selection.

## 1. Define the betting target

The primary target is win bets on runners whose estimated true chance is higher than the available market price implies.

The first version should focus on win-only betting. Place and each-way betting should be added later only if the win-bet pipeline proves stable.

Success means the process produces positive expected value over many races. A high strike rate alone is not enough.

## 2. Use the database as the source of truth

The base table is `race_runners`.

Key fields:

- race context: `race_id`, `competition_name`, `country`, `class_name`, `grade`, `tempo`, `distance_m`, `track_status`, `start_time_iso`, `field_size`
- runner identity: `selection_id`, `runner_number`, `runner_name`, `draw_number`
- runner quality: `speed_rating`, `dry_rating`, `wet_rating`, `win_percentage`, `place_percentage`, `prize_money`
- form: `last_six`, `form_fig`, `recent_*`
- suitability: `distance_*`, `track_*`, `good_*`, `soft_*`, `heavy_*`, `first_up_*`, `second_up_*`
- connections: `jockey`, `trainer`, `trainer_location`, `horse_jockey_*`
- live market: `open_price`, `fluc1`, `fluc2`
- resulted market: `sp_starting_price`
- results labels: `finish_place`, `result_code`, `status`, `is_winner`, `top3_mask`

The views can be used for extra signals only if they are confirmed to be pre-race safe. Any view that uses final results must not leak future information into a backtest.

## 3. Define the price model

For future `PRICED` races, the available pre-race prices are:

```sql
open_price,
fluc1,
fluc2
````

The live betting price should be:

```sql
COALESCE(fluc2, fluc1, open_price) AS live_price
```

Meaning:

* `open_price` = first available price
* `fluc1` = later available price
* `fluc2` = latest available pre-race price
* `sp_starting_price` = resulted SP / reference price

For live candidate selection, use `live_price`.

For historical backtesting, use `live_price` by default to simulate the same process used on future races.

Use `sp_starting_price` only for:

* SP comparison
* market movement analysis
* optional SP-mode backtests
* checking whether the selected price was better or worse than final SP

Do not use `sp_starting_price` for live candidate selection.

## 4. Build a simple pre-race ranking

Start with a score that ranks runners before the race starts.

Initial score inputs:

* recent form: `recent_wins`, `recent_places`, `recent_avg_place`, `recent_win_rate_5`, `recent_top3_rate_5`
* ability ratings: `speed_rating`, `dry_rating`, `wet_rating`
* suitability: `distance_starts`, `distance_wins`, `track_starts`, `track_wins`, `good_starts`, `good_wins`, `soft_starts`, `soft_wins`, `heavy_starts`, `heavy_wins`
* race shape: `tempo`, `expected_settling_position`, `draw_number`, `field_size`
* connections: `jockey`, `trainer`, `horse_jockey_starts`, `horse_jockey_wins`
* market sanity check: `open_price`, `fluc1`, `fluc2`, `live_price`

The first version should be simple:

* rank runners within each race
* calculate market rank from `live_price`
* compare model rank to market rank
* bet only when the model has a material edge over the market

## 5. Convert ratings into probabilities and edge

The system must not stop at ranking. It needs a price threshold.

For each runner:

1. Calculate `model_score`.
2. Convert `model_score` into `raw_model_prob` within the race.
3. Calibrate probability if enough historical data exists.
4. Convert `live_price` into `market_implied_prob`.
5. Calculate the edge.

```text
market_implied_prob = 1 / live_price
edge = model_prob - market_implied_prob
```

The first version can use softmax probability, but it should be treated as a rough estimate, not a proven true probability.

The architecture should leave room for calibration:

```text
model_score
→ raw_model_prob
→ calibrated_model_prob
→ edge
```

## 6. Focus on repeatable edge types

Do not bet every race.

Candidate edge buckets:

* favourite looks short but has weak condition, distance, or track fit
* mid-price runner has strong recent form and solid suitability
* runner is improving first-up or second-up
* strong jockey/trainer or horse/jockey combination
* market underrates runner because of draw, tempo, or recent beaten margin
* runner has positive market movement but is still priced above estimated chance

Useful questions:

* Are favourites reliable at this track and distance?
* Are certain draw bands useful at this track?
* Does the tempo favour leaders, midfield runners, or backmarkers?
* Do certain trainers or jockeys perform better under specific conditions?
* Does late price movement improve or weaken the signal?

## 7. Backtest on historical races

Use resulted races first.

Default backtest mode:

```text
price used for bet = COALESCE(fluc2, fluc1, open_price)
settlement result = is_winner / finish_place
profit = live_price - 1 if winner
profit = -1 if loser
```

Optional comparison mode:

```text
SP comparison = sp_starting_price
```

Backtest checks:

* total bets
* total staked
* profit
* ROI
* strike rate
* average price
* average edge
* profit by price band
* profit by track
* profit by distance band
* profit by class
* profit by field size
* profit by market rank
* profit by model rank
* performance where `fluc2` exists versus open-price-only races

The backtest must show whether the model works before staking complexity is added.

## 8. Separate candidate selection from staking

Do not let staking hide a weak selection process.

Selection logic:

* select runners by edge
* require minimum model rank
* require minimum edge
* require acceptable price range
* require acceptable field size
* require price availability
* optionally allow only one bet per race

Staking logic:

* start with flat 1-unit staking
* do not use Kelly in version 1
* cap per-race exposure
* cap daily exposure
* add variable staking only after the edge survives backtesting

## 9. Add race and price-quality filters

Before betting, filter out weak opportunities.

Initial filters:

* no bet if `live_price` is missing
* no bet if `live_price` is outside the tested range
* no bet if field size is too small or too large
* no bet if recent form data is sparse
* no bet if the model edge is small
* no bet if multiple runners are too close in model score
* no bet if the race type has not performed well in backtesting

Price-quality classification:

```sql
CASE
    WHEN fluc2 IS NOT NULL THEN 'FLUC2'
    WHEN fluc1 IS NOT NULL THEN 'FLUC1'
    WHEN open_price IS NOT NULL THEN 'OPEN_ONLY'
    ELSE 'NO_PRICE'
END AS price_quality
```

Race-level price coverage should also be tracked.

## 10. Track live performance

Every live candidate and every actual bet should be logged.

Log fields:

* race ID
* runner ID
* competition name
* race number
* race start time
* runner name
* model score
* model rank
* market rank
* open price
* fluc1
* fluc2
* live price
* price quality
* estimated probability
* market implied probability
* edge
* stake
* result
* profit/loss

Review performance by:

* track
* distance
* class
* price band
* field size
* favourite versus non-favourite
* model rank
* market rank
* price quality
* early price versus SP

## 11. Keep an exclusion list of failure cases

The data will show race types or runner profiles that look good but lose money.

Track and exclude:

* bad favourite profiles
* unreliable tracks
* misleading draw patterns
* overbet trainer/jockey combinations
* unprofitable price bands
* weak race classes
* race shapes where the model is overconfident
* open-price-only races if they underperform

Do not add exclusions manually unless the backtest supports them.

## 12. Python implementation shape

Build this as a small Python framework, not one large script.

Recommended structure:

```text
x8/
├── database/
│   └── race_reports.sqlite
│
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
│
├── scripts/
│   ├── run_backtest.py
│   ├── inspect_race.py
│   ├── list_live_bets.py
│   └── export_backtest_report.py
│
├── tests/
│   ├── test_prices.py
│   ├── test_probabilities.py
│   ├── test_edge.py
│   ├── test_settlement.py
│   └── test_no_leakage.py
│
└── outputs/
    ├── backtests/
    ├── live/
    └── reports/
```

Core flow:

```text
SQLite DB
   ↓
db.py
   ↓
validation.py
   ↓
features.py
   ↓
scoring.py
   ↓
probabilities.py
   ↓
calibration.py
   ↓
edge.py
   ↓
filters.py
   ↓
staking.py
   ↓
settlement.py / live_candidates.py
   ↓
reporting.py
```

## 13. Module purpose

```text
db.py
Reads race_runners and creates canonical fields such as live_price and price_quality.

schema.py
Defines required columns and standard column names.

validation.py
Checks missing columns, duplicate runners, bad prices, invalid field sizes, and missing result labels.

features.py
Cleans inputs and derives race-relative features, price bands, distance bands, and form summaries.

scoring.py
Creates model_score for each runner.

probabilities.py
Converts model_score into raw_model_prob within each race.

calibration.py
Calibrates raw_model_prob into model_prob where enough historical data exists.

edge.py
Compares model_prob against market implied probability.

filters.py
Removes races and runners that should not be bet.

staking.py
Applies flat staking first.

settlement.py
Calculates result and profit/loss for historical bets.

backtest.py
Runs historical testing on resulted races.

live_candidates.py
Runs the same pipeline on PRICED or non-resulted races.

reporting.py
Outputs ROI, strike rate, profit, bet count, and breakdowns.
```

## 14. Data object flow

Use this flow:

```python
raw_df
validated_df
features_df
scored_df
probability_df
edge_df
candidate_df
bet_df
result_df
report_df
```

One row must remain one runner.

Every calculation should preserve:

```text
race_id
selection_id
runner_name
competition_name
race_number
race_name
start_time_iso
source_betting_status
```

## 15. Shared pipeline

Do not split historical and live logic.

The same core pipeline should handle both:

```python
def run_pipeline(df, config):
    df = validate_input(df, config)
    df = build_features(df, config)
    df = score_runners(df, config)
    df = assign_probabilities(df, config)
    df = calibrate_probabilities(df, config)
    df = calculate_edges(df, config)
    df = apply_filters(df, config)
    df = assign_stakes(df, config)
    return df
```

Historical backtest adds settlement:

```python
def run_backtest(df, config):
    bets = run_pipeline(df, config)
    results = settle_bets(bets, config)
    return build_report(results, config)
```

Live betting only changes the input query:

```python
def run_live_candidates(df, config):
    candidates = run_pipeline(df, config)
    return candidates.sort_values("edge", ascending=False)
```

## 16. Strategy v1

The first strategy should be deliberately narrow.

```text
Strategy v1:
Win-only
One bet per race
Top 1 or Top 2 model-ranked runner
live_price between 2.00 and 12.00
field size between 6 and 14
model edge >= 5%
flat 1 unit stake
exclude runners with no live price
exclude races with poor price coverage
```

Do not add each-way, place betting, Kelly staking, or manual overrides in version 1.

## 17. Configuration

Configuration should live in one file.

```python
CONFIG = {
    "database_path": "database/race_reports.sqlite",

    "race_id_col": "race_id",
    "runner_id_col": "selection_id",
    "result_col": "finish_place",
    "winner_col": "is_winner",

    "live_price_columns": ["fluc2", "fluc1", "open_price"],
    "live_price_column": "live_price",
    "settlement_price_column": "live_price",
    "sp_reference_column": "sp_starting_price",

    "backtest_price_mode": "latest_pre_race",

    "min_price": 2.0,
    "max_price": 12.0,
    "min_field_size": 6,
    "max_field_size": 14,

    "max_model_rank": 2,
    "min_edge": 0.05,

    "stake": 1.0,
    "allow_multiple_bets_per_race": False,

    "exclude_runner_if_no_live_price": True,
    "exclude_race_if_price_coverage_below": 0.80,

    "min_recent_form_count": 2,
    "min_races_for_bucket": 30,
}
```

## 18. Suggested implementation order

Build in this order:

```text
1. db.py
2. schema.py
3. validation.py
4. features.py
5. scoring.py
6. probabilities.py
7. calibration.py
8. edge.py
9. filters.py
10. staking.py
11. settlement.py
12. backtest.py
13. reporting.py
14. live_candidates.py
15. tests
```

The first useful milestone is:

```text
Run a leakage-safe historical win-only backtest on resulted races using:
- live_price = COALESCE(fluc2, fluc1, open_price)
- flat 1-unit staking
- one bet per race
- no each-way
- no Kelly
- no manual overrides
```

Report:

```text
ROI
profit
strike rate
bet count
average price
average edge
performance by price band
performance by field size
performance by track condition
performance by distance band
performance by market rank
performance by price quality
```

## 19. What makes the plan useful

The system is useful only if it can answer these questions before betting:

* Is this runner priced above its estimated true chance?
* Is the available price based on open price, fluc1, or fluc2?
* Is this race type one the model has actually handled well?
* Is the edge large enough to survive variance?
* Is the stake capped so one bad day does not erase many small wins?

That is the standard before risking money.

```
```
