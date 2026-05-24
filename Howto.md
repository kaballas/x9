# Betting Framework — How To Use

A win-only horse racing betting framework that finds repeatable edges using
pre-race market prices. All candidate selection uses
`live_price = COALESCE(fluc2, fluc1, open_price)`. SP (`sp_starting_price`) is
never used for selection — it is loaded as a reference column only.

---

## Requirements

- Python 3.9+
- pandas, numpy (install once below)
- The SQLite database at `database/race_reports.sqlite`

```bash
pip install pandas numpy pytest
```

All scripts are run from the project root (`/home/theo/perplex/x7/x9`).

---

## Project layout

```
x9/
├── betting/               # Core pipeline package
│   ├── config.py          # All thresholds and weights — edit here
│   ├── db.py              # SQL queries, live_price derivation
│   ├── features.py        # Score components (speed, form, suitability…)
│   ├── scoring.py         # Composite model score + model rank
│   ├── probabilities.py   # Softmax model probs, market-implied probs
│   ├── calibration.py     # Isotonic calibration (passthrough if no .pkl)
│   ├── edge.py            # edge = model_prob − market_implied_prob
│   ├── filters.py         # 8-stage filter pipeline
│   ├── staking.py         # Flat + Fibonacci stake sizing
│   ├── settlement.py      # Profit/loss, ROI
│   ├── backtest.py        # run_pipeline(), run_backtest()
│   ├── live_candidates.py # run_live_candidates()
│   └── reporting.py       # Summary + 8 breakdowns, CSV/TXT export
├── scripts/
│   ├── run_backtest.py          # Run backtest, print + export report
│   ├── list_live_bets.py        # Show today's live candidates
│   ├── inspect_race.py          # Debug a single race (all filters off)
│   ├── fit_calibration.py       # Fit isotonic calibration model from backtest data
│   ├── recalibrate_cycle.py     # Automated weights → calibration → filters loop
│   └── export_backtest_report.py # Export full CSV report suite
├── tests/                 # 81 pytest tests
├── database/
│   └── race_reports.sqlite
└── outputs/
    ├── backtests/
    ├── live/
    └── reports/
```

---

## 1 — Run the backtest

Runs the full historical backtest on all `status='finished'` races and prints
a report to the terminal.

```bash
cd /home/theo/perplex/x7/x9
python3 scripts/run_backtest.py
```

**Sample output:**

```
SUMMARY
Total Bets: 114
Total Staked: 114.000
Total Profit: 52.500
Roi: 46.05%
Strike Rate: 15.79%
Avg Price: 9.560
Avg Edge: 9.23%
Wins: 18

PRICE BAND
 breakdown_type group_value  bets  wins  ...
...
```

Three output files are written to `outputs/backtests/` automatically:

| File | Contents |
|---|---|
| `backtest_YYYYMMDD_HHMMSS.csv` | One row per settled bet |
| `breakdown_YYYYMMDD_HHMMSS.csv` | All 8 breakdowns concatenated |
| `summary_YYYYMMDD_HHMMSS.txt` | Same text as terminal output |

### Override config on the command line

```bash
# Tighten edge threshold
python3 scripts/run_backtest.py --min-edge 0.10

# Narrow the price window
python3 scripts/run_backtest.py --min-price 3.0 --max-price 8.0

# Different field-size bracket
python3 scripts/run_backtest.py --min-field-size 8 --max-field-size 12

# Point at a different database
python3 scripts/run_backtest.py --db /path/to/other.sqlite

# Write outputs to a custom directory
python3 scripts/run_backtest.py --output-dir outputs/my_run
```

All overrides are applied on top of `betting/config.py`; the config file is
not modified.

---

## 2 — List today's live bet candidates

Queries all `status='no_result'` races (unresolved/upcoming) and runs them
through the same filter pipeline.

```bash
python3 scripts/list_live_bets.py
```

**Sample output:**

```
 start_time_iso  competition_name  race_number  runner_name  live_price  ...  edge  stake
...
Total candidates: 38
```

### Options

```bash
# Raise the edge threshold for live bets
python3 scripts/list_live_bets.py --min-edge 0.10

# Export candidates to CSV
python3 scripts/list_live_bets.py --output outputs/live/today.csv

# Both
python3 scripts/list_live_bets.py --min-edge 0.08 --output outputs/live/today.csv
```

Columns shown: `start_time_iso`, `competition_name`, `race_number`,
`runner_name`, `live_price`, `price_quality`, `open_price`, `fluc1`, `fluc2`,
`model_score`, `model_rank`, `market_rank`, `model_prob`,
`market_implied_prob`, `edge`, `stake`.

The `edge` column is displayed as a percentage (e.g. `12.3%`).

`market_implied_prob` is now normalised within each race to remove bookmaker overround. For example, raw prices in a 14-runner race might sum to 121% implied probability; the framework divides each runner's `1/live_price` by the field total so the race sums to exactly 100%, giving a fairer edge comparison.

**Before:** `market_implied_prob = 1 / live_price`  
**After:** `market_implied_prob = (1/live_price) / sum(1/live_price for all in race)`

---

## 7. Inspecting a Race

Use `inspect_race.py` to analyse a single race in detail.

### Basic usage
```bash
python3 scripts/inspect_race.py <race_id>
```

This shows:
1. **Race header** — name, venue, distance, track condition, field size
2. **Pipeline validation** — row counts at each filter step (useful for debugging)
3. **MODEL RANKING** — all runners ranked by model score with per-component weighted contributions
4. **BET CANDIDATES** — runners passing the full config thresholds (edge, price, rank)

### MODEL RANKING table columns

| Column   | Meaning                                                                              |
|----------|--------------------------------------------------------------------------------------|
| Rank     | Model rank (1 = highest model score)                                                 |
| Runner   | #number Name                                                                         |
| Price    | live_price (FLUC2 → FLUC1 → open_price)                                              |
| Speed    | weight_speed_rating × speed_score_norm (normalised condition_rating)                 |
| Form     | weight_recent_form × recent_form_score (50% aggregate stats + 50% position quality across last 6 runs) |
| Suit     | weight_suitability × suitability_score                                               |
| Conn     | weight_connections × connection_score                                                |
| Mkt      | weight_market_sanity × market_sanity_score (normalised inverse price)                |
| Margin   | weight_margin × margin_score — recent beaten margins; **smaller is better** (0 = beaten by ≥8 lengths avg, 1.0 = winner). Uses avg of last 3 runs (80%) + best run (20%). Caps: avg at 8 lengths, best at 4 lengths. Missing data defaults to 8 lengths avg (neutral-bad). |
| Fresh    | weight_freshness × freshness_score                                                   |
| Class    | weight_class × class_score                                                           |
| Score    | Sum of all weighted contributions = model_score                                      |
| ModelP%  | Normalised model probability (%)                                                     |
| MktP%    | Overround-normalised market implied probability = `(1/live_price) / sum(1/live_price for all runners in race)` (%) |
| Edge%    | ModelP% − MktP% (positive = value bet; fairer than raw `1/live_price` because the race sums to 100%) |
| Bet?     | ✓ BET if edge ≥ min_edge AND live_price in [min_price, max_price] AND model_rank ≤ max_model_rank — same three filters as the BET CANDIDATES section |

### Example output (race 10480193)

> The exact weight line shown by `inspect_race.py` comes from the current `betting/config.py`. The sample below is illustrative and may differ from the latest defaults documented in section 6.

```text
Race: R8 Hilton Nicholas Tab Straight Six Hcp | Flemington | Race 8
Distance: 1200m | Track: Good (4) | Field: 15
Start: 2026-05-16T06:10:00+00:00

════════════════════════════════════════════════════════════════════════════════════════════════════
  MODEL RANKING  (  weights: Speed=0.28  Form=0.22  Suit=0.15  Conn=0.07  Mkt=0.05  Margin=0.10  Fresh=0.08  Class=0.05)
════════════════════════════════════════════════════════════════════════════════════════════════════
 Rank               Runner  Price  Speed  Form  Suit  Conn   Mkt  Margin  Fresh  Class  Score  ModelP%  MktP%   Edge%  Bet?
    1      #12 Stoli Bolli  2.900  0.280 0.095 0.034 0.018 0.048   0.059  0.059  0.027  0.620    8.800 34.500 -25.700     ✗
    2   #8 Losesomewinmore 15.000  0.238 0.084 0.071 0.000 0.007   0.069  0.062  0.027  0.558    8.300  6.700   1.600     ✗
    ...
```

### Verbose mode (raw score inputs)
```bash
python3 scripts/inspect_race.py <race_id> --verbose
```

Shows a detailed breakdown per runner of the raw database inputs used to compute each component score. Example:

```text
  ── #12 Stoli Bolli ──────────────────────────────────────────────
  Speed  [Good (4)]  wet_rating=96.0  dry_rating=100.0  →  using dry_rating=100.0
         condition_rating=100.0  →  normalised=1.000  ×  0.28  =  0.2800
         field_min=76.0  field_max=100.0  →  (100.0 - 76.0) / (100.0 - 76.0)  =  1.000
  Form   win_rate_5=0.20  top3_rate_5=0.60  avg_place=4.2  avg_place_3=3.0
         pos_quality=0.433  runs[r1→r6]: 3/12  2/10  5/11  6/13  4/10  1/9  →  blended=0.433  ×  0.22  =  0.0953
  Suit   dist=3/8(0.38)  track=1/3(0.33)  →  score=0.227  ×  0.15  =  0.0341
  Conn   jockey_wins=1/4  →  score=0.250  ×  0.07  =  0.0175
  Mkt    live_price=2.90  →  norm_inv_price=0.960  ×  0.05  =  0.0480
  Margin avg_margin_3=+1.5  best_margin=+3.2  →  score=0.591  ×  0.10  =  0.0591
  Fresh  days_since=14  first_up=1/3  →  score=0.736  ×  0.08  =  0.0589
  Class  prize=$45,000  place_pct=45.0%  career=5/18  →  score=0.540  ×  0.05  =  0.0270
  ─────────────────────────────────────────────────────────────────
  MODEL SCORE = 0.6203   ModelP=8.8%   MktP=34.5%   Edge=-25.7%
```

Use `--verbose` when diagnosing why a runner ranked higher or lower than expected.

The Speed lines now show the actual track condition, both raw wet/dry ratings, the selected rating column, and the full field normalisation formula. The Form lines now show the aggregate stats plus the position-quality trace used in the 50/50 blend.

#### Form scoring detail

The form score is a 50/50 blend of two signals:

| Signal | Weight | Inputs |
|--------|--------|--------|
| Aggregate stats | 50% | `win_rate_5` (40%), `top3_rate_5` (30%), `avg_place` (15%), `avg_place_3` (15%) |
| Position quality | 50% | Last 6 finish positions, field-normalized and recency-weighted |

**Position quality** converts each finish position to a field-size-adjusted score: a 1st in a 15-runner race scores 1.0; 5th in a 10-runner race scores `(10-5)/(10-1) = 0.556`. Missing positions use a neutral prior of 0.30.

**Recency weights**: `[0.35, 0.25, 0.15, 0.10, 0.10, 0.05]` — most recent run gets 35% of the position quality score, oldest 5%.

The `runs[r1→r6]` display in verbose mode shows `place/field` for each of the last 6 starts (r1 = most recent). A `-` means no data for that slot.

---

## 4 — Export a full report suite

Same as `run_backtest.py` but defaults to `outputs/reports/` and is focused
on producing the export files rather than the terminal output.

```bash
python3 scripts/export_backtest_report.py

# Custom output directory
python3 scripts/export_backtest_report.py --output-dir outputs/weekly_review

# Override price filter for export only
python3 scripts/export_backtest_report.py --min-price 3.0 --max-price 10.0
```

---

## 5 — Run the test suite

```bash
cd /home/theo/perplex/x7/x9
PYTHONPATH=. python3 -m pytest tests/ -v
```

Expected: **81 passed, 0 failed**.

Key things the tests enforce:

| Test file | What it checks |
|---|---|
| `test_prices.py` | `live_price` fallback order; SP never used; `price_quality` labels |
| `test_probabilities.py` | Softmax sums to 1.0; market probs are overround-normalised within each race |
| `test_edge.py` | Positive/negative/zero edge; SP does not affect edge |
| `test_settlement.py` | Winner/loser profit; settlement uses `live_price` not SP |
| `test_no_leakage.py` | Backtest/live query separation; SP reference-only policy |
| `test_calibration.py` | Passthrough fallback; isotonic model applied + renormalised; cache; no mutation |

---

## 5b — Calibrate the model

Calibration maps the model's raw probability outputs to actual win rates using
isotonic regression fitted on all backtest rows. Without calibration the model
overrates longshots and underrates favourites.

**Fit the calibration model** (run whenever you have new data, after weight changes, or after feature engineering changes that affect scoring):

```bash
python3 scripts/fit_calibration.py
```

This loads all backtest rows, scores them without filters, then fits a monotone
mapping from `raw_model_prob` → actual win rate and saves it to
`betting/calibration_model.pkl`. The pipeline picks it up automatically on the
next run.

**Dry-run** (print stats, do not save):
```bash
python3 scripts/fit_calibration.py --dry-run
```

**What good calibration looks like** (calibration ratio ≈ 1.0 in every bucket):

```
prob_bucket  raw_ratio  cal_ratio
0-3%          0.00       NaN      ← too few winners to measure
3-5%          0.37       1.01     ← was badly overrated, now corrected
5-8%          0.68       1.01     ← was moderately overrated, now corrected
8-12%         0.96       1.00     ← near-perfect
12-17%        1.21       1.00     ← was underrated, now corrected
17-25%        1.26       0.99     ← near-perfect
25%+          1.20       0.97     ← near-perfect
```

**How calibration fits into the pipeline:**

```
raw_model_prob  (softmax T=4.0)
      ↓
calibrate_probabilities()
  — if betting/calibration_model.pkl exists: isotonic map + within-race renorm
  — if file missing: passthrough (raw_model_prob = model_prob)
      ↓
model_prob  (used for edge = model_prob − market_implied_prob)
```

**When to re-fit:**
- After any change to scoring weights (`weight_*` in config.py)
- After any feature engineering change that affects `model_score` or `raw_model_prob` (for example, adding neutral priors for low-sample history)
- After adding new runners to the database (significantly more data)
- After changing `prob_temperature` in config.py

Feature engineering changes matter just as much as weight changes here: if the raw inputs to `raw_model_prob` move, the calibration model is stale even when the filter thresholds are unchanged.

**Config key:**

| Key | Default | Meaning |
|-----|---------|---------|
| `calibration_model_path` | `None` (auto) | Override path to `.pkl`. `None` = look beside `calibration.py` |

---

## 5c — What to do when you load new race data

Calibration and finetune depend on each other in a specific order.
**Calibration must come before finetune** because finetune searches for the
best edge thresholds — and edge is computed from calibrated probabilities.
If the calibration model is stale, finetune will optimise against wrong probabilities.

### Recommended: automated cycle

`scripts/recalibrate_cycle.py` runs the full loop automatically — weights,
then calibration, then filters — repeating until ROI no longer improves:

```bash
# Dry run first (shows what would change, writes nothing)
python3 scripts/recalibrate_cycle.py --dry-run --max-iters 2

# Full run — apply best config to betting/config.py when done
python3 scripts/recalibrate_cycle.py --write

# With verbose per-key changes and tighter bet bar
python3 scripts/recalibrate_cycle.py --write --min-bets 50 --verbose
```

**Flags:**

| Flag | Default | Meaning |
|------|---------|---------|
| `--min-gain` | `0.001` (0.1%) | Stop if ROI improvement falls below this |
| `--max-iters` | `8` | Hard cap on iterations (safety) |
| `--min-bets` | `30` | Ignore filter combos with fewer bets than this |
| `--write` | off | Auto-write final config back to `betting/config.py` |
| `--verbose` | off | Print per-key changes at each iteration |

The script prints a per-iteration summary and the final config block. If you
do not pass `--write`, copy-paste the printed config into `betting/config.py` manually.

> ⚠️ **Overfitting warning:** a high ROI (e.g. 45%+) on a small number of bets
> (< 100) in backtest is likely overfit. The filter sweep can find edge cases
> that do not generalise. Always check the bet count alongside ROI. Use
> `--min-bets 50` or higher to reduce overfitting risk.

### Why this order matters

```
scoring weights
      ↓
model_score  →  raw_model_prob  (softmax)
                      ↓
              calibration model   ← fitted on (raw_model_prob, is_winner)
                      ↓
              model_prob  →  edge  →  filters  ← finetune sweeps these
```

Changing weights or feature engineering that alters `model_score`/`raw_model_prob`
invalidates the calibration model. Changing filters does not invalidate
calibration — only the edge thresholds and price/rank cutoffs change.

### Manual cycle (alternative to automated)

If you prefer to inspect each step:

```bash
# Step 1 — Find best weights
python3 scripts/finetune.py --mode weights
# → paste suggested weights into betting/config.py

# Step 2 — Re-fit calibration (weights/features changed → raw probs changed)
python3 scripts/fit_calibration.py

# Step 3 — Find best filter thresholds on updated calibrated probs
python3 scripts/finetune.py
# → paste suggested filters into betting/config.py
```

---

## 6 — Adjust the strategy

All tuneable parameters live in `betting/config.py`. Edit the values there and
re-run the backtest to evaluate the effect.

### 6a — Automated fine-tuning (grid search)

`scripts/finetune.py` sweeps every combination of the tunable parameters,
runs the shared backtest pipeline for each, and ranks results by ROI.

```bash
# Sweep filter thresholds only (fastest — default, ~5 seconds)
python3 scripts/finetune.py

# Sweep scoring weights only
python3 scripts/finetune.py --mode weights

# Sweep everything (filter × weight combos — ~50 seconds)
python3 scripts/finetune.py --mode full

# Raise the minimum-bets bar so ROI is more meaningful
python3 scripts/finetune.py --min-bets 50

# Show top 20, sort by total profit instead of ROI
python3 scripts/finetune.py --top 20 --sort-by profit

# Export all valid combinations to CSV
python3 scripts/finetune.py --output outputs/finetune/results.csv
```

**Output** — a deduplicated ranked table plus a ready-to-paste config block:

```
Loading backtest data from: database/race_reports.sqlite
Loaded 32,507 rows from 3,336 races

Pre-scoring dataset (once)…
Pre-computing coverage arrays…
Sweeping 2835 filter combinations (vectorized)…
  2835/2835 (100%)  kept=1425

Top 10 unique bet sets by roi (1425 combos → 475 distinct bet sets after deduplication):

   min_price  max_price  min_edge  max_model_rank  min_field_size  max_field_size  ...  total_bets  wins     roi  total_profit  duplicate_combos non_binding_params       confidence
0      2.000     15.000     0.080               2               7              14  ...          50    12  150.0%        75.000                 3          min_price  LOW (<100 bets)
1      2.000     15.000     0.080               2               6              14  ...          54    12  131.5%        71.000                 3          min_price  LOW (<100 bets)
...

⚠  245 of 475 results have <100 bets — ROI figures are unreliable at this sample size.

--- Best config (paste into betting/config.py) ---
    "min_price": 2.0,
    "max_price": 15.0,
    "min_field_size": 7,
    "max_field_size": 14,
    "min_edge": 0.08,
    "max_model_rank": 2,
    ...
# Note: min_price were non-binding (changing them didn't change the bet set)
---------------------------------------------------
```

**Understanding the output columns:**

| Column | Meaning |
|---|---|
| `duplicate_combos` | How many parameter combos produced this exact same set of bets |
| `non_binding_params` | Parameters that were irrelevant — changing them didn't change bets |
| `confidence` | `LOW (<100 bets)` flags statistically unreliable ROI figures |

**How the vectorized sweep works:**

The filter sweep is fast (~5 seconds for 2835 combos) because:
1. Data is loaded and fully scored **once** (features → scoring → probs → edges)
2. All 2835 filter combos are applied using **numpy array operations** — no
   DataFrame copies, no per-combo groupby
3. `min_field_size`/`max_field_size` are applied directly from the raw column,
   so they correctly change the bet set (earlier versions had a bug where they
   were ignored)

The script only includes combinations that produce at least `--min-bets` bets
(default: 30) so low-sample flukes don't dominate the ranking.

> **Warning:** Grid-search on historical data will overfit. Treat the best
> result as a hypothesis to validate on a hold-out period, not a guaranteed
> forward edge.

### 6b — Manual parameter tuning

All tuneable parameters live in `betting/config.py`. Edit the values there and
re-run the backtest to evaluate the effect.

### Filter thresholds

| Key | Default | Meaning |
|---|---|---|
| `min_price` | `2.0` | Minimum live price to consider |
| `max_price` | `12.0` | Maximum live price to consider |
| `min_field_size` | `6` | Smallest field accepted |
| `max_field_size` | `14` | Largest field accepted |
| `min_edge` | `0.05` | Edge must be ≥ 5 percentage points |
| `max_model_rank` | `2` | Only top-2 model-ranked runners per race |
| `exclude_race_if_price_coverage_below` | `0.80` | Drop race if < 80% of runners have a price |
| `min_recent_form_count` | `2` | Runners need ≥ 2 recent runs |

### Fibonacci staking

Fibonacci staking is a progressive win-betting system for handling losing and winning runs without changing the selection logic. You start at level 0 for a 1-unit stake, move forward one step after each loss, and move back after a win according to the configured variant. In this pipeline the sequence is capped at level 10 (144 units) to limit liability during long losing streaks.

**Fibonacci sequence used**

`1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144`

**Variants**

| Variant | After win |
|---|---|
| `two_back` (default) | Move 2 levels back |
| `one_back` | Move 1 level back |
| `reset` | Return to level 0 |

**Config keys**

```python
"staking_mode": "flat",    # flat | fibonacci
"fib_variant":  "two_back",  # two_back | one_back | reset
"fib_base_unit": 1.0,        # 1 unit = 1 base stake
"fib_max_level": 10,         # cap at 144 units
```

- `staking_mode` selects whether stake assignment stays flat or uses Fibonacci progression.
- `fib_variant` controls how far the level moves back after a win.
- `fib_base_unit` scales every Fibonacci number into a real stake amount.
- `fib_max_level` limits the progression so it cannot grow past level 10 / 144 units.

**Live use — `inspect_race.py`**

By default, the advisor starts at level 0 when there is no prior settled-bet history to replay.

```bash
python3 scripts/inspect_race.py <race_id>
python3 scripts/inspect_race.py <race_id> --fib-level 3
```

In the `FIBONACCI STAKE ADVISOR` section:
- `Level` is the current progression step.
- `Sequence` is the Fibonacci multiplier at that level.
- `Stake` is `fib_base_unit × sequence`.
- `On WIN` shows the next level/stake if the current bet wins.
- `On LOSS` shows the next level/stake if the current bet loses.

**Backtest comparison**

```bash
python3 scripts/run_backtest.py
python3 scripts/run_backtest.py --staking fibonacci
```

- `python3 scripts/run_backtest.py` always prints both flat and Fibonacci ROI so you can compare staking systems on the same bet set.
- `python3 scripts/run_backtest.py --staking fibonacci` makes Fibonacci the primary staking mode used for the main report/export path.
- Fibonacci total staked will usually be much higher than flat because stakes grow during losing streaks, so raw profit is not directly comparable — compare ROI.

> **Caution:** Fibonacci is a progressive staking system. During losing runs the stake grows quickly. The `fib_max_level` cap (default 144 units) is there to stop runaway stake size, but it does not remove risk — use it with discipline.

### Scoring weights (must sum to 1.0)

| Key | Default | Component |
|---|---|---|
| `weight_speed_rating` | `0.25` | Condition-aware speed rating (wet/dry) |
| `weight_recent_form` | `0.20` | Recent run form score (50% aggregate stats + 50% position quality across the last 6 runs) |
| `weight_suitability` | `0.15` | Distance, track, and going suitability |
| `weight_connections` | `0.05` | Horse–jockey pairing win history |
| `weight_market_sanity` | `0.10` | Inverse price — tie-breaker only |
| `weight_margin` | `0.10` | Recent beaten margins; smaller is better (1.0 = winner, 0 = beaten by ≥20 lengths) |
| `weight_freshness` | `0.10` | Fitness timing — peaks at ~21 days since last run; uses first/second-up rates |
| `weight_class` | `0.05` | Career class quality from prize money, place rate, and overall win rate |
| `prob_temperature` | `4.0` | Softmax sharpening factor. Higher = more separation between runners. `T=1` gives nearly flat probabilities; `T=4` gives ~4–5x ratio top-to-bottom. |

The three new weights add depth to the model:
- **`weight_margin`** scores recent beaten margins with **smaller being better**: avg of the last 3 runs (80%) plus best recent run (20%). Caps are 8 lengths for avg and 4 lengths for best — anything at or beyond the cap scores 0. Missing data defaults to 8 lengths avg, 4 lengths best. Winners (negative margin) score 1.0.
- **`weight_freshness`** captures fitness timing: a Gaussian curve peaks at 21 days since last run. First-up and second-up win rates adjust for horses returning from breaks.
- **`weight_class`** uses career prize money (log-scaled), place percentage, and career win rate to estimate the quality level a horse competes at.

**Neutral prior for unknown history**

When a runner has no recorded history for a stat (for example no jockey-horse starts or no runs at the distance), the model uses a neutral prior instead of treating zero starts as proven failure:
- Jockey-horse connection: `0.25` prior
- Distance/track suitability: `0.30` prior
- First-up/second-up freshness rate: `0.20` prior

The blend formula is `score = observed_rate × weight + prior × (1 − weight)`, where `weight = min(starts / 3, 1.0)`. At 3+ starts, the observed rate fully dominates.

### Price derivation

The live price is computed in `betting/db.py` using:

```sql
COALESCE(NULLIF(fluc2, 0), NULLIF(fluc1, 0), NULLIF(open_price, 0))
```

`sp_starting_price` is loaded as a reference column (`price_vs_sp` diagnostic
field in edge output) but **never enters this COALESCE**.

---

## 7 — Understanding the pipeline

Both backtest and live modes share the same pipeline in `betting/backtest.py`:

```
load_race_runners()          ← SQL query, live_price derivation
      │
build_features()             ← speed rating, form, suitability, connections
      │
score_runners()              ← weighted composite score, model rank
      │
assign_probabilities()       ← softmax model probs, overround-normalised market probs
      │
calibrate_probabilities()    ← passthrough in v1
      │
calculate_edges()            ← edge = model_prob − market_implied_prob
      │
assign_stakes()              ← flat 1.0 unit per candidate
      │
apply_filters()              ← 8-stage filter stack (see below)
```

### Filter stack (applied in order)

| Step | Filter | What it removes |
|---|---|---|
| 1 | `filter_no_live_price` | Runners with no valid price |
| 2 | `filter_price_coverage` | Races where < 80% of runners are priced |
| 3 | `filter_price_range` | Runners outside 2.0–12.0 price range |
| 4 | `filter_field_size` | Races with < 6 or > 14 runners |
| 5 | `filter_min_edge` | Runners with edge < 5pp |
| 6 | `filter_model_rank` | Runners ranked 3rd or lower by the model |
| 7 | `filter_sparse_form` | Runners with fewer than 2 recent runs |
| 8 | `filter_one_per_race` | All but the highest-edge runner per race |

Each step logs `rows_before → rows_after` to stdout.

After filters, the backtest continues:

```
settle_bets()   ← profit = (live_price − 1) × stake for wins, −stake for losses
```

---

## 8 — Query modes

| Mode | SQL discriminator | Excludes |
|---|---|---|
| `backtest` | `status = 'finished'` | `result_code = 'V'` (late scratched) |
| `live` | `status = 'no_result'` | `result_code = 'V'` |

The `status` column is used (not `result_code`) because some `no_result` rows
carry a non-blank `result_code` due to async data updates.

---

## 9 — Output files reference

| Location | Script | Contents |
|---|---|---|
| `outputs/backtests/backtest_*.csv` | `run_backtest.py` | Settled bets, one row per runner |
| `outputs/backtests/breakdown_*.csv` | `run_backtest.py` | 8 breakdowns concatenated |
| `outputs/backtests/summary_*.txt` | `run_backtest.py` | Human-readable text summary |
| `outputs/reports/` | `export_backtest_report.py` | Same 3 files, separate directory |
| `outputs/live/` | `list_live_bets.py --output` | Live candidates CSV (optional) |

---

## 10 — Important caveats (v1)

- **ROI is likely overfitted.** Scoring weights are hand-set, not calibrated
  against data. The 46% backtest ROI on 114 bets over ~3 months is not a
  reliable forward estimate.
- **SP is never used for selection.** If you see any live price that looks like
  SP in the output it is coming from `fluc2`, `fluc1`, or `open_price` — check
  the `price_quality` column.
- **Only flat and Fibonacci win staking are implemented.** Kelly sizing,
  each-way bets, and manual overrides are not implemented in v1.
- **One bet per race.** `allow_multiple_bets_per_race = False` by default.
  The highest-edge candidate survives.
