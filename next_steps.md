# Next Steps Handover

## Current State

The betting pipeline is now internally consistent and the latest `inspect_race.py` output is numerically aligned:

- effective weights are printed with `raw_total` and normalization state
- component contributions match the displayed multiplications
- longshot EV warnings are flagged in the watchlist
- `python3 scripts/inspect_race.py --verbose 10498355` passes and shows no bet qualifies
- `pytest -q` passes
- `python3 scripts/run_backtest.py` still runs cleanly

The current active config is the baseline in `betting/config.py`:

- `weight_market_sanity = 0.0092`
- `weight_steam = 0.0800`
- total weights sum to `1.0`

## What Was Verified

Recent targeted sweeps only varied `weight_market_sanity` and `weight_steam`, with all other weights rescaled proportionally to keep the total weight at `1.0`.

Files written by the sweep:

- `outputs/sweeps/market_steam_summary.csv`
- `outputs/sweeps/market_steam_calibration.csv`
- `outputs/sweeps/market_steam_calibration_delta_vs_baseline.csv`
- `outputs/sweeps/market_steam_price_band.csv`
- `outputs/sweeps/market_steam_price_band_delta_vs_baseline.csv`

Key results from that sweep:

- Best ROI combo: `mkt=0.080|steam=0.100`
  - ROI `15.11%`
  - Profit `68.6`
  - Bets `454`
  - Calibration WMAE worsened to `1.406 pp`
- Best calibration combo: `mkt=0.010|steam=0.040`
  - Calibration WMAE `1.042 pp`
  - ROI `-1.32%`
- Baseline:
  - ROI `11.05%`
  - Profit `61.2`
  - Bets `554`
  - Calibration WMAE `1.119 pp`

## Recommendation

Do not promote the aggressive ROI combo directly. The evidence so far says:

- higher `Mkt` improves ROI on this backtest slice
- but calibration gets worse
- bet count drops materially

The safer next step is a walk-forward validation on the top two candidates:

- baseline: `mkt=0.010|steam=0.080`
- aggressive ROI: `mkt=0.080|steam=0.100`

## Next Agent Tasks

1. Run a walk-forward or train/holdout comparison on the top two `Mkt/Steam` combinations.
2. Compare ROI, strike rate, and calibration by `model_prob` bucket and `price_band`.
3. Decide whether any config promotion is justified before changing `betting/config.py`.
4. If promotion is not justified, keep baseline config and narrow the search around:
   - `weight_market_sanity` in `0.01` to `0.03`
   - `weight_steam` in `0.08` to `0.10`

## Useful Entry Points

- `scripts/run_backtest.py`
- `scripts/inspect_race.py`
- `scripts/finetune.py`
- `scripts/recalibrate_cycle.py`
- `betting/reporting.py`
- `betting/config.py`

## Notes

- `sp_starting_price` should remain a reference/result field, not the live betting price.
- Live price selection should continue to use `COALESCE(fluc2, fluc1, open_price)`.
- The watchlist should keep flagging longshot EV as unreliable when `live_price > 50`.
