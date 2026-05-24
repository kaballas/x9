# Race Reports SQLite Database

This directory documents the SQLite database at:

```sh
database/race_reports.sqlite
```

The database contains one base table, `race_runners`, plus reporting views built on top of it. It is a runner-level horse-racing feature/reporting database: each row in `race_runners` represents one runner in one race, with race metadata, runner attributes, historical form features, prices, and result labels.

## Current Snapshot

Observed with `sqlite3 -readonly database/race_reports.sqlite`:

| Metric | Value |
| --- | ---: |
| Runner rows | 45,645 |
| Distinct races | 3,595 |
| Distinct competitions | 169 |
| First start time | `2026-02-14T03:55:00+00:00` |
| Last start time | `2026-05-20T08:40:00+00:00` |
| Feature schema version | `race_hf_v1` |

The database currently has WAL sidecar files:

```text
database/race_reports.sqlite-shm
database/race_reports.sqlite-wal
```

Keep those files with the main `.sqlite` file when copying a live or recently written database.

## Base Table

### `race_runners`

Runner-level table. Important column groups:

| Group | Columns |
| --- | --- |
| Schema/version | `feature_schema_version` |
| Race identity | `race_id`, `race_number`, `race_name`, `competition_id`, `competition_name`, `country` |
| Race conditions | `class_name`, `grade`, `tempo`, `distance_m`, `track_status`, `start_time_iso`, `field_size`, `active_field_size` |
| Race labels | `winner_index`, `is_trainable` |
| Runner identity | `selection_id`, `runner_number`, `runner_name`, `runner_country` |
| Connections/profile | `draw_number`, `jockey`, `trainer`, `trainer_location`, `weight_kg`, `age`, `sex`, `colour`, `sire`, `dam`, `blinkers` |
| Ratings and career stats | `speed_rating`, `dry_rating`, `wet_rating`, `win_percentage`, `place_percentage`, `prize_money`, `career_starts`, `career_wins`, `career_seconds`, `career_thirds` |
| Condition/distance/track stats | `good_starts`, `good_wins`, `soft_starts`, `soft_wins`, `heavy_starts`, `heavy_wins`, `distance_starts`, `distance_wins`, `track_starts`, `track_wins` |
| Fresh and jockey stats | `first_up_starts`, `first_up_wins`, `second_up_starts`, `second_up_wins`, `horse_jockey_starts`, `horse_jockey_wins` |
| Form summary | `last_six`, `form_fig`, `expected_settling_position` |
| Market prices | `open_price`, `fluc1`, `fluc2`, `sp_starting_price` |
| Recent runs | `recent_1_*` through `recent_6_*`, covering place, distance, weight, barrier, condition, starting price, margin, field size, date, track, jockey, class, and time |
| Derived recent-form features | `recent_runs_count`, `recent_wins`, `recent_places`, `recent_avg_place`, `recent_best_place`, `recent_avg_place_3`, `recent_avg_place_5`, `recent_win_rate_5`, `recent_top3_rate_5`, `recent_avg_margin`, `recent_best_margin`, `recent_avg_margin_3`, `recent_avg_starting_price`, `recent_same_distance_runs`, `recent_same_track_runs`, `recent_same_condition_runs`, `recent_days_since_last_run` |
| Results | `finish_place`, `result_code`, `status`, `runner_mask`, `rank_label`, `top3_mask`, `is_winner` |

Result/status values observed:

| Status | Result code | Rows |
| --- | --- | ---: |
| `finished` | `L` | 19,155 |
| `late_scratched` | `V` | 11,819 |
| `finished` | `P` | 10,013 |
| `finished` | `W` | 3,339 |
| `no_result` | `-` | 683 |
| `no_result` | `L` | 630 |

Interpretation from the data:

- `result_code = 'W'` marks winners.
- `result_code = 'P'` marks placed non-winners.
- `result_code = 'L'` marks unplaced runners.
- `result_code = 'V'` marks late scratched runners.
- `status = 'no_result'` marks races or runners without final results yet.

## Reporting Views

The database includes these views:

| View | Purpose |
| --- | --- |
| `race_runners_active_races` | Active/no-result race rows. |
| `race_runners_winner_rows` | Winner-only rows. |
| `race_runners_bad_favourite_candidates` | Favourite candidates that underperformed. |
| `race_runners_value_candidate_rows` | Value-candidate runner rows. |
| `race_runners_non_favourite_winner_profile` | Profiles of winners that were not favourites. |
| `race_runners_favourite_by_track_stats` | Favourite performance by track/competition. |
| `race_runners_favourite_by_track_distance_stats` | Favourite performance by track and distance. |
| `race_runners_favourite_track_distance_condition` | Favourite performance by track, distance, and condition. |
| `race_runners_market_rank_stats` | Performance by market rank. |
| `race_runners_market_rank_by_track_stats` | Market-rank performance by track/competition. |
| `race_runners_market_rank_by_race_type` | Market-rank performance by race type. |
| `race_runners_price_band_stats` | Performance grouped by price bands. |
| `race_runners_recent_form_band_stats` | Performance grouped by recent-form bands. |
| `race_runners_recent_form_market_profile` | Combined recent-form and market profile. |
| `race_runners_field_size_stats` | Performance grouped by field size. |
| `race_runners_weight_band_stats` | Performance grouped by carried weight bands. |
| `race_runners_class_grade_stats` | Performance grouped by class and grade. |
| `race_runners_track_draw_stats` | Draw performance by track. |
| `race_runners_best_draw_by_track` | Best draw bands by track. |
| `race_runners_draw_band_stats` | Performance grouped by draw bands. |
| `race_runners_draw_band_race_shape` | Draw bands by race-shape context. |
| `race_runners_draw_by_track_distance_condition_stats` | Draw performance by track, distance, and condition. |
| `race_runners_distance_specialist_rows` | Distance specialist runners. |
| `race_runners_track_specialist_rows` | Track specialist runners. |
| `race_runners_condition_specialist_rows` | Track-condition specialist runners. |
| `race_runners_fresh_profile_rows` | First-up/second-up profile rows. |
| `race_runners_jockey_stats` | Jockey performance stats. |
| `race_runners_trainer_stats` | Trainer performance stats. |
| `race_runners_trainer_jockey_stats` | Trainer/jockey combination stats. |
| `race_runners_trainer_jockey_track_stats` | Trainer/jockey combination stats by track. |
| `race_runners_sire_condition_distance_stats` | Sire performance by condition and distance. |
| `race_runners_settling_position_stats` | Expected settling-position performance. |
| `race_runners_tempo_settling_position_stats` | Settling-position performance by tempo. |
| `race_runners_track_race_number_profile` | Track and race-number profile. |
| `race_runners_track_race_number_distance_condition_profile` | Track/race-number profile with distance and condition. |
| `race_runners_race_number_competition_prize_summary` | Race-number/competition/prize summary. |
| `race_runners_race_number_competition_prize_ranked` | Ranked version of race-number/competition/prize profile. |

## Indexes

Indexes on `race_runners`:

```text
idx_race_runners_competition_race      (competition_id, race_number)
idx_race_runners_country_class         (country, class_name)
idx_race_runners_prices                (race_id, runner_mask, sp_starting_price)
idx_race_runners_race_id               (race_id)
idx_race_runners_race_selection        (race_id, selection_id)
idx_race_runners_result_code           (result_code)
idx_race_runners_start_time            (start_time_iso)
idx_race_runners_status                (status)
idx_race_runners_track_distance        (track_status, distance_m)
idx_race_runners_winner                (is_winner, top3_mask)
```

## Useful Commands

Open the database read-only:

```sh
sqlite3 -readonly database/race_reports.sqlite
```

List tables and views:

```sql
.tables
```

Inspect the base table schema:

```sql
PRAGMA table_info(race_runners);
```

Count races and runners:

```sql
SELECT
  COUNT(*) AS runner_rows,
  COUNT(DISTINCT race_id) AS races,
  COUNT(DISTINCT competition_name) AS competitions,
  MIN(start_time_iso) AS first_start,
  MAX(start_time_iso) AS last_start
FROM race_runners;
```

Show upcoming or unresolved runners:

```sql
SELECT
  race_id,
  race_number,
  competition_name,
  distance_m,
  track_status,
  start_time_iso,
  runner_number,
  runner_name
FROM race_runners
WHERE status = 'no_result'
ORDER BY start_time_iso, race_id, runner_number;
```

Show winners:

```sql
SELECT
  race_id,
  race_number,
  competition_name,
  start_time_iso,
  runner_number,
  runner_name,
  sp_starting_price
FROM race_runners_winner_rows
ORDER BY start_time_iso DESC
LIMIT 20;
```

Find favourite performance by track:

```sql
SELECT *
FROM race_runners_favourite_by_track_stats
ORDER BY favourite_starts DESC
LIMIT 20;
```

## Notes

- This README documents the database as observed in the current checkout. If the database is regenerated, rerun the snapshot queries above and update the counts/date range.
- No local script reference to `race_reports.sqlite` was found in this directory, so the database generation path is not documented here.
