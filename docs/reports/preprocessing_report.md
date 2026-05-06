# Preprocessing Report

The unit of analysis is a student's ordered online exercise attempt sequence.

## Counts

- raw_rows: 16217311
- content_rows: 1330
- rows_missing_content: 0
- rows_before_cleaning: 16217311
- rows_missing_required_fields: 0
- rows_invalid_correctness: 0
- rows_with_invalid_numeric_fields: 0
- rows_invalid_timestamp: 0
- exact_duplicate_attempt_rows: 0
- rows_with_negative_numeric_fields: 0
- rows_after_cleaning: 16217311
- unique_students_after_cleaning: 72758
- unique_exercises_after_cleaning: 1326
- unique_problems_after_cleaning: 25785
- correct_attempts_after_cleaning: 11412558
- incorrect_attempts_after_cleaning: 4804753
- date_min: 2018-08-01 07:45:00+00:00
- date_max: 2019-08-01 00:00:00+00:00
- mode: full
- output_rows: 16144553

## Temporal Splits

- train: 11247540
- test: 2465669
- val: 2431344

## Leakage Controls

- Rows are sorted within `uuid` by `timestamp_TW` plus stable raw row order.
- Behavioral history features use `shift`, `cumcount`, or prior cumulative counts.
- Current-attempt hint usage, solve time, and attempt count are retained for sequence history, but tabular baselines use only their previous-attempt versions.
- Outcome-derived exercise and topic difficulty proxies are intentionally deferred to model training, where they are estimated from the training split only with leave-one-out values for training rows.

Processed feature file: `/home/ucl/Documents/DSCI-project/hasil_project_junyi_journal/processed_data/attempt_features_full.csv`
