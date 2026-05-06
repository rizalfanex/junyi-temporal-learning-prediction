# Ablation Study Report

This experiment retrains the selected tabular baseline models after removing one feature family at a time.
The same student-wise temporal train/validation/test split from the processed Junyi learning logs is reused.

## Leakage Controls

- The target remains next-attempt correctness for each temporally ordered student attempt row.
- The existing temporal split is reused; rows are not randomly mixed across train, validation, and test.
- Rolling, cumulative, streak, topic-history, and previous-attempt features are already shifted in preprocessing.
- Exercise/topic difficulty proxies are recomputed inside this script from the training split only, with leave-one-out values for training rows.
- Validation and test rows receive only train-estimated difficulty values.

## Run Settings

- Models: logistic_regression
- Ablations: full, no_rolling_accuracy, no_temporal_activity, no_attempt_behavior, no_content_metadata, no_difficulty_proxies, no_topic_history, no_problem_identity, behavior_history_only
- Max train rows: full train split
- Max eval rows: full validation/test splits

## Test Metrics

| model | ablation | n | roc_auc | pr_auc | f1 | brier_score | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | no_content_metadata | 2465669 | 0.8021 | 0.8947 | 0.7665 | 0.1890 | 0.7114 |
| logistic_regression | full | 2465669 | 0.8014 | 0.8942 | 0.7664 | 0.1890 | 0.7111 |
| logistic_regression | no_difficulty_proxies | 2465669 | 0.8013 | 0.8941 | 0.7661 | 0.1891 | 0.7108 |
| logistic_regression | no_rolling_accuracy | 2465669 | 0.8012 | 0.8940 | 0.7658 | 0.1890 | 0.7105 |
| logistic_regression | no_temporal_activity | 2465669 | 0.8009 | 0.8937 | 0.7694 | 0.1880 | 0.7134 |
| logistic_regression | no_attempt_behavior | 2465669 | 0.7998 | 0.8934 | 0.7660 | 0.1893 | 0.7104 |
| logistic_regression | no_topic_history | 2465669 | 0.7991 | 0.8928 | 0.7625 | 0.1908 | 0.7074 |
| logistic_regression | no_problem_identity | 2465669 | 0.7555 | 0.8642 | 0.7337 | 0.2084 | 0.6741 |
| logistic_regression | behavior_history_only | 2465669 | 0.7074 | 0.8257 | 0.7319 | 0.2198 | 0.6601 |

## Drop Relative To Full Feature Set

| model | ablation | roc_auc_drop_vs_full | pr_auc_drop_vs_full | f1_drop_vs_full | brier_score_increase_vs_full | removed_feature_count |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | behavior_history_only | 0.0940 | 0.0685 | 0.0345 | 0.0308 | 15 |
| logistic_regression | no_problem_identity | 0.0459 | 0.0300 | 0.0327 | 0.0194 | 2 |
| logistic_regression | no_topic_history | 0.0023 | 0.0014 | 0.0039 | 0.0018 | 2 |
| logistic_regression | no_attempt_behavior | 0.0016 | 0.0008 | 0.0004 | 0.0003 | 5 |
| logistic_regression | no_temporal_activity | 0.0005 | 0.0005 | -0.0030 | -0.0011 | 3 |
| logistic_regression | no_rolling_accuracy | 0.0002 | 0.0002 | 0.0006 | -0.0000 | 2 |
| logistic_regression | no_difficulty_proxies | 0.0001 | 0.0001 | 0.0003 | 0.0001 | 2 |
| logistic_regression | no_content_metadata | -0.0007 | -0.0005 | -0.0001 | -0.0001 | 6 |

## Feature Sets

| ablation | removed_families | removed_features | numeric_feature_count | categorical_feature_count |
| --- | --- | --- | --- | --- |
| full | none | none | 24 | 7 |
| no_rolling_accuracy | rolling_accuracy | rolling_accuracy_10;rolling_accuracy_5 | 22 | 7 |
| no_temporal_activity | temporal_activity | daily_activity_count_prior;session_attempt_index;time_gap_sec | 21 | 7 |
| no_attempt_behavior | attempt_behavior | prev_is_hint_used;prev_total_attempt_cnt;prev_total_sec_taken;prev_used_hint_cnt;student_exercise_attempt_count_prior | 19 | 7 |
| no_content_metadata | content_metadata | content_difficulty_ordinal;difficulty;learning_stage;level2_id;level3_id;level4_id | 23 | 2 |
| no_difficulty_proxies | difficulty_proxies | exercise_incorrect_rate_train;topic_incorrect_rate_train | 22 | 7 |
| no_topic_history | topic_history | topic_attempt_count_prior;topic_prev_accuracy | 22 | 7 |
| no_problem_identity | problem_identity | ucid;upid | 24 | 5 |
| behavior_history_only | problem_identity;content_metadata;difficulty_proxies;topic_history;problem_context | content_difficulty_ordinal;difficulty;exercise_incorrect_rate_train;exercise_problem_repeat_session;learning_stage;level;level2_id;level3_id;level4_id;problem_number;topic_attempt_count_prior;topic_incorrect_rate_train;topic_prev_accuracy;ucid;upid | 16 | 0 |
