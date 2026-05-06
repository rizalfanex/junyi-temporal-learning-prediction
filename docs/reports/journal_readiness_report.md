# Journal Readiness Report

## Verdict

The current full experiment is methodologically sound and strong enough for a serious academic project. For journal positioning, the safest claim is not that the Transformer is the best predictor, but that a Bayesian temporal sequence modeling pipeline was implemented and compared against strong behavioral baselines.

## Main Test Results

| Rank | Model | ROC-AUC | PR-AUC | F1 | Brier |
|---:|---|---:|---:|---:|---:|
| 1 | logistic_regression | 0.801 | 0.894 | 0.766 | 0.189 |
| 2 | random_forest | 0.771 | 0.879 | 0.772 | 0.188 |
| 3 | gru | 0.759 | 0.869 | 0.752 | 0.201 |
| 4 | transformer | 0.747 | 0.859 | 0.698 | 0.223 |
| 5 | hist_gradient_boosting | 0.711 | 0.840 | 0.824 | 0.199 |
| 6 | previous_correct | 0.598 | 0.736 | 0.751 | 0.344 |
| 7 | majority | 0.500 | 0.691 | 0.817 | 0.214 |

## Recommended Framing

- Best overall test ROC-AUC: `logistic_regression` with `0.801`.
- Best sequence model: `gru` with test ROC-AUC `0.759`.
- Transformer test ROC-AUC: `0.747`. This supports temporal modeling, but it should not be overclaimed as the top-performing model in the current run.
- The strongest journal story is comparative: engineered learning-behavior features are highly predictive, sequence models provide temporal representations, and MC Dropout adds uncertainty-aware risk interpretation.

## Confidence Intervals

Approximate 95% intervals are written to `outputs/metrics/journal_metric_confidence_intervals.csv`. Accuracy, precision, and recall use Wilson intervals; ROC-AUC uses the Hanley-McNeil large-sample approximation.

## Risk Analysis

- `high_risk_confident`: n=1549224, observed correctness=0.462, mean predicted correctness=0.245, uncertainty=0.019.
- `high_risk_uncertain`: n=130360, observed correctness=0.553, mean predicted correctness=0.328, uncertainty=0.042.
- `low_risk_confident`: n=953093, observed correctness=0.909, mean predicted correctness=0.825, uncertainty=0.022.
- `low_risk_uncertain`: n=281190, observed correctness=0.892, mean predicted correctness=0.785, uncertainty=0.045.
- `medium_risk_confident`: n=1430155, observed correctness=0.741, mean predicted correctness=0.546, uncertainty=0.023.
- `medium_risk_uncertain`: n=571568, observed correctness=0.754, mean predicted correctness=0.562, uncertainty=0.045.

## Feature Interpretation

- `logistic_regression` strongest feature families: problem_id=248.033.
- `random_forest` strongest feature families: prior_correctness_history=0.191, topic_history=0.183, other=0.180, content_hierarchy=0.145, rolling_accuracy=0.095, attempt_volume_history=0.078.

## Ablation Study

- `behavior_history_only`: ROC-AUC drop=0.094, PR-AUC drop=0.068, Brier increase=0.031.
- `no_problem_identity`: ROC-AUC drop=0.046, PR-AUC drop=0.030, Brier increase=0.019.
- `no_topic_history`: ROC-AUC drop=0.002, PR-AUC drop=0.001, Brier increase=0.002.
- `no_attempt_behavior`: ROC-AUC drop=0.002, PR-AUC drop=0.001, Brier increase=0.000.
- `no_temporal_activity`: ROC-AUC drop=0.001, PR-AUC drop=0.001, Brier increase=-0.001.
- `no_rolling_accuracy`: ROC-AUC drop=0.000, PR-AUC drop=0.000, Brier increase=-0.000.
- `no_difficulty_proxies`: ROC-AUC drop=0.000, PR-AUC drop=0.000, Brier increase=0.000.
- `no_content_metadata`: ROC-AUC drop=-0.001, PR-AUC drop=-0.000, Brier increase=-0.000.
- Ablation supports the interpretation that problem/exercise identity and content context carry substantial signal, while individual engineered behavioral families produce smaller marginal changes in the Logistic Regression model.

## Highest-Impact Improvements Before Submission

1. Treat the current ablation study as the main feature-family evidence for the journal draft.
2. Add LightGBM or XGBoost only if installation is available and extra comparison time is acceptable.
3. Consider a short sequence-model ablation later, but keep it optional because the full tabular ablation already addresses feature-family contribution.
4. Keep prediction CSVs and processed full features as archived artifacts, not as the main submission bundle, because they are very large.

## Suggested Journal Claim

Offline temporal learning logs from Junyi Academy can predict next-attempt correctness with strong discrimination. Tabular behavioral-history baselines achieved the best predictive performance in the current run, while Bayesian sequence models enabled uncertainty-aware risk grouping that can support cautious educational interpretation.
