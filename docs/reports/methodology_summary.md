# Methodology Summary

This project is an offline educational data mining study using historical Junyi Academy online learning logs.

## Data Unit

The unit of analysis is a student's ordered exercise-attempt sequence from `Log_Problem.csv`. Attempts are linked to exercise metadata from `Info_Content.csv` through `ucid`. User metadata from `Info_UserData.csv` is report-only by default and is not the focus of prediction.

## Prediction Target

The target is next-attempt correctness. For tabular baselines, each row's correctness is predicted from features available before that attempt. For neural sequence models, a sliding window of prior attempts predicts the following attempt's correctness.

## Leakage Prevention

- Student records are sorted by `uuid`, `timestamp_TW`, and a stable raw row id.
- Cumulative, rolling, streak, session, and topic-history features are shifted so the current correctness is not used to predict itself.
- Current-attempt solve time and hint usage are not used as tabular predictors for that same attempt.
- Outcome-derived exercise/topic difficulty proxies are estimated from the training split only.
- Splits are temporal within student rather than random row-level splits.

## Models

Baselines include majority correctness, previous-correctness heuristic, logistic regression, and random forest. The sequence model is a PyTorch Transformer Encoder with categorical embeddings, numerical temporal features, positional embeddings, dropout, and a binary classification head. A GRU baseline is available through the same training script.

## Bayesian Approximation

Monte Carlo Dropout keeps dropout active during inference and performs repeated stochastic forward passes. The mean predicted probability is the final prediction, while predictive standard deviation summarizes uncertainty.

## Educational Interpretation

Risk groups separate low predicted correctness from high uncertainty. The distinction between high-risk confident and high-risk uncertain predictions matters: the former suggests prioritized intervention, while the latter suggests collecting more learning evidence or using a cautious human review.

## Scope

The work does not use camera tracking, image monitoring, object trajectories, face recognition, or person re-identification. It should not be overclaimed as a deployed intervention system; it is an offline study of historical learning behavior.
