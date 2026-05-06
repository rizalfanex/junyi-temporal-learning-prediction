# Bayesian Transformer-Based Temporal Learning Behavior Modeling for Student Performance Prediction

**A reproducible educational data mining project using Junyi Academy online learning activity logs.**

This repository studies temporal student learning behavior from the **Junyi Academy Online Learning Activity Dataset** and predicts **next-attempt correctness** from ordered online exercise logs. The project is strictly based on educational learning logs: it does **not** use computer vision, camera monitoring, face recognition, object trajectories, bounding boxes, person re-identification, or physical spatiotemporal tracking.

The unit of analysis is a student's chronological sequence of online problem attempts.

---

## Table Of Contents

- [Abstract](#abstract)
- [Research Scope](#research-scope)
- [Repository Highlights](#repository-highlights)
- [Dataset Schema](#dataset-schema)
- [Experimental Pipeline](#experimental-pipeline)
- [Leakage Prevention](#leakage-prevention)
- [Feature Engineering](#feature-engineering)
- [Models](#models)
- [Main Results](#main-results)
- [Ablation Study](#ablation-study)
- [Bayesian Uncertainty And Risk Groups](#bayesian-uncertainty-and-risk-groups)
- [Figures](#figures)
- [Reproducibility](#reproducibility)
- [Repository Structure](#repository-structure)
- [Academic Framing](#academic-framing)
- [Limitations](#limitations)

---

## Abstract

Online learning platforms generate dense temporal traces of student behavior. This project models Junyi Academy exercise attempt logs as student-wise chronological learning sequences and predicts whether a future problem attempt will be correct. The study builds leakage-aware temporal features, compares interpretable and strong tabular baselines against sequence neural networks, and estimates predictive uncertainty using Monte Carlo Dropout.

On the full processed dataset of more than **16 million** attempt-level records, the strongest model by ROC-AUC was a Logistic Regression baseline using engineered temporal behavior and exercise/content context features. Sequence models, including GRU and Transformer Encoder architectures, provided temporal modeling baselines and enabled uncertainty-aware risk grouping through MC Dropout. The safest academic claim is therefore comparative: **behavioral history and exercise context strongly predict next-attempt correctness, while Bayesian sequence modeling supports uncertainty-aware educational interpretation.**

---

## Research Scope

| Item | Description |
|---|---|
| Dataset | Junyi Academy Online Learning Activity Dataset |
| Core file | `Log_Problem.csv` |
| Metadata files | `Info_Content.csv`, `Info_UserData.csv` |
| Prediction task | Binary next-attempt correctness |
| Student identifier | `uuid` |
| Exercise/content identifier | `ucid` |
| Problem identifier | `upid` |
| Timestamp | `timestamp_TW` |
| Correctness label | `is_correct` |
| Modeling unit | Student-wise ordered learning attempt sequence |
| Study type | Offline educational data mining |

---

## Repository Highlights

- Full-data preprocessing for **16,217,311** Junyi problem-attempt logs.
- Student-wise chronological sorting by `uuid`, timestamp, and stable raw row id.
- Leakage-aware feature engineering with shifted rolling, cumulative, streak, topic-history, and previous-attempt features.
- Temporal train/validation/test split within each student.
- Baselines: Majority, previous correctness, Logistic Regression, Random Forest, HistGradientBoosting.
- Sequence models: PyTorch GRU and Transformer Encoder.
- Bayesian approximation: Monte Carlo Dropout during inference.
- Risk grouping: high/medium/low predicted correctness crossed with high/low uncertainty.
- Full feature-family ablation study for journal-style analysis.
- Academic reports, tables, and figures are included under `docs/`.

---

## Dataset Schema

The local dataset inventory verified the following files:

| File | Rows | Role |
|---|---:|---|
| `Log_Problem.csv` | 16,217,311 | Main online exercise attempt log |
| `Info_Content.csv` | 1,330 | Exercise/content metadata |
| `Info_UserData.csv` | 72,758 | Optional user metadata |

Key schema alignment:

| Source | Verified columns used |
|---|---|
| `Log_Problem.csv` | `timestamp_TW`, `uuid`, `ucid`, `upid`, `problem_number`, `exercise_problem_repeat_session`, `is_correct`, `total_sec_taken`, `total_attempt_cnt`, `used_hint_cnt`, `is_hint_used`, `is_downgrade`, `is_upgrade`, `level` |
| `Info_Content.csv` | `ucid`, `content_pretty_name`, `content_kind`, `difficulty`, `subject`, `learning_stage`, `level1_id`, `level2_id`, `level3_id`, `level4_id` |
| `Info_UserData.csv` | Available for optional reporting, not used as the central predictive focus |

Content metadata is joined through:

```text
Log_Problem.ucid = Info_Content.ucid
```

User metadata is intentionally optional and not central to the study, because the project emphasizes learning behavior rather than demographic profiling.

---

## Experimental Pipeline

```text
Raw Junyi CSVs
    -> dataset scan and schema validation
    -> cleaning and type standardization
    -> student-wise chronological sorting
    -> temporal feature engineering
    -> temporal train/validation/test split
    -> tabular baseline training
    -> sliding-window sequence construction
    -> GRU and Transformer training
    -> MC Dropout uncertainty inference
    -> risk-group analysis
    -> journal-style reports, tables, and figures
```

Final processed split:

| Split | Rows |
|---|---:|
| Train | 11,247,540 |
| Validation | 2,431,344 |
| Test | 2,465,669 |

Additional final preprocessing summary:

| Quantity | Value |
|---|---:|
| Raw attempts | 16,217,311 |
| Output rows after temporal filtering | 16,144,553 |
| Students | 72,758 |
| Exercises | 1,326 |
| Problems | 25,785 |
| Date range | 2018-08-01 to 2019-08-01 |
| Correct attempts | 11,412,558 |
| Incorrect attempts | 4,804,753 |

---

## Leakage Prevention

| Risk | Control |
|---|---|
| Random row splitting leaks future student behavior | Splits are temporal within each `uuid` |
| Current correctness leaking into rolling features | Rolling features are computed from shifted correctness history |
| Current attempt hint/time fields leaking into tabular prediction | Tabular baselines use previous-attempt versions such as `prev_total_sec_taken` |
| Outcome-derived difficulty leaking test labels | Exercise/topic difficulty proxies are estimated from training split only |
| Training row target encoding leaking itself | Training difficulty proxies use leave-one-out values |
| Ambiguous ordering for equal timestamps | Sorting uses timestamp plus stable `raw_row_id` |

---

## Feature Engineering

Feature families are educationally meaningful and derived from online learning behavior:

| Family | Examples | Interpretation |
|---|---|---|
| Prior correctness history | `student_prev_accuracy`, `prev_is_correct`, `hist_correct_count` | Student mastery signal before the target attempt |
| Rolling performance | `rolling_accuracy_5`, `rolling_accuracy_10` | Recent learning momentum |
| Streaks | `consecutive_correct_count`, `consecutive_wrong_count` | Local persistence or struggle |
| Temporal rhythm | `time_gap_sec`, `daily_activity_count_prior`, `session_attempt_index` | Learning spacing and session behavior |
| Attempt behavior | `prev_total_sec_taken`, `prev_used_hint_cnt`, `student_exercise_attempt_count_prior` | Prior effort, hint use, and repeated practice |
| Topic history | `topic_attempt_count_prior`, `topic_prev_accuracy` | Topic-level prior experience |
| Content context | `difficulty`, `learning_stage`, `level2_id`, `level3_id`, `level4_id` | Curriculum/content structure |
| Problem identity | `ucid`, `upid` | Exercise/problem-specific context |
| Difficulty proxies | `exercise_incorrect_rate_train`, `topic_incorrect_rate_train` | Train-only empirical difficulty estimates |

---

## Models

| Model | Role |
|---|---|
| Majority baseline | Naive class-frequency reference |
| Previous correctness | Simple temporal heuristic |
| Logistic Regression | Interpretable high-performing tabular baseline |
| Random Forest | Nonlinear tabular baseline |
| HistGradientBoosting | Additional tree-based tabular model |
| GRU | Conventional neural temporal sequence baseline |
| Transformer Encoder | Attention-based sequence model aligned with the project title |
| MC Dropout | Bayesian approximation for predictive uncertainty |

The Transformer uses categorical embeddings, numerical feature projection, positional embeddings, Transformer Encoder blocks, dropout, and a binary classification head.

---

## Main Results

Final test performance:

| Rank | Model | ROC-AUC | PR-AUC | F1 | Brier |
|---:|---|---:|---:|---:|---:|
| 1 | Logistic Regression | 0.801 | 0.894 | 0.766 | 0.189 |
| 2 | Random Forest | 0.771 | 0.879 | 0.772 | 0.188 |
| 3 | GRU | 0.759 | 0.869 | 0.752 | 0.201 |
| 4 | Transformer | 0.747 | 0.859 | 0.698 | 0.223 |
| 5 | HistGradientBoosting | 0.711 | 0.840 | 0.824 | 0.199 |
| 6 | Previous Correct | 0.598 | 0.736 | 0.751 | 0.344 |
| 7 | Majority | 0.500 | 0.691 | 0.817 | 0.214 |

Interpretation:

- Logistic Regression performed best by ROC-AUC and PR-AUC.
- GRU was the strongest sequence model in this run.
- Transformer results support temporal sequence modeling but should not be overclaimed as the best predictive model.
- The project is strongest as a comparative educational data mining study with uncertainty-aware sequence modeling.

Detailed machine-readable outputs:

- [`docs/tables/journal_model_ranking.csv`](docs/tables/journal_model_ranking.csv)
- [`docs/tables/model_performance_comparison.csv`](docs/tables/model_performance_comparison.csv)
- [`docs/tables/journal_metric_confidence_intervals.csv`](docs/tables/journal_metric_confidence_intervals.csv)

---

## Ablation Study

The full feature-family ablation retrained Logistic Regression after removing one feature family at a time.

| Ablation | ROC-AUC | ROC-AUC Drop | PR-AUC Drop | Brier Increase |
|---|---:|---:|---:|---:|
| Full feature set | 0.801 | 0.000 | 0.000 | 0.000 |
| Behavior-history-only | 0.707 | 0.094 | 0.068 | 0.031 |
| No problem/exercise identity | 0.756 | 0.046 | 0.030 | 0.019 |
| No topic history | 0.799 | 0.002 | 0.001 | 0.002 |
| No previous attempt behavior | 0.800 | 0.002 | 0.001 | 0.000 |
| No temporal activity | 0.801 | 0.001 | 0.001 | -0.001 |
| No rolling accuracy | 0.801 | 0.000 | 0.000 | -0.000 |
| No difficulty proxies | 0.801 | 0.000 | 0.000 | 0.000 |
| No content metadata | 0.802 | -0.001 | -0.000 | -0.000 |

Main ablation conclusion:

> Problem/exercise identity and broader content context carry substantial signal for the strongest tabular model. Individual behavioral feature families have smaller marginal effects in Logistic Regression, but they remain central for educational interpretation and sequence modeling.

Detailed ablation outputs:

- [`docs/tables/ablation_test_deltas.csv`](docs/tables/ablation_test_deltas.csv)
- [`docs/reports/ablation_study_report.md`](docs/reports/ablation_study_report.md)

---

## Bayesian Uncertainty And Risk Groups

MC Dropout keeps dropout active at inference time and performs repeated stochastic forward passes. The mean predicted probability is used as the final prediction, while predictive standard deviation summarizes uncertainty.

Final risk-group analysis:

| Risk group | Count | Observed correctness | Mean predicted correctness | Mean predictive std |
|---|---:|---:|---:|---:|
| High-risk confident | 1,549,224 | 0.462 | 0.245 | 0.019 |
| High-risk uncertain | 130,360 | 0.553 | 0.328 | 0.042 |
| Medium-risk confident | 1,430,155 | 0.741 | 0.546 | 0.023 |
| Medium-risk uncertain | 571,568 | 0.754 | 0.562 | 0.045 |
| Low-risk confident | 953,093 | 0.909 | 0.825 | 0.022 |
| Low-risk uncertain | 281,190 | 0.892 | 0.785 | 0.045 |

Educational interpretation:

- **High-risk confident** predictions are the clearest candidates for prioritized support.
- **High-risk uncertain** predictions should be interpreted more cautiously, because the model expects difficulty but is less confident.
- This is an offline analysis and should not be treated as an automated intervention system.

Detailed output:

- [`docs/tables/risk_group_analysis.csv`](docs/tables/risk_group_analysis.csv)
- [`docs/reports/risk_group_analysis.md`](docs/reports/risk_group_analysis.md)

---

## Figures

### Dataset Overview

<p align="center">
  <img src="docs/figures/correctness_distribution.png" width="32%" alt="Correctness distribution">
  <img src="docs/figures/activity_over_time.png" width="32%" alt="Activity over time">
  <img src="docs/figures/student_attempt_distribution.png" width="32%" alt="Student attempt distribution">
</p>

<p align="center">
  <img src="docs/figures/exercise_difficulty_distribution.png" width="45%" alt="Exercise difficulty distribution">
</p>

### Model Comparison

<p align="center">
  <img src="docs/figures/model_comparison_roc_auc.png" width="45%" alt="Model comparison ROC-AUC">
  <img src="docs/figures/model_comparison_pr_auc.png" width="45%" alt="Model comparison PR-AUC">
</p>

<p align="center">
  <img src="docs/figures/model_comparison_brier_score.png" width="45%" alt="Model comparison Brier score">
</p>

### Curves And Calibration

<p align="center">
  <img src="docs/figures/logistic_regression_roc_curve.png" width="30%" alt="Logistic regression ROC curve">
  <img src="docs/figures/logistic_regression_pr_curve.png" width="30%" alt="Logistic regression PR curve">
  <img src="docs/figures/logistic_regression_calibration.png" width="30%" alt="Logistic regression calibration">
</p>

<p align="center">
  <img src="docs/figures/gru_sequence_roc_curve.png" width="45%" alt="GRU ROC curve">
  <img src="docs/figures/transformer_sequence_roc_curve.png" width="45%" alt="Transformer ROC curve">
</p>

### Ablation And Uncertainty

<p align="center">
  <img src="docs/figures/logistic_regression_roc_auc_drop_vs_full.png" width="45%" alt="Ablation ROC-AUC drop">
  <img src="docs/figures/logistic_regression_pr_auc_drop_vs_full.png" width="45%" alt="Ablation PR-AUC drop">
</p>

<p align="center">
  <img src="docs/figures/uncertainty_distribution.png" width="45%" alt="Uncertainty distribution">
</p>

---

## Reproducibility

### 1. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Place Dataset Files

The raw Junyi CSV files are intentionally not tracked in Git because they are large. Place them locally as:

```text
dataset/
  Log_Problem.csv
  Info_Content.csv
  Info_UserData.csv
```

### 3. Full Journal Run

```bash
python src/data/scan_dataset.py --config configs/gx10_journal.json --full
python src/features/build_features.py --config configs/gx10_journal.json
python src/models/train_baseline.py --config configs/gx10_journal.json
python src/models/train_transformer.py --config configs/gx10_journal.json --model-type transformer
python src/models/train_transformer.py --config configs/gx10_journal.json --model-type gru
python src/evaluation/evaluate_models.py --config configs/gx10_journal.json
python src/evaluation/ablation_study.py --config configs/gx10_ablation.json
python src/evaluation/journal_analysis.py --results-dir hasil_project_junyi_journal
```

### 4. Quick Smoke Tests

```bash
python src/features/build_features.py --config configs/preflight.json
python src/models/train_baseline.py --config configs/preflight.json
python src/models/train_transformer.py --config configs/preflight.json --model-type transformer
python src/models/train_transformer.py --config configs/preflight.json --model-type gru
python src/evaluation/evaluate_models.py --config configs/preflight.json
python src/evaluation/ablation_study.py --config configs/preflight_ablation.json
```

---

## Repository Structure

```text
configs/                 Experiment configurations
data/raw/README.md       Dataset placement note
docs/figures/            Curated GitHub-safe figures
docs/reports/            Journal-style summaries
docs/tables/             Curated CSV tables
src/common/              Configuration, logging, environment helpers
src/data/                Dataset scanning
src/features/            Cleaning and temporal feature engineering
src/models/              Baseline, GRU, Transformer, MC Dropout training
src/evaluation/          Metrics, plots, journal analysis, ablation study
requirements.txt         Python dependencies
```

Large files such as raw datasets, processed features, predictions, checkpoints, and local experiment folders are excluded from Git through `.gitignore`.

---

## Academic Framing

Recommended claim:

> Offline temporal learning logs from Junyi Academy can predict next-attempt correctness with strong discrimination. Tabular behavioral-history and content-context baselines achieved the best predictive performance in the current run, while Bayesian sequence models enabled uncertainty-aware risk grouping for cautious educational interpretation.

Claims to avoid:

- Do not claim the Transformer is the best-performing model in the current experiment.
- Do not claim deployment readiness.
- Do not claim causal intervention effects.
- Do not frame this as image monitoring, physical tracking, or student surveillance.

---

## Limitations

- The study is offline and historical; it does not validate real-time intervention outcomes.
- The strongest model in the current run is tabular, not Transformer-based.
- Exercise/problem identity is highly predictive, which may reflect item-specific context as much as general learner state.
- MC Dropout provides practical approximate uncertainty, not a full Bayesian posterior.
- User metadata is not the main focus and should be handled carefully in future work.

---

## Project Status

**Complete experimental pipeline.** The remaining work is manuscript writing, optional external model comparisons, and any additional robustness checks desired before submission.
