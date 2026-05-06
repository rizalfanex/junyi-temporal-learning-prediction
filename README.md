# Bayesian Transformer-Based Temporal Learning Behavior Modeling for Student Performance Prediction

<p align="center">
  <b>A leakage-aware educational data mining project using Junyi Academy online learning activity logs.</b>
</p>

<p align="center">
  <i>Temporal learning behavior modeling · Next-attempt correctness prediction · Comparative machine learning · Bayesian uncertainty-aware risk analysis</i>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-GRU%20%7C%20Transformer-red">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-Tabular%20Baselines-orange">
  <img alt="Dataset" src="https://img.shields.io/badge/Dataset-Junyi%20Academy-green">
  <img alt="Task" src="https://img.shields.io/badge/Task-Next--Attempt%20Correctness-purple">
  <img alt="Status" src="https://img.shields.io/badge/Status-Complete-brightgreen">
</p>

---

## Executive Summary

This repository presents a complete, reproducible **educational data mining** project for predicting student learning performance from large-scale online learning logs. The project uses the **Junyi Academy Online Learning Activity Dataset**, a public dataset released through Kaggle that contains more than 16 million exercise attempt logs from more than 72,000 students over approximately one year.

The task is formulated as **temporal next-attempt correctness prediction**. Given a student's prior online learning history, the system predicts whether the next problem attempt will be correct. This framing is directly aligned with learning analytics, educational data mining, student performance prediction, and knowledge tracing-adjacent research.

The project title is:

> **Bayesian Transformer-Based Temporal Learning Behavior Modeling for Student Performance Prediction**

The final experimental result should be interpreted carefully. Although the project includes GRU and Transformer sequence models, the best-performing model in the completed experiment is a leakage-aware **Logistic Regression** model using engineered temporal behavior and exercise/content context features. The sequence models remain important because they provide neural temporal baselines and enable **Bayesian uncertainty-aware risk grouping** through Monte Carlo Dropout.

The strongest academic interpretation is therefore:

> Junyi Academy online learning logs contain strong predictive signals for next-attempt correctness. Leakage-aware temporal feature engineering and content-context information provide strong predictive performance, while sequence models and Bayesian uncertainty estimation support risk-aware educational interpretation.

---

## Why This Project Matters

Online learning platforms capture detailed traces of how students interact with learning materials: what exercises they attempt, when they attempt them, whether they answer correctly, how long they spend, whether they use hints, and how their learning behavior evolves over time. These records are valuable because they can support data-driven learning analytics, early warning systems, adaptive practice, personalized learning pathways, and teacher-facing decision support.

However, educational log modeling is methodologically sensitive. A model can easily appear strong if future information leaks into training features. For example, random row-level splitting may allow a student's future behavior to influence training, and aggregate difficulty features can accidentally include the label of the same row being predicted. This repository explicitly addresses these concerns through temporal within-student splitting, shifted historical features, train-only difficulty estimates, and leave-one-out encodings for training aggregates.

This makes the project more than a simple machine learning exercise. It is an end-to-end example of how to build a **methodologically defensible educational data science pipeline**.

---

## Scope Clarification

This project is strictly based on **educational learning logs**.

It does **not** use:

- computer vision,
- camera monitoring,
- face recognition,
- object detection,
- bounding boxes,
- physical trajectories,
- person re-identification,
- classroom surveillance.

The unit of analysis is:

> **A student's chronological sequence of online problem attempts.**

The term **temporal** refers to the ordering of student learning events over time, not physical movement or camera-based tracking.

---

## Dataset Source

Dataset used in this project:

**Junyi Academy Online Learning Activity Dataset**  
Kaggle source:  
https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy

The dataset is released by **Junyi Academy Foundation**, a Taiwan-based nonprofit educational organization. The Kaggle dataset description states that the dataset contains over 16 million exercise attempt logs from more than 72,000 students across the period from 2018/08 to 2019/07.

### Original Dataset Citation

```bibtex
@article{JunyiOnlineLearningDataset,
  title={Junyi Academy Online Learning Activity Dataset: A large-scale public online learning activity dataset from elementary to senior high school students.},
  author={Pojen, Chen and Mingen, Hsieh and Tzuyang, Tsai},
  journal={Dataset available from https://www.kaggle.com/junyiacademy/learning-activity-public-dataset-by-junyi-academy},
  year={2020}
}
```

---

## Research Questions

| ID | Research Question | Why It Matters |
|---|---|---|
| RQ1 | Can prior online learning behavior predict next-attempt correctness? | Tests whether student learning logs contain useful temporal predictive signals. |
| RQ2 | Which feature families contribute most to prediction? | Identifies whether behavior history, problem identity, topic history, or content metadata carries the strongest signal. |
| RQ3 | How do sequence models compare with tabular baselines? | Evaluates whether GRU/Transformer architectures improve temporal representation learning over engineered features. |
| RQ4 | Can Bayesian uncertainty estimation improve educational interpretation? | Distinguishes confident risk predictions from uncertain predictions. |
| RQ5 | What type of risk grouping is suitable for cautious learning analytics? | Supports human-in-the-loop educational decision support rather than automated intervention. |

---

## Dataset Inventory

A full streaming scan was performed before modeling. This was done to avoid assuming the structure of the dataset before inspecting the actual CSV files.

| File | Rows | Role |
|---|---:|---|
| `Log_Problem.csv` | 16,217,311 | Main online exercise attempt log |
| `Info_Content.csv` | 1,330 | Exercise/content metadata |
| `Info_UserData.csv` | 72,758 | Optional user metadata |

Additional verified dataset statistics:

| Quantity | Value |
|---|---:|
| Raw attempt records | 16,217,311 |
| Processed temporal rows | 16,144,553 |
| Unique students | 72,758 |
| Unique exercises | 1,326 |
| Unique problems | 25,785 |
| Correct attempts | 11,412,558 |
| Incorrect attempts | 4,804,753 |
| Date range | 2018-08-01 to 2019-08-01 |

### Dataset Interpretation

The dataset is sufficiently large for full-scale data science experimentation. It contains millions of attempt-level learning events, which makes it suitable for temporal modeling, feature engineering, sequence modeling, and uncertainty-aware evaluation. The large number of students and exercises also makes the dataset appropriate for examining both student-level behavior and content-level difficulty signals.

---

## Dataset Schema Alignment

The project uses the actual Junyi schema after scanning the CSV files.

### Main Log Table: `Log_Problem.csv`

| Column | Meaning in This Project |
|---|---|
| `timestamp_TW` | Timestamp used for chronological ordering |
| `uuid` | Student identifier |
| `ucid` | Exercise/content identifier |
| `upid` | Problem identifier |
| `problem_number` | Problem order or index within content |
| `exercise_problem_repeat_session` | Repeat/session information |
| `is_correct` | Correctness label |
| `total_sec_taken` | Time spent on the attempt |
| `total_attempt_cnt` | Attempt count |
| `used_hint_cnt` | Number of hints used |
| `is_hint_used` | Whether hint was used |
| `is_downgrade` | Whether student downgraded |
| `is_upgrade` | Whether student upgraded |
| `level` | Exercise or mastery-related level field |

### Content Metadata: `Info_Content.csv`

| Column | Meaning in This Project |
|---|---|
| `ucid` | Join key with `Log_Problem.csv` |
| `content_pretty_name` | Human-readable content name |
| `content_kind` | Type of content |
| `difficulty` | Provided content difficulty |
| `subject` | Subject area |
| `learning_stage` | Learning stage |
| `level1_id` | Curriculum hierarchy level 1 |
| `level2_id` | Curriculum hierarchy level 2 |
| `level3_id` | Curriculum hierarchy level 3 |
| `level4_id` | Curriculum hierarchy level 4 |

### User Metadata: `Info_UserData.csv`

User metadata is available, but it is intentionally not the central predictive focus. The study emphasizes **learning behavior** and **content interaction patterns**, not demographic profiling.

---

## Verified Table Relationship

The main verified join is:

```text
Log_Problem.ucid = Info_Content.ucid
```

The dataset scan confirmed that logged content identifiers have matching content metadata. This allows the project to enrich attempt logs with curriculum/content information while maintaining schema validity.

---

## Problem Formulation

The task is formulated as **binary next-attempt correctness prediction**.

For each student, attempts are sorted chronologically:

```text
Attempt 1 -> Attempt 2 -> Attempt 3 -> ... -> Attempt t -> Attempt t+1
```

The model uses information available before the target attempt to estimate:

```text
P(is_correct at next attempt = 1 | prior learning history)
```

In practical terms:

| Component | Definition |
|---|---|
| Input | Historical learning behavior before the target attempt |
| Output | Probability that the target attempt will be correct |
| Label | `is_correct` of the next attempt |
| Task type | Binary classification |
| Study type | Offline educational data mining |

This problem formulation is appropriate for learning analytics because it asks whether prior learning behavior can predict future performance.

---

## Methodology Overview

The full experimental pipeline is:

```text
Raw Junyi CSV files
    -> dataset scan and schema validation
    -> data cleaning and type standardization
    -> content metadata joining
    -> student-wise chronological sorting
    -> leakage-aware temporal feature engineering
    -> temporal train/validation/test split
    -> baseline model training
    -> sliding-window sequence construction
    -> GRU and Transformer sequence modeling
    -> Monte Carlo Dropout uncertainty estimation
    -> model evaluation and comparison
    -> ablation study
    -> risk group analysis
    -> journal-style tables, figures, and reports
```

### Methodological Rationale

The pipeline is designed around a core principle:

> Each prediction must use only information that would have been available before the target attempt.

This is critical for educational data mining because model performance can be inflated if future attempts, test-set labels, or current-attempt outcomes are accidentally included in features.

---

## Leakage-Aware Experimental Design

Preventing data leakage is one of the central strengths of this repository.

| Leakage Risk | Prevention Strategy |
|---|---|
| Random row splitting leaks future student behavior | Temporal split within each `uuid` |
| Current correctness leaks into rolling statistics | Rolling features are computed from shifted correctness history |
| Current attempt time/hint fields leak outcome information | Tabular features use previous-attempt versions where appropriate |
| Validation/test outcomes leak into difficulty estimates | Exercise/topic difficulty proxies are estimated from training split only |
| Training aggregate encoding includes its own row label | Leave-one-out encoding is used for training rows |
| Equal timestamps create unstable order | Sorting uses timestamp plus stable `raw_row_id` |
| Target attempt appears inside model input | Sequence windows exclude the target attempt |

### Why This Matters

Without these controls, a model might appear to perform well because it indirectly sees the answer. The leakage-aware design makes the results more conservative, credible, and suitable for academic review.

---

## Data Splitting Strategy

The final processed data is split temporally within each student.

| Split | Rows |
|---|---:|
| Train | 11,247,540 |
| Validation | 2,431,344 |
| Test | 2,465,669 |

This strategy is more realistic than random row splitting because it evaluates whether earlier behavior can predict later behavior for the same student. It also reduces the risk of future information leaking into model training.

---

## Feature Engineering

The engineered features are designed to represent educationally meaningful learning behavior.

| Feature Family | Example Features | Educational Meaning |
|---|---|---|
| Prior correctness history | `student_prev_accuracy`, `prev_is_correct`, `hist_correct_count` | Student mastery signal before the target attempt |
| Rolling performance | `rolling_accuracy_5`, `rolling_accuracy_10` | Short-term learning momentum |
| Streak features | `consecutive_correct_count`, `consecutive_wrong_count` | Local persistence, struggle, or confidence |
| Temporal rhythm | `time_gap_sec`, `daily_activity_count_prior`, `session_attempt_index` | Learning spacing and study rhythm |
| Previous attempt behavior | `prev_total_sec_taken`, `prev_used_hint_cnt`, `student_exercise_attempt_count_prior` | Prior effort, hint use, and repeated practice |
| Topic history | `topic_attempt_count_prior`, `topic_prev_accuracy` | Prior topic-level experience |
| Content context | `difficulty`, `learning_stage`, `level2_id`, `level3_id`, `level4_id` | Curriculum/content structure |
| Problem identity | `ucid`, `upid` | Exercise/problem-specific context |
| Difficulty proxies | `exercise_incorrect_rate_train`, `topic_incorrect_rate_train` | Empirical item/topic difficulty estimated safely |

### Feature Engineering Interpretation

The feature design combines two types of signals:

1. **Learning behavior signals**, such as previous accuracy, time gaps, streaks, and hint-related history.
2. **Learning content signals**, such as exercise ID, problem ID, curriculum level, and difficulty.

The ablation study shows that content/problem identity contributes strongly to predictive performance, while behavioral features remain important for educational interpretation and sequence modeling.

---

## Modeling Strategy

The project compares multiple model families rather than relying on a single advanced method.

| Model | Purpose |
|---|---|
| Majority baseline | Naive reference based on the majority class |
| Previous correctness | Simple temporal heuristic |
| Logistic Regression | Interpretable and high-performing tabular baseline |
| Random Forest | Nonlinear tabular baseline |
| HistGradientBoosting | Additional tree-based baseline |
| GRU | Recurrent neural temporal baseline |
| Transformer Encoder | Attention-based temporal sequence model |
| MC Dropout | Bayesian approximation for uncertainty estimation |

### Why Include Simple Models?

In educational data mining, complex neural models are not automatically better. Strong engineered features can allow simpler models to perform very well. Including simple and complex models provides a fairer comparison and prevents overclaiming the Transformer architecture.

---

## Transformer Sequence Model

The Transformer model treats each student's learning history as a sequence.

```text
[Attempt t-k, ..., Attempt t-2, Attempt t-1] -> Predict Attempt t correctness
```

The architecture includes:

- categorical embeddings for exercise/problem/content identifiers,
- numerical feature projection,
- positional or sequence-order representation,
- Transformer Encoder blocks,
- dropout,
- binary classification head.

### Transformer Interpretation

The Transformer is included because online learning logs are naturally sequential. It can model interactions among prior attempts and use attention mechanisms to represent temporal dependencies. However, in the completed run, it was not the strongest predictive model. Its value in this project is as an attention-based temporal modeling baseline and as the model used for uncertainty-aware risk analysis.

---

## Bayesian Uncertainty Estimation

The Bayesian component is implemented using **Monte Carlo Dropout**.

During inference:

1. dropout remains active,
2. the model performs repeated stochastic forward passes,
3. the mean predicted probability becomes the final prediction,
4. the predictive standard deviation becomes the uncertainty score.

```text
MC predictions: p1, p2, p3, ..., pN

Final probability = mean(p1, p2, ..., pN)
Uncertainty score = standard deviation(p1, p2, ..., pN)
```

### Why Uncertainty Matters in Education

Educational predictions should not be treated as absolute decisions. A student predicted to be high-risk with low uncertainty is different from a student predicted to be high-risk with high uncertainty. Uncertainty-aware prediction supports more cautious, human-in-the-loop interpretation.

---

## Evaluation Metrics

The project uses several metrics because no single metric fully describes model quality.

| Metric | Purpose |
|---|---|
| ROC-AUC | Measures ranking ability across classification thresholds |
| PR-AUC | Evaluates precision-recall tradeoff under class imbalance |
| F1-score | Balances precision and recall at a selected threshold |
| Brier Score | Measures probability calibration error |
| Calibration Curve | Visualizes reliability of predicted probabilities |
| Confusion Matrix | Shows classification error patterns |
| Risk Group Observed Correctness | Tests whether predicted risk groups align with actual outcomes |

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

### Result Interpretation

The best-performing model by ROC-AUC and PR-AUC is **Logistic Regression**. This is an important and defensible result. It shows that carefully engineered temporal and content-context features can be highly predictive even with a simple linear model.

The sequence models remain valuable:

- **GRU** is the strongest neural sequence model in this experiment.
- **Transformer** provides an attention-based sequence modeling baseline.
- **MC Dropout** enables uncertainty-aware analysis beyond standard classification metrics.

The correct conclusion is not that the Transformer dominates all models. The correct conclusion is that **leakage-aware temporal feature engineering is highly effective**, and sequence models provide additional temporal and uncertainty-aware perspectives.

---

## Ablation Study

A feature-family ablation study was conducted using Logistic Regression.

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

### Ablation Interpretation

The ablation study shows that the model benefits strongly from problem/exercise identity and broader content context. When only behavior-history features are used, ROC-AUC drops from 0.801 to 0.707. Removing problem/exercise identity also causes a substantial ROC-AUC drop to 0.756.

This suggests that student performance prediction in this dataset is driven by both:

1. **What the student is attempting**, represented by problem, exercise, and content context.
2. **How the student has been learning**, represented by temporal behavior history.

The result is educationally meaningful because student success depends not only on the learner's prior behavior, but also on the difficulty and identity of the content being attempted.

---

## Risk Group Analysis

Monte Carlo Dropout uncertainty was used to build risk-aware prediction groups.

| Risk Group | Count | Observed Correctness | Mean Predicted Correctness | Mean Predictive Std |
|---|---:|---:|---:|---:|
| High-risk confident | 1,549,224 | 0.462 | 0.245 | 0.019 |
| High-risk uncertain | 130,360 | 0.553 | 0.328 | 0.042 |
| Medium-risk confident | 1,430,155 | 0.741 | 0.546 | 0.023 |
| Medium-risk uncertain | 571,568 | 0.754 | 0.562 | 0.045 |
| Low-risk confident | 953,093 | 0.909 | 0.825 | 0.022 |
| Low-risk uncertain | 281,190 | 0.892 | 0.785 | 0.045 |

### Educational Interpretation

The risk grouping is one of the most important educational outputs of the project.

- **High-risk confident** attempts are the clearest candidates for prioritized educational review.
- **High-risk uncertain** attempts should be interpreted cautiously because the model predicts difficulty but has higher uncertainty.
- **Low-risk confident** attempts correspond to strong expected performance.
- **Medium-risk groups** represent cases where support may depend on additional instructional context.

This analysis supports a human-in-the-loop perspective. The model should not automatically decide interventions. Instead, it can help identify where a teacher, tutor, or learning platform may want to pay closer attention.

---

# Figure Walkthrough and Research Narrative

The figures below are not only visual outputs; they form a reader journey from dataset understanding, to model comparison, to uncertainty-aware educational interpretation.

---

## 1. Dataset Overview Figures

### 1.1 Correctness Distribution

<p align="center">
  <img src="docs/figures/correctness_distribution.png" width="65%" alt="Correctness distribution">
</p>

**What this figure shows:**  
This figure summarizes the distribution of correct and incorrect attempts in the Junyi learning logs. It provides the first view of the classification target. Since correct attempts are more common than incorrect attempts, accuracy alone is not sufficient for evaluating model quality.

**Why it matters:**  
The class distribution explains why this project reports ROC-AUC, PR-AUC, F1-score, and Brier Score rather than relying only on accuracy. A majority baseline can achieve a deceptively high F1 or accuracy-like impression, but it does not provide meaningful discrimination between likely correct and likely incorrect attempts.

---

### 1.2 Activity Over Time

<p align="center">
  <img src="docs/figures/activity_over_time.png" width="75%" alt="Activity over time">
</p>

**What this figure shows:**  
This figure visualizes online learning activity across the dataset period. It helps verify that the dataset spans a full year and captures temporal variation in platform usage.

**Why it matters:**  
Temporal variation is central to the project. Student behavior changes over time, and the prediction task is designed to respect chronological order. This figure supports the decision to use temporal splits and sequence-based modeling rather than random row-level evaluation.

---

### 1.3 Student Attempt Distribution

<p align="center">
  <img src="docs/figures/student_attempt_distribution.png" width="75%" alt="Student attempt distribution">
</p>

**What this figure shows:**  
This figure shows how many attempts students contributed. In online learning datasets, student activity is usually highly uneven: some students attempt only a small number of problems, while others generate long learning histories.

**Why it matters:**  
This distribution motivates student-wise temporal modeling. Students with long histories provide richer signals for rolling accuracy, streaks, and sequence models, while students with shorter histories are more challenging. This also motivates uncertainty-aware prediction because models may be less confident when historical evidence is limited.

---

### 1.4 Exercise Difficulty Distribution

<p align="center">
  <img src="docs/figures/exercise_difficulty_distribution.png" width="65%" alt="Exercise difficulty distribution">
</p>

**What this figure shows:**  
This figure summarizes the empirical difficulty of exercises, estimated from incorrect rates or content-level difficulty-related signals.

**Why it matters:**  
Exercise difficulty is educationally important. A student's correctness does not depend only on ability or behavior; it also depends on the difficulty of the attempted item. This figure supports the use of exercise/content context and helps explain why removing problem/exercise identity in the ablation study substantially reduces ROC-AUC.

---

## 2. Model Comparison Figures

### 2.1 ROC-AUC Model Comparison

<p align="center">
  <img src="docs/figures/model_comparison_roc_auc.png" width="70%" alt="Model comparison ROC-AUC">
</p>

**What this figure shows:**  
This figure compares models based on ROC-AUC. ROC-AUC measures how well a model ranks correct attempts above incorrect attempts across thresholds.

**Why it matters:**  
The figure shows that Logistic Regression achieved the strongest ranking performance. This result is methodologically important because it demonstrates that strong leakage-aware feature engineering can outperform more complex neural models in this dataset.

**Key takeaway:**  
The project should be presented as a comparative educational data mining study, not as a claim that the Transformer is the best model.

---

### 2.2 PR-AUC Model Comparison

<p align="center">
  <img src="docs/figures/model_comparison_pr_auc.png" width="70%" alt="Model comparison PR-AUC">
</p>

**What this figure shows:**  
This figure compares models using Precision-Recall AUC. PR-AUC is useful when the class distribution is imbalanced or when the positive class is the main focus.

**Why it matters:**  
The PR-AUC result confirms the same general pattern as ROC-AUC: the engineered tabular model performs strongly. It also helps ensure that the ranking result is not an artifact of a single metric.

---

### 2.3 Brier Score Comparison

<p align="center">
  <img src="docs/figures/model_comparison_brier_score.png" width="70%" alt="Model comparison Brier score">
</p>

**What this figure shows:**  
This figure compares the probability calibration quality of different models. Lower Brier Score indicates better calibrated probability predictions.

**Why it matters:**  
Calibration is important in education because predicted probabilities may be interpreted as risk levels. A model with good ranking but poor calibration may still be problematic for decision support. This figure connects model evaluation to responsible educational interpretation.

---

## 3. Curves and Calibration Figures

### 3.1 Logistic Regression ROC Curve

<p align="center">
  <img src="docs/figures/logistic_regression_roc_curve.png" width="65%" alt="Logistic regression ROC curve">
</p>

**What this figure shows:**  
This curve shows the tradeoff between true positive rate and false positive rate for the strongest model.

**Why it matters:**  
It provides a visual explanation of the ROC-AUC value and demonstrates that the best tabular model has meaningful discriminative power beyond the majority or previous-correctness baselines.

---

### 3.2 Logistic Regression Precision-Recall Curve

<p align="center">
  <img src="docs/figures/logistic_regression_pr_curve.png" width="65%" alt="Logistic regression PR curve">
</p>

**What this figure shows:**  
This curve shows how precision and recall trade off under different thresholds.

**Why it matters:**  
In educational applications, threshold choice matters. A strict threshold may identify fewer students/attempts but with higher confidence, while a more permissive threshold may capture more possible risk cases but include more false alarms.

---

### 3.3 Logistic Regression Calibration Curve

<p align="center">
  <img src="docs/figures/logistic_regression_calibration.png" width="65%" alt="Logistic regression calibration">
</p>

**What this figure shows:**  
This figure compares predicted probabilities against observed correctness rates.

**Why it matters:**  
A calibrated model is more trustworthy when probabilities are interpreted as risk estimates. For example, among attempts predicted at around 0.80 correctness probability, the observed correctness rate should ideally be close to 0.80.

---

### 3.4 GRU Sequence ROC Curve

<p align="center">
  <img src="docs/figures/gru_sequence_roc_curve.png" width="65%" alt="GRU sequence ROC curve">
</p>

**What this figure shows:**  
This curve summarizes the discrimination performance of the GRU sequence model.

**Why it matters:**  
GRU is the strongest sequence model in the completed experiment. It provides evidence that neural temporal models can learn useful patterns from student attempt sequences, even though the best overall model remains tabular.

---

### 3.5 Transformer Sequence ROC Curve

<p align="center">
  <img src="docs/figures/transformer_sequence_roc_curve.png" width="65%" alt="Transformer sequence ROC curve">
</p>

**What this figure shows:**  
This curve summarizes the discrimination performance of the Transformer sequence model.

**Why it matters:**  
The Transformer is aligned with the project title and provides an attention-based temporal modeling approach. Its performance is competitive but not the best. This is an important research finding: architecture complexity alone does not guarantee superior performance when strong tabular features are available.

---

## 4. Ablation and Uncertainty Figures

### 4.1 ROC-AUC Drop by Feature Ablation

<p align="center">
  <img src="docs/figures/logistic_regression_roc_auc_drop_vs_full.png" width="75%" alt="Ablation ROC-AUC drop">
</p>

**What this figure shows:**  
This figure shows how much ROC-AUC decreases when specific feature families are removed.

**Why it matters:**  
The largest drop occurs when the model is restricted to behavior-history-only features or when problem/exercise identity is removed. This indicates that content context is essential for predicting student correctness in this dataset.

---

### 4.2 PR-AUC Drop by Feature Ablation

<p align="center">
  <img src="docs/figures/logistic_regression_pr_auc_drop_vs_full.png" width="75%" alt="Ablation PR-AUC drop">
</p>

**What this figure shows:**  
This figure shows how PR-AUC changes after removing feature families.

**Why it matters:**  
It confirms that feature-family importance is not limited to ROC-AUC. Problem/exercise identity and broader context remain important under precision-recall evaluation.

---

### 4.3 Uncertainty Distribution

<p align="center">
  <img src="docs/figures/uncertainty_distribution.png" width="70%" alt="Uncertainty distribution">
</p>

**What this figure shows:**  
This figure visualizes the distribution of predictive uncertainty estimated through MC Dropout.

**Why it matters:**  
Uncertainty distribution is central to the Bayesian component of the project. It shows that predictions are not all equally reliable. Some predictions have low uncertainty and may be more suitable for confident interpretation, while others have high uncertainty and require caution.

---

## Figure-Level Summary

Together, the figures support the full research story:

1. The dataset contains large-scale and temporally structured learning activity.
2. The prediction target is imbalanced enough to require multiple evaluation metrics.
3. Exercise/content context provides strong predictive signal.
4. Logistic Regression with leakage-aware features performs best overall.
5. GRU and Transformer provide useful sequence modeling baselines.
6. MC Dropout uncertainty enables risk-aware educational interpretation.
7. The project is strongest when framed as a comparative and methodologically careful educational data mining study.

---

## Output Artifacts

Important output files include:

| Path | Description |
|---|---|
| `docs/tables/journal_model_ranking.csv` | Final model ranking table |
| `docs/tables/model_performance_comparison.csv` | Full model comparison metrics |
| `docs/tables/journal_metric_confidence_intervals.csv` | Confidence interval summary |
| `docs/tables/ablation_test_deltas.csv` | Ablation study results |
| `docs/tables/risk_group_analysis.csv` | Risk group table |
| `docs/reports/ablation_study_report.md` | Detailed ablation interpretation |
| `docs/reports/risk_group_analysis.md` | Detailed risk-group interpretation |
| `docs/figures/` | GitHub-safe figures for README, presentation, and reporting |

---

## Reproducibility

### 1. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Place Dataset Files

Raw Junyi files are intentionally not tracked in Git because of size and dataset distribution constraints.

Place the files locally as:

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

### 4. Quick Smoke Test

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
configs/
  Experiment configurations for preflight, GX10 full run, journal run, and ablation run.

data/
  raw/README.md
  Dataset placement note. Raw CSV files are not tracked.

docs/
  figures/
    Curated GitHub-safe figures used in README, reports, and presentation.
  reports/
    Journal-style summaries and interpretation files.
  tables/
    Curated CSV result tables.

src/
  common/
    Configuration loading, environment reporting, and logging helpers.
  data/
    Dataset scanning and schema validation.
  features/
    Cleaning, joining, temporal sorting, and leakage-aware feature engineering.
  models/
    Baseline models, GRU, Transformer, training loops, and MC Dropout inference.
  evaluation/
    Metrics, plots, ablation study, journal tables, and risk analysis.

requirements.txt
  Python dependencies.

README.md
  Main project documentation.
```

---

## Academic Interpretation

The main conclusion is:

> Junyi Academy online learning logs contain strong predictive signals for next-attempt correctness. A leakage-aware temporal feature engineering pipeline with content-context information achieved strong predictive performance. The best-performing model in this run was Logistic Regression, while GRU and Transformer sequence models provided temporal neural baselines. Monte Carlo Dropout enabled uncertainty-aware risk grouping, allowing predictions to be interpreted more cautiously for educational decision support.

This framing is intentionally conservative and academically defensible.

---

## Claims to Avoid

The following claims should not be made:

- The Transformer is the best-performing model.
- The system is ready for real-time deployment.
- The model proves causal effects of student behavior on achievement.
- The model can replace teachers or educational experts.
- The model performs student surveillance.
- The project uses visual tracking or camera data.
- The risk groups should automatically trigger interventions without human review.

---

## Limitations

- This is an offline historical analysis, not a deployed intervention system.
- The strongest predictive model is tabular, not Transformer-based.
- Problem/exercise identity is highly predictive and may reflect item-specific effects.
- Monte Carlo Dropout is an approximate Bayesian method, not a full Bayesian posterior.
- User metadata is intentionally not emphasized to avoid overreliance on profile-based prediction.
- Predictive performance does not imply causal explanation.
- The project evaluates next-attempt correctness, not long-term educational outcomes.

---

## Future Work

Potential extensions include:

- comparison with Deep Knowledge Tracing and SAINT-like architectures,
- stronger student-level generalization experiments,
- cold-start student analysis,
- cold-start exercise analysis,
- calibration improvement using temperature scaling or isotonic regression,
- teacher-facing explanation dashboard,
- content-level diagnostic reports,
- intervention simulation under human-in-the-loop constraints,
- external comparison with other educational log datasets.

---

## Project Status

The project is complete as a reproducible educational data mining pipeline.

Completed components:

- full dataset scan,
- schema validation,
- content metadata join,
- leakage-aware feature engineering,
- temporal within-student split,
- baseline model training,
- GRU and Transformer sequence modeling,
- MC Dropout uncertainty estimation,
- ablation study,
- risk-group analysis,
- curated tables and figures,
- academic-style documentation.

Remaining optional work:

- manuscript writing,
- slide preparation,
- additional robustness checks,
- external benchmark comparison.

---

## Suggested One-Sentence Presentation Summary

> This project uses large-scale Junyi Academy online learning logs to predict next-attempt correctness through leakage-aware temporal feature engineering, comparative machine learning, GRU/Transformer sequence modeling, and Bayesian uncertainty-aware risk analysis.
