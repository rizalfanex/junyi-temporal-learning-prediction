from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create journal-oriented post-hoc summaries from completed experiment outputs.")
    parser.add_argument("--results-dir", default="hasil_project_junyi")
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def require_pandas() -> Any:
    if pd is None:
        raise ImportError("pandas is required. Install dependencies with `python -m pip install -r requirements.txt`.")
    return pd


def normal_z(alpha: float) -> float:
    if abs(alpha - 0.05) < 1e-9:
        return 1.959963984540054
    try:
        from scipy.stats import norm

        return float(norm.ppf(1.0 - alpha / 2.0))
    except Exception:
        return 1.959963984540054


def wilson_interval(successes: float, total: float, z: float) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    p = successes / total
    denom = 1.0 + z**2 / total
    center = (p + z**2 / (2.0 * total)) / denom
    half = z * np.sqrt((p * (1.0 - p) + z**2 / (4.0 * total)) / total) / denom
    return max(0.0, center - half), min(1.0, center + half)


def auc_hanley_mcneil_interval(auc: float, n_pos: float, n_neg: float, z: float) -> tuple[float, float, float]:
    if n_pos <= 1 or n_neg <= 1 or not np.isfinite(auc):
        return np.nan, np.nan, np.nan
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc**2 / (1.0 + auc)
    variance = (auc * (1.0 - auc) + (n_pos - 1.0) * (q1 - auc**2) + (n_neg - 1.0) * (q2 - auc**2)) / (n_pos * n_neg)
    se = float(np.sqrt(max(variance, 0.0)))
    return se, max(0.0, auc - z * se), min(1.0, auc + z * se)


def add_confidence_intervals(metrics: Any, alpha: float) -> Any:
    z = normal_z(alpha)
    rows = []
    for row in metrics.to_dict("records"):
        tn = float(row.get("tn", np.nan))
        fp = float(row.get("fp", np.nan))
        fn = float(row.get("fn", np.nan))
        tp = float(row.get("tp", np.nan))
        total = tn + fp + fn + tp
        n_pos = tp + fn
        n_neg = tn + fp
        acc_low, acc_high = wilson_interval(tp + tn, total, z)
        precision_low, precision_high = wilson_interval(tp, tp + fp, z)
        recall_low, recall_high = wilson_interval(tp, tp + fn, z)
        auc_se, auc_low, auc_high = auc_hanley_mcneil_interval(float(row.get("roc_auc", np.nan)), n_pos, n_neg, z)
        rows.append(
            {
                "model": row.get("model"),
                "split": row.get("split"),
                "source_file": row.get("source_file", ""),
                "n": int(total) if np.isfinite(total) else row.get("n"),
                "accuracy": row.get("accuracy"),
                "accuracy_ci_low": acc_low,
                "accuracy_ci_high": acc_high,
                "precision": row.get("precision"),
                "precision_ci_low": precision_low,
                "precision_ci_high": precision_high,
                "recall": row.get("recall"),
                "recall_ci_low": recall_low,
                "recall_ci_high": recall_high,
                "f1": row.get("f1"),
                "roc_auc": row.get("roc_auc"),
                "roc_auc_se_hanley_mcneil": auc_se,
                "roc_auc_ci_low": auc_low,
                "roc_auc_ci_high": auc_high,
                "pr_auc": row.get("pr_auc"),
                "brier_score": row.get("brier_score"),
            }
        )
    return pd.DataFrame(rows)


def metric_rows_for_latest_test(metrics: Any) -> Any:
    test = metrics[metrics["split"].eq("test")].copy()
    if "source_file" not in test.columns:
        test["source_file"] = ""
    return test.sort_values(["roc_auc", "pr_auc"], ascending=[False, False]).reset_index(drop=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def submission_manifest(results_dir: Path) -> list[dict[str, Any]]:
    keep_patterns = {
        "reports": "include",
        "outputs/figures": "include",
        "outputs/metrics/model_performance_comparison.csv": "include",
        "outputs/metrics/baseline_metrics.csv": "include",
        "outputs/metrics/transformer_sequence_metrics.csv": "include",
        "outputs/metrics/gru_sequence_metrics.csv": "include",
        "outputs/metrics/risk_group_analysis.csv": "include",
        "outputs/metrics/journal_metric_confidence_intervals.csv": "include",
        "outputs/metrics/journal_model_ranking.csv": "include",
        "outputs/ablation": "include",
        "outputs/models": "optional_large",
        "processed_data": "exclude_or_archive",
        "outputs/metrics/baseline_predictions.csv": "exclude_or_archive",
        "outputs/metrics/transformer_sequence_predictions.csv": "optional_large",
        "outputs/metrics/gru_sequence_predictions.csv": "optional_large",
    }
    rows = []
    for path in sorted(results_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(results_dir).as_posix()
        status = "supporting"
        for pattern, value in keep_patterns.items():
            if rel == pattern or rel.startswith(pattern.rstrip("/") + "/"):
                status = value
                break
        rows.append({"relative_path": rel, "size_bytes": path.stat().st_size, "submission_status": status})
    return rows


def feature_family(feature_name: str) -> str:
    name = feature_name
    if "__" in name:
        name = name.split("__", 1)[1]
    if name.startswith("upid"):
        return "problem_id"
    if name.startswith("ucid"):
        return "exercise_id"
    if name.startswith("level"):
        return "content_hierarchy"
    if "difficulty" in name:
        return "content_difficulty"
    if "learning_stage" in name:
        return "learning_stage"
    if "rolling_accuracy" in name:
        return "rolling_accuracy"
    if "prev_accuracy" in name or "prev_is_correct" in name or "hist_correct" in name:
        return "prior_correctness_history"
    if "attempt_count" in name or "attempt_index" in name or "hist_attempt" in name:
        return "attempt_volume_history"
    if "consecutive" in name:
        return "streak_history"
    if "time_gap" in name or "sec_taken" in name:
        return "time_behavior"
    if "hint" in name:
        return "hint_behavior"
    if "topic" in name:
        return "topic_history"
    if "session" in name or "daily" in name:
        return "activity_rhythm"
    return "other"


def write_feature_family_importance(metrics_dir: Path, feature_importance: Any | None) -> Any | None:
    if feature_importance is None or feature_importance.empty:
        return None
    frame = feature_importance.copy()
    frame["feature_family"] = frame["feature"].map(feature_family)
    grouped = (
        frame.groupby(["model", "feature_family"], as_index=False)
        .agg(total_importance=("importance", "sum"), mean_importance=("importance", "mean"), top_features=("feature", "count"))
        .sort_values(["model", "total_importance"], ascending=[True, False])
    )
    grouped.to_csv(metrics_dir / "journal_feature_family_importance.csv", index=False)
    return grouped


def write_report(
    path: Path,
    metrics: Any,
    ci: Any,
    risk: Any | None,
    feature_importance: Any | None,
    family_importance: Any | None,
    ablation_delta: Any | None,
) -> None:
    test = metric_rows_for_latest_test(metrics)
    best = test.iloc[0].to_dict()
    sequence = test[test["model"].isin(["transformer", "gru"])].copy()
    best_sequence = sequence.iloc[0].to_dict() if not sequence.empty else None
    transformer = test[test["model"].eq("transformer")]
    transformer_row = transformer.iloc[0].to_dict() if not transformer.empty else None

    lines = [
        "# Journal Readiness Report",
        "",
        "## Verdict",
        "",
        "The current full experiment is methodologically sound and strong enough for a serious academic project. For journal positioning, the safest claim is not that the Transformer is the best predictor, but that a Bayesian temporal sequence modeling pipeline was implemented and compared against strong behavioral baselines.",
        "",
        "## Main Test Results",
        "",
        "| Rank | Model | ROC-AUC | PR-AUC | F1 | Brier |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(test.to_dict("records"), start=1):
        lines.append(
            f"| {rank} | {row['model']} | {row['roc_auc']:.3f} | {row['pr_auc']:.3f} | {row['f1']:.3f} | {row['brier_score']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Recommended Framing",
            "",
            f"- Best overall test ROC-AUC: `{best['model']}` with `{best['roc_auc']:.3f}`.",
        ]
    )
    if best_sequence:
        lines.append(f"- Best sequence model: `{best_sequence['model']}` with test ROC-AUC `{best_sequence['roc_auc']:.3f}`.")
    if transformer_row:
        lines.append(f"- Transformer test ROC-AUC: `{transformer_row['roc_auc']:.3f}`. This supports temporal modeling, but it should not be overclaimed as the top-performing model in the current run.")
    lines.extend(
        [
            "- The strongest journal story is comparative: engineered learning-behavior features are highly predictive, sequence models provide temporal representations, and MC Dropout adds uncertainty-aware risk interpretation.",
            "",
            "## Confidence Intervals",
            "",
            "Approximate 95% intervals are written to `outputs/metrics/journal_metric_confidence_intervals.csv`. Accuracy, precision, and recall use Wilson intervals; ROC-AUC uses the Hanley-McNeil large-sample approximation.",
            "",
            "## Risk Analysis",
            "",
        ]
    )
    if risk is not None and not risk.empty:
        for row in risk.to_dict("records"):
            lines.append(
                f"- `{row['risk_group']}`: n={int(row['count'])}, observed correctness={row['observed_correctness']:.3f}, "
                f"mean predicted correctness={row['mean_predicted_correctness']:.3f}, uncertainty={row['mean_predictive_std']:.3f}."
            )
    else:
        lines.append("- Risk group file was not available.")

    lines.extend(["", "## Feature Interpretation", ""])
    if family_importance is not None and not family_importance.empty:
        for model, part in family_importance.groupby("model", sort=False):
            top_families = part.head(6)
            text = ", ".join(
                f"{row['feature_family']}={row['total_importance']:.3f}" for row in top_families.to_dict("records")
            )
            lines.append(f"- `{model}` strongest feature families: {text}.")
    elif feature_importance is not None and not feature_importance.empty:
        top = feature_importance.head(8)
        for row in top.to_dict("records"):
            lines.append(f"- `{row['model']}` rank {int(row['rank'])}: `{row['feature']}` importance={row['importance']:.4f}")
    else:
        lines.append("- Feature importance file was not available.")

    lines.extend(["", "## Ablation Study", ""])
    if ablation_delta is not None and not ablation_delta.empty:
        ablation_rows = ablation_delta[ablation_delta["ablation"].ne("full")].copy()
        if "roc_auc_drop_vs_full" in ablation_rows.columns:
            ablation_rows = ablation_rows.sort_values("roc_auc_drop_vs_full", ascending=False)
        for row in ablation_rows.head(8).to_dict("records"):
            lines.append(
                f"- `{row['ablation']}`: ROC-AUC drop={row['roc_auc_drop_vs_full']:.3f}, "
                f"PR-AUC drop={row['pr_auc_drop_vs_full']:.3f}, Brier increase={row['brier_score_increase_vs_full']:.3f}."
            )
        lines.append(
            "- Ablation supports the interpretation that problem/exercise identity and content context carry substantial signal, while individual engineered behavioral families produce smaller marginal changes in the Logistic Regression model."
        )
    else:
        lines.append("- Ablation study outputs were not available. Run `python src/evaluation/ablation_study.py --config configs/gx10_ablation.json`.")

    lines.extend(
        [
            "",
            "## Highest-Impact Improvements Before Submission",
            "",
            "1. Treat the current ablation study as the main feature-family evidence for the journal draft.",
            "2. Add LightGBM or XGBoost only if installation is available and extra comparison time is acceptable.",
            "3. Consider a short sequence-model ablation later, but keep it optional because the full tabular ablation already addresses feature-family contribution.",
            "4. Keep prediction CSVs and processed full features as archived artifacts, not as the main submission bundle, because they are very large.",
            "",
            "## Suggested Journal Claim",
            "",
            "Offline temporal learning logs from Junyi Academy can predict next-attempt correctness with strong discrimination. Tabular behavioral-history baselines achieved the best predictive performance in the current run, while Bayesian sequence models enabled uncertainty-aware risk grouping that can support cautious educational interpretation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    pandas = require_pandas()
    results_dir = Path(args.results_dir)
    metrics_dir = results_dir / "outputs" / "metrics"
    reports_dir = results_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics = pandas.read_csv(metrics_dir / "model_performance_comparison.csv")
    ranking = metric_rows_for_latest_test(metrics)
    ci = add_confidence_intervals(ranking, args.alpha)
    ranking.to_csv(metrics_dir / "journal_model_ranking.csv", index=False)
    ci.to_csv(metrics_dir / "journal_metric_confidence_intervals.csv", index=False)

    risk_path = metrics_dir / "risk_group_analysis.csv"
    risk = pandas.read_csv(risk_path) if risk_path.exists() else None
    importance_path = metrics_dir / "baseline_feature_importance.csv"
    feature_importance = pandas.read_csv(importance_path) if importance_path.exists() else None
    family_importance = write_feature_family_importance(metrics_dir, feature_importance)
    ablation_path = results_dir / "outputs" / "ablation" / "ablation_test_deltas.csv"
    ablation_delta = pandas.read_csv(ablation_path) if ablation_path.exists() else None
    write_csv(metrics_dir / "journal_submission_manifest.csv", submission_manifest(results_dir))
    write_report(reports_dir / "journal_readiness_report.md", metrics, ci, risk, feature_importance, family_importance, ablation_delta)


if __name__ == "__main__":
    main()
