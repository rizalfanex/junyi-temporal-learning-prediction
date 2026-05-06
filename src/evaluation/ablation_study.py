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

from src.common.config import configured_path, load_config
from src.common.logging_utils import setup_logger
from src.evaluation.metrics import classification_metrics
from src.models.train_baseline import (
    add_train_only_difficulty_proxies,
    build_hist_gradient_boosting_pipeline,
    build_logistic_pipeline,
    build_random_forest_pipeline,
    sample_split,
)


FEATURE_FAMILIES: dict[str, dict[str, list[str]]] = {
    "student_history": {
        "numeric": [
            "hist_attempt_count",
            "hist_correct_count",
            "student_prev_accuracy",
            "prev_is_correct",
            "consecutive_wrong_count",
            "consecutive_correct_count",
        ],
        "categorical": [],
    },
    "rolling_accuracy": {
        "numeric": ["rolling_accuracy_5", "rolling_accuracy_10"],
        "categorical": [],
    },
    "temporal_activity": {
        "numeric": ["time_gap_sec", "daily_activity_count_prior", "session_attempt_index"],
        "categorical": [],
    },
    "attempt_behavior": {
        "numeric": [
            "prev_total_sec_taken",
            "prev_total_attempt_cnt",
            "prev_used_hint_cnt",
            "prev_is_hint_used",
            "student_exercise_attempt_count_prior",
        ],
        "categorical": [],
    },
    "problem_context": {
        "numeric": ["problem_number", "exercise_problem_repeat_session", "level"],
        "categorical": [],
    },
    "topic_history": {
        "numeric": ["topic_attempt_count_prior", "topic_prev_accuracy"],
        "categorical": [],
    },
    "content_metadata": {
        "numeric": ["content_difficulty_ordinal"],
        "categorical": ["difficulty", "learning_stage", "level2_id", "level3_id", "level4_id"],
    },
    "problem_identity": {
        "numeric": [],
        "categorical": ["ucid", "upid"],
    },
    "difficulty_proxies": {
        "numeric": ["exercise_incorrect_rate_train", "topic_incorrect_rate_train"],
        "categorical": [],
    },
}


ABLATION_REMOVALS: dict[str, list[str]] = {
    "full": [],
    "no_rolling_accuracy": ["rolling_accuracy"],
    "no_temporal_activity": ["temporal_activity"],
    "no_attempt_behavior": ["attempt_behavior"],
    "no_content_metadata": ["content_metadata"],
    "no_difficulty_proxies": ["difficulty_proxies"],
    "no_topic_history": ["topic_history"],
    "no_problem_identity": ["problem_identity"],
    "behavior_history_only": [
        "problem_identity",
        "content_metadata",
        "difficulty_proxies",
        "topic_history",
        "problem_context",
    ],
}

DEFAULT_ABLATIONS = list(ABLATION_REMOVALS)
SUPPORTED_MODELS = {"logistic_regression", "random_forest", "hist_gradient_boosting"}
HIGHER_IS_BETTER = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
LOWER_IS_BETTER = ["brier_score"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage-aware feature ablation experiments.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--features", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--models", nargs="+", default=None, choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--ablations", nargs="+", default=None, choices=sorted(ABLATION_REMOVALS))
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Training cap per ablation. Use 0 for full training split.",
    )
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=None,
        help="Validation/test cap per ablation. Use 0 for full validation/test splits.",
    )
    return parser.parse_args()


def require_pandas() -> Any:
    if pd is None:
        raise ImportError(
            "pandas is required for ablation study. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        )
    return pd


def configured_optional_int(args_value: int | None, config_value: Any) -> int | None:
    if args_value is not None:
        return None if args_value == 0 else int(args_value)
    if config_value is None:
        return None
    return int(config_value)


def configured_output_dir(config: dict[str, Any], args_output_dir: str | None) -> Path:
    if args_output_dir:
        path = Path(args_output_dir)
        return path if path.is_absolute() else ROOT / path
    configured = config.get("ablation", {}).get("output_dir")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else ROOT / path
    metrics_dir = configured_path(config, "metrics_dir")
    return metrics_dir.parent / "ablation"


def read_header(path: Path) -> list[str]:
    pandas = require_pandas()
    return list(pandas.read_csv(path, nrows=0).columns)


def feature_columns_from_config(config: dict[str, Any], available_columns: set[str]) -> tuple[list[str], list[str]]:
    numeric_cols = [
        col
        for col in config["features"]["baseline_numeric_columns"]
        if col in available_columns or col in FEATURE_FAMILIES["difficulty_proxies"]["numeric"]
    ]
    categorical_cols = [col for col in config["features"]["categorical_columns"] if col in available_columns]
    return numeric_cols, categorical_cols


def read_feature_frame(path: Path, config: dict[str, Any], logger: Any) -> Any:
    pandas = require_pandas()
    header = read_header(path)
    available = set(header)
    numeric_cols, categorical_cols = feature_columns_from_config(config, available)
    target_col = config["dataset"]["correct_col"]
    required = {"split", target_col, "ucid", "level3_id", "level4_id"}
    raw_numeric_cols = [col for col in numeric_cols if col in available]
    usecols = sorted((set(raw_numeric_cols) | set(categorical_cols) | required) & available)
    dtype: dict[str, str] = {}
    for col in usecols:
        if col in categorical_cols or col in {"split", "ucid", "level3_id", "level4_id"}:
            dtype[col] = "category"
    if target_col in usecols:
        dtype[target_col] = "int8"
    for col in raw_numeric_cols:
        if col in usecols:
            dtype[col] = "float32"

    logger.info("Reading ablation columns from %s", path)
    logger.info("Columns loaded: %s", len(usecols))
    df = pandas.read_csv(path, usecols=usecols, dtype=dtype)
    df = add_train_only_difficulty_proxies(df)
    return df


def removed_columns(families: list[str]) -> tuple[set[str], set[str]]:
    numeric: set[str] = set()
    categorical: set[str] = set()
    for family in families:
        spec = FEATURE_FAMILIES[family]
        numeric.update(spec["numeric"])
        categorical.update(spec["categorical"])
    return numeric, categorical


def ablation_feature_set(
    ablation_name: str,
    all_numeric_cols: list[str],
    all_categorical_cols: list[str],
    df_columns: set[str],
) -> tuple[list[str], list[str], list[str]]:
    families = ABLATION_REMOVALS[ablation_name]
    removed_numeric, removed_categorical = removed_columns(families)
    numeric_cols = [col for col in all_numeric_cols if col in df_columns and col not in removed_numeric]
    categorical_cols = [col for col in all_categorical_cols if col in df_columns and col not in removed_categorical]
    removed = sorted((set(all_numeric_cols) & removed_numeric) | (set(all_categorical_cols) & removed_categorical))
    return numeric_cols, categorical_cols, removed


def build_estimator(model_name: str, config: dict[str, Any], numeric_cols: list[str], categorical_cols: list[str], seed: int) -> Any:
    if model_name == "logistic_regression":
        return build_logistic_pipeline(numeric_cols, categorical_cols)
    if model_name == "random_forest":
        return build_random_forest_pipeline(
            numeric_cols,
            categorical_cols,
            n_estimators=int(config["baseline"]["random_forest_estimators"]),
            seed=seed,
        )
    if model_name == "hist_gradient_boosting":
        return build_hist_gradient_boosting_pipeline(
            numeric_cols,
            categorical_cols,
            max_iter=int(config["baseline"]["hist_gradient_boosting_max_iter"]),
            learning_rate=float(config["baseline"]["hist_gradient_boosting_learning_rate"]),
            seed=seed,
        )
    raise ValueError(f"Unsupported ablation model: {model_name}")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_delta_table(metrics: Any) -> Any:
    pandas = require_pandas()
    rows: list[dict[str, Any]] = []
    test_metrics = metrics[metrics["split"].eq("test")].copy()
    for model_name, model_part in test_metrics.groupby("model", sort=False):
        full = model_part[model_part["ablation"].eq("full")]
        if full.empty:
            continue
        full_row = full.iloc[0]
        for _, row in model_part.iterrows():
            out = {
                "model": model_name,
                "ablation": row["ablation"],
                "removed_families": row["removed_families"],
                "removed_feature_count": row["removed_feature_count"],
                "numeric_feature_count": row["numeric_feature_count"],
                "categorical_feature_count": row["categorical_feature_count"],
            }
            for metric in HIGHER_IS_BETTER:
                if metric in row and metric in full_row:
                    out[f"{metric}_value"] = row[metric]
                    out[f"{metric}_delta_vs_full"] = row[metric] - full_row[metric]
                    out[f"{metric}_drop_vs_full"] = full_row[metric] - row[metric]
            for metric in LOWER_IS_BETTER:
                if metric in row and metric in full_row:
                    out[f"{metric}_value"] = row[metric]
                    out[f"{metric}_delta_vs_full"] = row[metric] - full_row[metric]
                    out[f"{metric}_increase_vs_full"] = row[metric] - full_row[metric]
            rows.append(out)
    delta = pandas.DataFrame(rows)
    if not delta.empty and "roc_auc_drop_vs_full" in delta.columns:
        delta = delta.sort_values(["model", "roc_auc_drop_vs_full"], ascending=[True, False])
    return delta


def save_ablation_plots(delta: Any, output_dir: Path, logger: Any) -> None:
    if delta.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed; skipping ablation plots")
        return

    for metric_name, ylabel in [
        ("roc_auc_drop_vs_full", "ROC-AUC drop vs full"),
        ("pr_auc_drop_vs_full", "PR-AUC drop vs full"),
        ("f1_drop_vs_full", "F1 drop vs full"),
        ("brier_score_increase_vs_full", "Brier increase vs full"),
    ]:
        if metric_name not in delta.columns:
            continue
        for model_name, part in delta.groupby("model", sort=False):
            part = part[part["ablation"].ne("full")].sort_values(metric_name, ascending=False)
            if part.empty:
                continue
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.bar(part["ablation"], part[metric_name], color="#5f6f52")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(f"{model_name} Ablation: {ylabel}")
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            safe_model = str(model_name).replace("/", "_")
            fig.savefig(output_dir / f"{safe_model}_{metric_name}.png", dpi=160)
            plt.close(fig)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    if not rows:
        return ["No rows available."]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    path: Path,
    metrics: Any,
    delta: Any,
    feature_rows: list[dict[str, Any]],
    models: list[str],
    ablations: list[str],
    max_train_rows: int | None,
    max_eval_rows: int | None,
) -> None:
    test = metrics[metrics["split"].eq("test")].copy()
    test = test.sort_values(["model", "roc_auc"], ascending=[True, False])
    top_delta = delta[delta["ablation"].ne("full")].copy() if not delta.empty else delta
    if not top_delta.empty and "roc_auc_drop_vs_full" in top_delta.columns:
        top_delta = top_delta.sort_values(["model", "roc_auc_drop_vs_full"], ascending=[True, False])

    lines = [
        "# Ablation Study Report",
        "",
        "This experiment retrains the selected tabular baseline models after removing one feature family at a time.",
        "The same student-wise temporal train/validation/test split from the processed Junyi learning logs is reused.",
        "",
        "## Leakage Controls",
        "",
        "- The target remains next-attempt correctness for each temporally ordered student attempt row.",
        "- The existing temporal split is reused; rows are not randomly mixed across train, validation, and test.",
        "- Rolling, cumulative, streak, topic-history, and previous-attempt features are already shifted in preprocessing.",
        "- Exercise/topic difficulty proxies are recomputed inside this script from the training split only, with leave-one-out values for training rows.",
        "- Validation and test rows receive only train-estimated difficulty values.",
        "",
        "## Run Settings",
        "",
        f"- Models: {', '.join(models)}",
        f"- Ablations: {', '.join(ablations)}",
        f"- Max train rows: {max_train_rows if max_train_rows is not None else 'full train split'}",
        f"- Max eval rows: {max_eval_rows if max_eval_rows is not None else 'full validation/test splits'}",
        "",
        "## Test Metrics",
        "",
    ]
    metric_rows = test[
        ["model", "ablation", "n", "roc_auc", "pr_auc", "f1", "brier_score", "accuracy"]
    ].to_dict("records")
    lines.extend(markdown_table(metric_rows, ["model", "ablation", "n", "roc_auc", "pr_auc", "f1", "brier_score", "accuracy"]))
    lines.extend(["", "## Drop Relative To Full Feature Set", ""])
    if not top_delta.empty:
        delta_rows = top_delta[
            [
                "model",
                "ablation",
                "roc_auc_drop_vs_full",
                "pr_auc_drop_vs_full",
                "f1_drop_vs_full",
                "brier_score_increase_vs_full",
                "removed_feature_count",
            ]
        ].to_dict("records")
        lines.extend(
            markdown_table(
                delta_rows,
                [
                    "model",
                    "ablation",
                    "roc_auc_drop_vs_full",
                    "pr_auc_drop_vs_full",
                    "f1_drop_vs_full",
                    "brier_score_increase_vs_full",
                    "removed_feature_count",
                ],
            )
        )
    else:
        lines.append("No full-feature baseline was available, so deltas were not computed.")
    lines.extend(["", "## Feature Sets", ""])
    lines.extend(markdown_table(feature_rows, ["ablation", "removed_families", "removed_features", "numeric_feature_count", "categorical_feature_count"]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    pandas = require_pandas()
    config = load_config(args.config)
    logger = setup_logger("ablation_study")
    seed = int(config["baseline"]["random_seed"])
    np.random.seed(seed)

    ablation_config = config.get("ablation", {})
    models = args.models or ablation_config.get("models") or ["logistic_regression"]
    ablations = args.ablations or ablation_config.get("ablations") or DEFAULT_ABLATIONS
    max_train_rows = configured_optional_int(args.max_train_rows, ablation_config.get("max_train_rows", config["baseline"].get("max_train_rows")))
    max_eval_rows = configured_optional_int(args.max_eval_rows, ablation_config.get("max_eval_rows", config["baseline"].get("max_eval_rows")))

    unknown_models = sorted(set(models) - SUPPORTED_MODELS)
    if unknown_models:
        raise ValueError(f"Unsupported ablation models: {unknown_models}")
    unknown_ablations = sorted(set(ablations) - set(ABLATION_REMOVALS))
    if unknown_ablations:
        raise ValueError(f"Unsupported ablations: {unknown_ablations}")

    features_path = Path(args.features) if args.features else configured_path(config, "processed_dir") / config["preprocessing"]["output_file"]
    if not features_path.is_absolute():
        features_path = ROOT / features_path
    output_dir = configured_output_dir(config, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configured_path(config, "reports_dir").mkdir(parents=True, exist_ok=True)

    df = read_feature_frame(features_path, config, logger)
    all_numeric_cols = [col for col in config["features"]["baseline_numeric_columns"] if col in df.columns]
    all_categorical_cols = [col for col in config["features"]["categorical_columns"] if col in df.columns]
    target_col = config["dataset"]["correct_col"]

    train = sample_split(df, "train", max_train_rows, seed)
    val = sample_split(df, "val", max_eval_rows, seed)
    test = sample_split(df, "test", max_eval_rows, seed)
    logger.info("Ablation frame rows: train=%s val=%s test=%s", len(train), len(val), len(test))

    metrics_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    df_columns = set(df.columns)
    eval_sets = {"val": val, "test": test}
    for ablation_name in ablations:
        numeric_cols, categorical_cols, removed = ablation_feature_set(
            ablation_name,
            all_numeric_cols,
            all_categorical_cols,
            df_columns,
        )
        feature_cols = numeric_cols + categorical_cols
        if not feature_cols:
            raise ValueError(f"Ablation {ablation_name} removed all available features.")
        removed_families = ABLATION_REMOVALS[ablation_name]
        feature_rows.append(
            {
                "ablation": ablation_name,
                "removed_families": ";".join(removed_families) if removed_families else "none",
                "removed_features": ";".join(removed) if removed else "none",
                "numeric_feature_count": len(numeric_cols),
                "categorical_feature_count": len(categorical_cols),
            }
        )
        X_train = train[feature_cols]
        y_train = train[target_col].astype(int).to_numpy()
        for model_name in models:
            logger.info(
                "Training ablation=%s model=%s numeric=%s categorical=%s",
                ablation_name,
                model_name,
                len(numeric_cols),
                len(categorical_cols),
            )
            estimator = build_estimator(model_name, config, numeric_cols, categorical_cols, seed)
            estimator.fit(X_train, y_train)
            for split_name, split_df in eval_sets.items():
                if split_df.empty:
                    continue
                X_eval = split_df[feature_cols]
                y_eval = split_df[target_col].astype(int).to_numpy()
                y_prob = estimator.predict_proba(X_eval)[:, 1]
                row = classification_metrics(y_eval, y_prob)
                row.update(
                    {
                        "model": model_name,
                        "split": split_name,
                        "ablation": ablation_name,
                        "removed_families": ";".join(removed_families) if removed_families else "none",
                        "removed_features": ";".join(removed) if removed else "none",
                        "removed_feature_count": len(removed),
                        "numeric_feature_count": len(numeric_cols),
                        "categorical_feature_count": len(categorical_cols),
                        "train_rows": len(train),
                        "eval_rows": len(split_df),
                    }
                )
                metrics_rows.append(row)

    metrics = pandas.DataFrame(metrics_rows)
    delta = build_delta_table(metrics)
    metrics_path = output_dir / "ablation_metrics.csv"
    delta_path = output_dir / "ablation_test_deltas.csv"
    feature_path = output_dir / "ablation_feature_sets.csv"
    metrics.to_csv(metrics_path, index=False)
    delta.to_csv(delta_path, index=False)
    write_rows(feature_path, feature_rows)
    save_ablation_plots(delta, output_dir, logger)

    report_path = output_dir / "ablation_study_report.md"
    write_report(report_path, metrics, delta, feature_rows, list(models), list(ablations), max_train_rows, max_eval_rows)
    reports_copy = configured_path(config, "reports_dir") / "ablation_study_report.md"
    reports_copy.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    logger.info("Saved ablation metrics to %s", metrics_path)
    logger.info("Saved ablation report to %s", report_path)


if __name__ == "__main__":
    main()
