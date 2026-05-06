from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.common.config import configured_path, ensure_project_dirs, load_config
from src.common.logging_utils import setup_logger
from src.evaluation.metrics import classification_metrics
from src.evaluation.plots import (
    save_calibration_plot,
    save_confusion_matrix_plot,
    save_model_comparison,
    save_roc_pr_curves,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train leakage-aware tabular baselines for next-attempt correctness.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--features", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-eval-rows", type=int, default=None)
    return parser.parse_args()


def require_pandas() -> Any:
    if pd is None:
        raise ImportError(
            "pandas is required for baseline training. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        )
    return pd


def add_train_only_difficulty_proxies(df: Any) -> Any:
    df = df.copy()
    df["exercise_incorrect_rate_train"] = leakage_safe_incorrect_rate(df, "ucid")
    topic_col = "level4_id" if "level4_id" in df.columns else "level3_id"
    df["topic_incorrect_rate_train"] = leakage_safe_incorrect_rate(df, topic_col)
    return df


def leakage_safe_incorrect_rate(df: Any, group_col: str) -> Any:
    """Target encoding for difficulty without letting a training row encode itself.

    Validation and test rows receive the aggregate incorrect rate estimated from
    the training split only. Training rows receive a leave-one-out version of the
    same aggregate, so the row's own correctness label is not embedded in its
    feature value.
    """
    train_mask = df["split"].eq("train")
    train = df.loc[train_mask, [group_col, "is_correct"]].copy()
    train["incorrect"] = 1.0 - train["is_correct"].astype(float)
    total_train = len(train)
    if total_train == 0:
        return np.full(len(df), 0.5, dtype=np.float32)

    total_incorrect = float(train["incorrect"].sum())
    global_incorrect = total_incorrect / total_train
    grouped = train.groupby(group_col, dropna=False)["incorrect"].agg(["sum", "count"])
    aggregate = (grouped["sum"] / grouped["count"]).astype(float)
    encoded = df[group_col].map(aggregate).fillna(global_incorrect).astype(float)

    train_indices = df.index[train_mask]
    train_group_sum = df.loc[train_mask, group_col].map(grouped["sum"]).astype(float).to_numpy()
    train_group_count = df.loc[train_mask, group_col].map(grouped["count"]).astype(float).to_numpy()
    current_incorrect = (1.0 - df.loc[train_mask, "is_correct"].astype(float)).to_numpy()
    loo_denominator = train_group_count - 1.0
    if total_train > 1:
        global_loo = (total_incorrect - current_incorrect) / (total_train - 1.0)
    else:
        global_loo = np.full_like(current_incorrect, global_incorrect, dtype=float)
    loo_value = np.divide(
        train_group_sum - current_incorrect,
        loo_denominator,
        out=global_loo.astype(float, copy=True),
        where=loo_denominator > 0,
    )
    encoded.loc[train_indices] = loo_value
    return encoded.astype("float32")


def sample_split(df: Any, split: str, max_rows: int | None, seed: int) -> Any:
    part = df[df["split"] == split].copy()
    if max_rows and len(part) > max_rows:
        part = part.sample(max_rows, random_state=seed)
    return part


def build_logistic_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_cols),
            ("cat", categorical, categorical_cols),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )


def build_random_forest_pipeline(numeric_cols: list[str], categorical_cols: list[str], n_estimators: int, seed: int) -> Pipeline:
    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_cols),
            ("cat", categorical, categorical_cols),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=24,
                    min_samples_leaf=20,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_hist_gradient_boosting_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    max_iter: int,
    learning_rate: float,
    seed: int,
) -> Pipeline:
    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_cols),
            ("cat", categorical, categorical_cols),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=max_iter,
                    learning_rate=learning_rate,
                    l2_regularization=0.01,
                    random_state=seed,
                    early_stopping=True,
                    validation_fraction=0.1,
                ),
            ),
        ]
    )


def evaluate_prediction_frame(model_name: str, split: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    metrics = classification_metrics(y_true, y_prob)
    metrics["model"] = model_name
    metrics["split"] = split
    return metrics


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def top_feature_rows(model_name: str, pipeline: Pipeline, limit: int = 100) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model = pipeline.named_steps["model"]
    preprocess = pipeline.named_steps["preprocess"]
    try:
        names = preprocess.get_feature_names_out()
    except Exception:
        names = np.array([f"feature_{i}" for i in range(getattr(model, "n_features_in_", 0))])
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        return rows
    order = np.argsort(values)[::-1][:limit]
    for rank, idx in enumerate(order, start=1):
        rows.append(
            {
                "model": model_name,
                "rank": rank,
                "feature": str(names[idx]),
                "importance": float(values[idx]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    pandas = require_pandas()
    config = load_config(args.config)
    ensure_project_dirs(config)
    logger = setup_logger("train_baseline")
    seed = int(config["baseline"]["random_seed"])
    np.random.seed(seed)

    features_path = Path(args.features) if args.features else configured_path(config, "processed_dir") / config["preprocessing"]["output_file"]
    if not features_path.is_absolute():
        features_path = ROOT / features_path
    logger.info("Reading features from %s", features_path)
    df = pandas.read_csv(features_path)
    df = add_train_only_difficulty_proxies(df)

    numeric_cols = [col for col in config["features"]["baseline_numeric_columns"] if col in df.columns]
    categorical_cols = [col for col in config["features"]["categorical_columns"] if col in df.columns]
    target_col = config["dataset"]["correct_col"]
    models_to_run = args.models or config["baseline"]["models"]
    max_train_rows = args.max_train_rows if args.max_train_rows is not None else config["baseline"]["max_train_rows"]
    max_eval_rows = args.max_eval_rows if args.max_eval_rows is not None else config["baseline"]["max_eval_rows"]

    train = sample_split(df, "train", max_train_rows, seed)
    val = sample_split(df, "val", max_eval_rows, seed)
    test = sample_split(df, "test", max_eval_rows, seed)
    logger.info("Training rows=%s, validation rows=%s, test rows=%s", len(train), len(val), len(test))

    feature_cols = numeric_cols + categorical_cols
    X_train = train[feature_cols]
    y_train = train[target_col].astype(int).to_numpy()
    eval_sets = {"val": val, "test": test}
    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[Any] = []
    importance_rows: list[dict[str, Any]] = []
    models_dir = configured_path(config, "models_dir")
    figures_dir = configured_path(config, "figures_dir")
    metrics_dir = configured_path(config, "metrics_dir")

    base_rate = float(np.mean(y_train))
    for model_name in models_to_run:
        logger.info("Training/evaluating baseline: %s", model_name)
        estimator: Pipeline | None = None
        if model_name == "majority":
            pass
        elif model_name == "previous_correct":
            pass
        elif model_name == "logistic_regression":
            estimator = build_logistic_pipeline(numeric_cols, categorical_cols)
            estimator.fit(X_train, y_train)
        elif model_name == "random_forest":
            estimator = build_random_forest_pipeline(
                numeric_cols,
                categorical_cols,
                n_estimators=int(config["baseline"]["random_forest_estimators"]),
                seed=seed,
            )
            estimator.fit(X_train, y_train)
        elif model_name == "hist_gradient_boosting":
            estimator = build_hist_gradient_boosting_pipeline(
                numeric_cols,
                categorical_cols,
                max_iter=int(config["baseline"]["hist_gradient_boosting_max_iter"]),
                learning_rate=float(config["baseline"]["hist_gradient_boosting_learning_rate"]),
                seed=seed,
            )
            estimator.fit(X_train, y_train)
        else:
            raise ValueError(f"Unknown baseline model: {model_name}")

        if estimator is not None:
            joblib.dump(estimator, models_dir / f"{model_name}.joblib")
            importance_rows.extend(top_feature_rows(model_name, estimator))

        for split, split_df in eval_sets.items():
            if split_df.empty:
                continue
            X_eval = split_df[feature_cols]
            y_eval = split_df[target_col].astype(int).to_numpy()
            if model_name == "majority":
                y_prob = np.full(len(split_df), base_rate, dtype=float)
            elif model_name == "previous_correct":
                y_prob = split_df["prev_is_correct"].fillna(base_rate).astype(float).to_numpy()
            else:
                y_prob = estimator.predict_proba(X_eval)[:, 1]  # type: ignore[union-attr]
            metrics_rows.append(evaluate_prediction_frame(model_name, split, y_eval, y_prob))
            pred_frame = pandas.DataFrame(
                {
                    "model": model_name,
                    "split": split,
                    "uuid": split_df["uuid"].to_numpy(),
                    "timestamp": split_df["timestamp"].to_numpy() if "timestamp" in split_df.columns else "",
                    "y_true": y_eval,
                    "y_prob": y_prob,
                }
            )
            prediction_frames.append(pred_frame)
            if split == "test":
                safe_name = model_name.replace("/", "_")
                save_roc_pr_curves(pred_frame, figures_dir, safe_name)
                save_calibration_plot(pred_frame, figures_dir, safe_name)
                save_confusion_matrix_plot(pred_frame, figures_dir, safe_name)

    metrics = pandas.DataFrame(metrics_rows)
    predictions = pandas.concat(prediction_frames, ignore_index=True) if prediction_frames else pandas.DataFrame()
    metrics.to_csv(metrics_dir / "baseline_metrics.csv", index=False)
    predictions.to_csv(metrics_dir / "baseline_predictions.csv", index=False)
    save_rows(metrics_dir / "baseline_feature_importance.csv", importance_rows)
    if not metrics.empty:
        save_model_comparison(metrics[metrics["split"] == "test"], figures_dir)
    logger.info("Saved baseline metrics, predictions, model artifacts, and plots")


if __name__ == "__main__":
    main()
