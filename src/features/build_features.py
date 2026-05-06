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
except ImportError:  # pragma: no cover - handled at runtime
    pd = None

from src.common.config import configured_path, ensure_project_dirs, load_config, raw_file_path
from src.common.logging_utils import setup_logger


LOG_COLUMNS = [
    "timestamp_TW",
    "uuid",
    "ucid",
    "upid",
    "problem_number",
    "exercise_problem_repeat_session",
    "is_correct",
    "total_sec_taken",
    "total_attempt_cnt",
    "used_hint_cnt",
    "is_hint_used",
    "is_downgrade",
    "is_upgrade",
    "level",
]
CONTENT_COLUMNS = [
    "ucid",
    "content_pretty_name",
    "content_kind",
    "difficulty",
    "subject",
    "learning_stage",
    "level1_id",
    "level2_id",
    "level3_id",
    "level4_id",
]
NUMERIC_COLUMNS = [
    "problem_number",
    "exercise_problem_repeat_session",
    "total_sec_taken",
    "total_attempt_cnt",
    "used_hint_cnt",
    "level",
]
BOOL_COLUMNS = ["is_correct", "is_hint_used", "is_downgrade", "is_upgrade"]
REQUIRED_COLUMNS = ["timestamp_TW", "uuid", "ucid", "upid", "is_correct"]
DIFFICULTY_ORDER = {"unset": 0, "easy": 1, "normal": 2, "hard": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-aware temporal learning features from Junyi logs.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--mode", choices=["quick", "full"], default=None)
    parser.add_argument("--quick-rows", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--include-user-metadata", action="store_true")
    return parser.parse_args()


def require_pandas() -> Any:
    if pd is None:
        raise ImportError(
            "pandas is required for feature engineering. Install project dependencies with "
            "`python -m pip install -r requirements.txt`."
        )
    return pd


def standardize_bool(series: Any) -> Any:
    mapping = {
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "True": 1,
        "False": 0,
        True: 1,
        False: 0,
    }
    return series.map(mapping)


def memory_mb(df: Any) -> float:
    return float(df.memory_usage(deep=True).sum() / 1024**2)


def read_raw_tables(config: dict[str, Any], mode: str, quick_rows: int | None, include_user_metadata: bool, logger: Any) -> tuple[Any, dict[str, int]]:
    pandas = require_pandas()
    log_path = raw_file_path(config, "log_file")
    content_path = raw_file_path(config, "content_file")
    user_path = raw_file_path(config, "user_file")
    nrows = quick_rows if mode == "quick" else None
    logger.info("Reading %s (mode=%s, nrows=%s)", log_path, mode, nrows)
    df = pandas.read_csv(log_path, usecols=LOG_COLUMNS, nrows=nrows, dtype="string")
    df.insert(0, "raw_row_id", np.arange(len(df), dtype=np.int64))
    report = {"raw_rows": len(df)}

    logger.info("Reading content metadata from %s", content_path)
    content = pandas.read_csv(content_path, usecols=CONTENT_COLUMNS, dtype="string")
    report["content_rows"] = len(content)
    df = df.merge(content, on="ucid", how="left", validate="many_to_one")
    report["rows_missing_content"] = int(df["content_kind"].isna().sum())

    if include_user_metadata:
        logger.info("Reading optional user metadata from %s", user_path)
        user_cols = ["uuid", "user_grade", "has_teacher_cnt", "belongs_to_class_cnt", "has_class_cnt"]
        user = pandas.read_csv(user_path, usecols=user_cols, dtype="string")
        report["user_rows"] = len(user)
        df = df.merge(user, on="uuid", how="left", validate="many_to_one")
        report["rows_missing_user_metadata"] = int(df["user_grade"].isna().sum())
    return df, report


def clean_attempts(df: Any, logger: Any) -> tuple[Any, dict[str, int]]:
    pandas = require_pandas()
    report: dict[str, int] = {}
    report["rows_before_cleaning"] = len(df)
    missing_required = df[REQUIRED_COLUMNS].isna().any(axis=1)
    missing_required |= (df[REQUIRED_COLUMNS] == "").any(axis=1)
    report["rows_missing_required_fields"] = int(missing_required.sum())
    df = df.loc[~missing_required].copy()

    for col in BOOL_COLUMNS:
        df[col] = standardize_bool(df[col])
    invalid_correct = df["is_correct"].isna()
    report["rows_invalid_correctness"] = int(invalid_correct.sum())
    df = df.loc[~invalid_correct].copy()
    df["is_correct"] = df["is_correct"].astype("int8")
    for col in ["is_hint_used", "is_downgrade", "is_upgrade"]:
        df[f"{col}_missing"] = df[col].isna().astype("int8")
        df[col] = df[col].fillna(0).astype("int8")

    for col in NUMERIC_COLUMNS:
        df[col] = pandas.to_numeric(df[col], errors="coerce")
    invalid_numeric = df[NUMERIC_COLUMNS].isna().any(axis=1)
    report["rows_with_invalid_numeric_fields"] = int(invalid_numeric.sum())
    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].fillna(0)

    df["timestamp"] = pandas.to_datetime(df["timestamp_TW"], errors="coerce", utc=True)
    invalid_timestamp = df["timestamp"].isna()
    report["rows_invalid_timestamp"] = int(invalid_timestamp.sum())
    df = df.loc[~invalid_timestamp].copy()

    duplicate_mask = df.duplicated(subset=LOG_COLUMNS)
    report["exact_duplicate_attempt_rows"] = int(duplicate_mask.sum())
    df = df.loc[~duplicate_mask].copy()

    negative_like = (
        (df["total_sec_taken"] < 0)
        | (df["total_attempt_cnt"] < 0)
        | (df["used_hint_cnt"] < 0)
        | (df["problem_number"] < 0)
        | (df["exercise_problem_repeat_session"] < 0)
    )
    report["rows_with_negative_numeric_fields"] = int(negative_like.sum())
    df = df.loc[~negative_like].copy()

    df["difficulty"] = df["difficulty"].fillna("unknown")
    df["learning_stage"] = df["learning_stage"].fillna("unknown")
    for col in ["level1_id", "level2_id", "level3_id", "level4_id", "content_kind", "subject"]:
        df[col] = df[col].fillna("unknown")
    df["content_difficulty_ordinal"] = df["difficulty"].map(DIFFICULTY_ORDER).fillna(0).astype("int8")

    report["rows_after_cleaning"] = len(df)
    report["unique_students_after_cleaning"] = int(df["uuid"].nunique())
    report["unique_exercises_after_cleaning"] = int(df["ucid"].nunique())
    report["unique_problems_after_cleaning"] = int(df["upid"].nunique())
    report["correct_attempts_after_cleaning"] = int(df["is_correct"].sum())
    report["incorrect_attempts_after_cleaning"] = int(len(df) - df["is_correct"].sum())
    if len(df):
        report["date_min"] = str(df["timestamp"].min())
        report["date_max"] = str(df["timestamp"].max())
    logger.info("Cleaned attempts: %s -> %s rows", report["rows_before_cleaning"], report["rows_after_cleaning"])
    return df, report


def add_temporal_features(df: Any, config: dict[str, Any], logger: Any) -> Any:
    pandas = require_pandas()
    logger.info("Sorting attempts by student, timestamp, and stable raw row id")
    df = df.sort_values(["uuid", "timestamp", "raw_row_id"], kind="mergesort").reset_index(drop=True)
    group = df.groupby("uuid", sort=False)

    df["attempt_index"] = group.cumcount().astype("int32")
    df["hist_attempt_count"] = df["attempt_index"].astype("int32")
    cumulative_correct = group["is_correct"].cumsum()
    df["hist_correct_count"] = (cumulative_correct - df["is_correct"]).astype("int32")
    df["student_prev_accuracy"] = (
        df["hist_correct_count"] / df["hist_attempt_count"].replace(0, np.nan)
    ).fillna(0.5).astype("float32")

    prev_correct = group["is_correct"].shift(1)
    df["prev_is_correct"] = prev_correct.fillna(0.5).astype("float32")
    for col in ["total_sec_taken", "total_attempt_cnt", "used_hint_cnt", "is_hint_used"]:
        df[f"prev_{col}"] = group[col].shift(1).fillna(0).astype("float32")

    for window in (5, 10):
        rolled = (
            prev_correct.groupby(df["uuid"], sort=False)
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[f"rolling_accuracy_{window}"] = rolled.fillna(0.5).astype("float32")

    prev_timestamp = group["timestamp"].shift(1)
    raw_gap = (df["timestamp"] - prev_timestamp).dt.total_seconds()
    clip_seconds = int(config["preprocessing"]["clip_time_gap_hours"] * 3600)
    df["time_gap_sec"] = raw_gap.fillna(0).clip(lower=0, upper=clip_seconds).astype("float32")
    df["negative_time_gap_after_sort_count"] = (raw_gap < 0).fillna(False).astype("int8")

    df["attempt_date"] = df["timestamp"].dt.date.astype("string")
    df["daily_activity_count_prior"] = df.groupby(["uuid", "attempt_date"], sort=False).cumcount().astype("int16")

    session_gap_sec = int(config["preprocessing"]["session_gap_minutes"] * 60)
    new_session = (df["hist_attempt_count"] == 0) | (df["time_gap_sec"] > session_gap_sec)
    df["session_id"] = new_session.groupby(df["uuid"], sort=False).cumsum().astype("int32")
    df["session_attempt_index"] = df.groupby(["uuid", "session_id"], sort=False).cumcount().astype("int16")

    df["student_exercise_attempt_count_prior"] = df.groupby(["uuid", "ucid"], sort=False).cumcount().astype("int16")
    topic_key = "level4_id" if "level4_id" in df.columns else "level3_id"
    topic_group = df.groupby(["uuid", topic_key], sort=False)
    df["topic_attempt_count_prior"] = topic_group.cumcount().astype("int16")
    topic_correct = topic_group["is_correct"].cumsum() - df["is_correct"]
    df["topic_prev_accuracy"] = (
        topic_correct / df["topic_attempt_count_prior"].replace(0, np.nan)
    ).fillna(0.5).astype("float32")

    correct_streak, wrong_streak = previous_streaks(df["is_correct"].to_numpy(), group.indices)
    df["consecutive_correct_count"] = correct_streak
    df["consecutive_wrong_count"] = wrong_streak
    logger.info("Temporal feature frame uses %.1f MB", memory_mb(df))
    return df


def previous_streaks(values: np.ndarray, group_indices: dict[Any, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    correct = np.zeros(len(values), dtype=np.int16)
    wrong = np.zeros(len(values), dtype=np.int16)
    for indices in group_indices.values():
        correct_run = 0
        wrong_run = 0
        for idx in indices:
            correct[idx] = correct_run
            wrong[idx] = wrong_run
            if values[idx] == 1:
                correct_run += 1
                wrong_run = 0
            else:
                wrong_run += 1
                correct_run = 0
    return correct, wrong


def assign_temporal_splits(df: Any, config: dict[str, Any], logger: Any) -> Any:
    train_frac = float(config["splits"]["train_frac"])
    val_frac = float(config["splits"]["val_frac"])
    min_attempts = int(config["splits"]["min_attempts_per_student"])
    group = df.groupby("uuid", sort=False)
    student_attempts = group["uuid"].transform("size")
    before = len(df)
    df = df.loc[student_attempts >= min_attempts].copy()
    group = df.groupby("uuid", sort=False)
    pos = group.cumcount()
    n = group["uuid"].transform("size")
    train_cut = np.floor(n * train_frac).astype(int)
    val_cut = np.floor(n * (train_frac + val_frac)).astype(int)
    df["split"] = np.where(pos < train_cut, "train", np.where(pos < val_cut, "val", "test"))
    if config["preprocessing"]["drop_first_attempt_without_history"]:
        df = df.loc[df["hist_attempt_count"] > 0].copy()
    logger.info("Assigned temporal splits after min-attempt filtering: %s -> %s rows", before, len(df))
    return df


def write_report_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_feature_documentation(path: Path) -> None:
    rows = [
        {"feature": "student_prev_accuracy", "meaning": "Cumulative accuracy for the student before the target attempt.", "leakage_control": "Uses cumulative correct count shifted before current row."},
        {"feature": "rolling_accuracy_5/10", "meaning": "Recent student performance over the last 5 or 10 prior attempts.", "leakage_control": "Computed from shifted correctness history only."},
        {"feature": "consecutive_wrong_count/correct_count", "meaning": "Immediate streaks before the target attempt.", "leakage_control": "Stores streak length observed before current correctness is known."},
        {"feature": "time_gap_sec", "meaning": "Time elapsed since previous attempt by the same student.", "leakage_control": "Depends only on timestamps and prior attempt."},
        {"feature": "daily_activity_count_prior", "meaning": "Student's earlier attempts on the same date.", "leakage_control": "Uses per-student per-day cumcount before current row."},
        {"feature": "session_attempt_index", "meaning": "Position within a session using a 30-minute inactivity break by default.", "leakage_control": "Session boundaries use previous timestamp gaps only."},
        {"feature": "student_exercise_attempt_count_prior", "meaning": "How often the student previously attempted the same exercise.", "leakage_control": "Uses grouped cumcount before current exercise attempt."},
        {"feature": "topic_prev_accuracy", "meaning": "Prior accuracy for the student within the finest available topic level.", "leakage_control": "Uses topic-level cumulative correctness shifted before current row."},
        {"feature": "prev_total_sec_taken/prev_used_hint_cnt", "meaning": "Behavior observed on the student's previous attempt.", "leakage_control": "Current attempt hint/time fields are not used by tabular baselines."},
        {"feature": "content_difficulty_ordinal", "meaning": "Metadata difficulty encoded from Info_Content.csv.", "leakage_control": "Comes from content metadata rather than outcome aggregation."},
    ]
    write_report_rows(path, rows)


def write_summary_report(path: Path, report: dict[str, Any], split_counts: dict[str, int], output_path: Path) -> None:
    lines = [
        "# Preprocessing Report",
        "",
        "The unit of analysis is a student's ordered online exercise attempt sequence.",
        "",
        "## Counts",
        "",
    ]
    for key, value in report.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Temporal Splits", ""])
    for split, count in split_counts.items():
        lines.append(f"- {split}: {count}")
    lines.extend(
        [
            "",
            "## Leakage Controls",
            "",
            "- Rows are sorted within `uuid` by `timestamp_TW` plus stable raw row order.",
            "- Behavioral history features use `shift`, `cumcount`, or prior cumulative counts.",
            "- Current-attempt hint usage, solve time, and attempt count are retained for sequence history, but tabular baselines use only their previous-attempt versions.",
            "- Outcome-derived exercise and topic difficulty proxies are intentionally deferred to model training, where they are estimated from the training split only with leave-one-out values for training rows.",
            "",
            f"Processed feature file: `{output_path}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_basic_plots(df: Any, figures_dir: Path, logger: Any) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed; skipping feature plots")
        return

    correctness = df["is_correct"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["incorrect", "correct"], [correctness.get(0, 0), correctness.get(1, 0)], color=["#b94b5f", "#3f7f5f"])
    ax.set_title("Correctness Distribution")
    ax.set_ylabel("Attempts")
    fig.tight_layout()
    fig.savefig(figures_dir / "correctness_distribution.png", dpi=160)
    plt.close(fig)

    daily = df.groupby("attempt_date").size()
    fig, ax = plt.subplots(figsize=(9, 4))
    daily.plot(ax=ax, color="#2f5d83")
    ax.set_title("Activity Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Attempts")
    fig.tight_layout()
    fig.savefig(figures_dir / "activity_over_time.png", dpi=160)
    plt.close(fig)

    student_counts = df.groupby("uuid").size()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(student_counts, bins=50, color="#667a3e")
    ax.set_title("Student Attempt Distribution")
    ax.set_xlabel("Attempts per student")
    ax.set_ylabel("Students")
    fig.tight_layout()
    fig.savefig(figures_dir / "student_attempt_distribution.png", dpi=160)
    plt.close(fig)

    exercise_difficulty = 1.0 - df.groupby("ucid")["is_correct"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(exercise_difficulty, bins=40, color="#8f6d3d")
    ax.set_title("Observed Exercise Incorrect Rate")
    ax.set_xlabel("Incorrect rate")
    ax.set_ylabel("Exercises")
    fig.tight_layout()
    fig.savefig(figures_dir / "exercise_difficulty_distribution.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_project_dirs(config)
    logger = setup_logger("build_features")
    mode = args.mode or config["preprocessing"]["mode"]
    quick_rows = args.quick_rows or config["preprocessing"]["quick_rows"]
    include_user_metadata = args.include_user_metadata or bool(config["preprocessing"]["include_user_metadata"])

    df, raw_report = read_raw_tables(config, mode=mode, quick_rows=quick_rows, include_user_metadata=include_user_metadata, logger=logger)
    logger.info("Raw joined frame uses %.1f MB", memory_mb(df))
    df, clean_report = clean_attempts(df, logger)
    df = add_temporal_features(df, config, logger)
    df = assign_temporal_splits(df, config, logger)

    processed_dir = configured_path(config, "processed_dir")
    output_name = args.output or config["preprocessing"]["output_file"]
    output_path = processed_dir / output_name
    logger.info("Writing processed features to %s", output_path)
    df.to_csv(output_path, index=False)

    metrics_dir = configured_path(config, "metrics_dir")
    figures_dir = configured_path(config, "figures_dir")
    reports_dir = configured_path(config, "reports_dir")
    split_counts = df["split"].value_counts().to_dict()
    report = {**raw_report, **clean_report, "mode": mode, "output_rows": len(df)}
    write_report_rows(metrics_dir / "preprocessing_summary.csv", [report])
    write_report_rows(metrics_dir / "split_counts.csv", [{"split": k, "rows": v} for k, v in split_counts.items()])
    write_feature_documentation(metrics_dir / "feature_documentation.csv")
    write_summary_report(reports_dir / "preprocessing_report.md", report, split_counts, output_path)
    make_basic_plots(df, figures_dir, logger)
    logger.info("Feature build complete")


if __name__ == "__main__":
    main()
