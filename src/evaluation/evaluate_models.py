from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from src.common.config import configured_path, ensure_project_dirs, load_config
from src.common.logging_utils import setup_logger
from src.evaluation.plots import save_model_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate model metrics and generate final evaluation reports.")
    parser.add_argument("--config", default="configs/default.json")
    return parser.parse_args()


def require_pandas() -> Any:
    if pd is None:
        raise ImportError(
            "pandas is required for evaluation. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        )
    return pd


def collect_metrics(metrics_dir: Path) -> Any:
    pandas = require_pandas()
    frames = []
    for path in sorted(metrics_dir.glob("*metrics.csv")):
        frame = pandas.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    return pandas.concat(frames, ignore_index=True, sort=False) if frames else pandas.DataFrame()


def write_methodology_summary(path: Path) -> None:
    lines = [
        "# Methodology Summary",
        "",
        "This project is an offline educational data mining study using historical Junyi Academy online learning logs.",
        "",
        "## Data Unit",
        "",
        "The unit of analysis is a student's ordered exercise-attempt sequence from `Log_Problem.csv`. Attempts are linked to exercise metadata from `Info_Content.csv` through `ucid`. User metadata from `Info_UserData.csv` is report-only by default and is not the focus of prediction.",
        "",
        "## Prediction Target",
        "",
        "The target is next-attempt correctness. For tabular baselines, each row's correctness is predicted from features available before that attempt. For neural sequence models, a sliding window of prior attempts predicts the following attempt's correctness.",
        "",
        "## Leakage Prevention",
        "",
        "- Student records are sorted by `uuid`, `timestamp_TW`, and a stable raw row id.",
        "- Cumulative, rolling, streak, session, and topic-history features are shifted so the current correctness is not used to predict itself.",
        "- Current-attempt solve time and hint usage are not used as tabular predictors for that same attempt.",
        "- Outcome-derived exercise/topic difficulty proxies are estimated from the training split only.",
        "- Splits are temporal within student rather than random row-level splits.",
        "",
        "## Models",
        "",
        "Baselines include majority correctness, previous-correctness heuristic, logistic regression, and random forest. The sequence model is a PyTorch Transformer Encoder with categorical embeddings, numerical temporal features, positional embeddings, dropout, and a binary classification head. A GRU baseline is available through the same training script.",
        "",
        "## Bayesian Approximation",
        "",
        "Monte Carlo Dropout keeps dropout active during inference and performs repeated stochastic forward passes. The mean predicted probability is the final prediction, while predictive standard deviation summarizes uncertainty.",
        "",
        "## Educational Interpretation",
        "",
        "Risk groups separate low predicted correctness from high uncertainty. The distinction between high-risk confident and high-risk uncertain predictions matters: the former suggests prioritized intervention, while the latter suggests collecting more learning evidence or using a cautious human review.",
        "",
        "## Scope",
        "",
        "The work does not use camera tracking, image monitoring, object trajectories, face recognition, or person re-identification. It should not be overclaimed as a deployed intervention system; it is an offline study of historical learning behavior.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_uncertainty_plots(metrics_dir: Path, figures_dir: Path, reports_dir: Path) -> None:
    pandas = require_pandas()
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    frames = []
    for path in sorted(metrics_dir.glob("*sequence_predictions.csv")):
        frame = pandas.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        return
    predictions = pandas.concat(frames, ignore_index=True)
    if "predictive_std" not in predictions.columns:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(predictions["predictive_std"], bins=40, color="#6d5b8c")
    ax.set_title("MC Dropout Predictive Uncertainty")
    ax.set_xlabel("Predictive standard deviation")
    ax.set_ylabel("Predictions")
    fig.tight_layout()
    fig.savefig(figures_dir / "uncertainty_distribution.png", dpi=160)
    plt.close(fig)

    if "risk_group" in predictions.columns:
        risk = (
            predictions.groupby("risk_group")
            .agg(
                count=("y_true", "size"),
                observed_correctness=("y_true", "mean"),
                mean_predicted_correctness=("y_prob", "mean"),
                mean_predictive_std=("predictive_std", "mean"),
            )
            .reset_index()
        )
        risk.to_csv(metrics_dir / "risk_group_analysis.csv", index=False)
        lines = ["# Risk Group Analysis", ""]
        for row in risk.to_dict("records"):
            lines.append(
                f"- {row['risk_group']}: n={row['count']}, observed correctness={row['observed_correctness']:.3f}, "
                f"mean predicted correctness={row['mean_predicted_correctness']:.3f}, mean uncertainty={row['mean_predictive_std']:.3f}"
            )
        (reports_dir / "risk_group_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    pandas = require_pandas()
    config = load_config(args.config)
    ensure_project_dirs(config)
    logger = setup_logger("evaluate_models")
    metrics_dir = configured_path(config, "metrics_dir")
    figures_dir = configured_path(config, "figures_dir")
    reports_dir = configured_path(config, "reports_dir")

    metrics = collect_metrics(metrics_dir)
    if metrics.empty:
        logger.warning("No metrics files found under %s", metrics_dir)
    else:
        metrics.to_csv(metrics_dir / "model_performance_comparison.csv", index=False)
        if "split" in metrics.columns and "model" in metrics.columns:
            save_model_comparison(metrics[metrics["split"] == "test"], figures_dir)
        logger.info("Wrote model performance comparison with %s rows", len(metrics))

    make_uncertainty_plots(metrics_dir, figures_dir, reports_dir)
    write_methodology_summary(reports_dir / "methodology_summary.md")
    logger.info("Wrote final methodology and evaluation reports")


if __name__ == "__main__":
    main()
