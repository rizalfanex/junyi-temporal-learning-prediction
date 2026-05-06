from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix


def save_roc_pr_curves(predictions: pd.DataFrame, figures_dir: Path, prefix: str) -> None:
    import matplotlib.pyplot as plt

    if predictions["y_true"].nunique() < 2:
        return
    y_true = predictions["y_true"].astype(int)
    y_prob = predictions["y_prob"].astype(float)
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title(f"{prefix} ROC Curve")
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}_roc_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title(f"{prefix} PR Curve")
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}_pr_curve.png", dpi=160)
    plt.close(fig)


def save_calibration_plot(predictions: pd.DataFrame, figures_dir: Path, prefix: str) -> None:
    import matplotlib.pyplot as plt

    if predictions["y_true"].nunique() < 2:
        return
    y_true = predictions["y_true"].astype(int)
    y_prob = predictions["y_prob"].astype(float).clip(0, 1)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(prob_pred, prob_true, marker="o", label=prefix)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed correctness rate")
    ax.set_title(f"{prefix} Calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}_calibration.png", dpi=160)
    plt.close(fig)


def save_confusion_matrix_plot(predictions: pd.DataFrame, figures_dir: Path, prefix: str, threshold: float = 0.5) -> None:
    import matplotlib.pyplot as plt

    y_true = predictions["y_true"].astype(int)
    y_pred = (predictions["y_prob"].astype(float) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", color="black")
    ax.set_title(f"{prefix} Confusion Matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}_confusion_matrix.png", dpi=160)
    plt.close(fig)


def save_model_comparison(metrics: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if metrics.empty:
        return
    metric_names = [m for m in ["roc_auc", "pr_auc", "f1", "brier_score"] if m in metrics.columns]
    for metric_name in metric_names:
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_df = metrics.dropna(subset=[metric_name]).sort_values(metric_name, ascending=(metric_name == "brier_score"))
        ax.bar(plot_df["model"], plot_df[metric_name], color="#426b72")
        ax.set_title(f"Model Comparison: {metric_name}")
        ax.set_ylabel(metric_name)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(figures_dir / f"model_comparison_{metric_name}.png", dpi=160)
        plt.close(fig)
