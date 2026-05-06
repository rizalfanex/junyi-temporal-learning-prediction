from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from src.common.config import configured_path, ensure_project_dirs, load_config
from src.common.env import environment_summary
from src.common.logging_utils import setup_logger
from src.evaluation.metrics import classification_metrics
from src.evaluation.plots import (
    save_calibration_plot,
    save_confusion_matrix_plot,
    save_roc_pr_curves,
)
from src.models.temporal_nn import TemporalGRU, TemporalTransformer, enable_mc_dropout


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        categorical: np.ndarray,
        numeric: np.ndarray,
        labels: np.ndarray,
        windows: list[tuple[int, int, int]],
        sequence_length: int,
    ) -> None:
        self.categorical = categorical
        self.numeric = numeric
        self.labels = labels
        self.windows = windows
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start, end, target_idx = self.windows[index]
        length = end - start
        cat = np.zeros((self.sequence_length, self.categorical.shape[1]), dtype=np.int64)
        num = np.zeros((self.sequence_length, self.numeric.shape[1]), dtype=np.float32)
        mask = np.zeros(self.sequence_length, dtype=bool)
        cat[:length] = self.categorical[start:end]
        num[:length] = self.numeric[start:end]
        mask[:length] = True
        return {
            "categorical": torch.from_numpy(cat),
            "numeric": torch.from_numpy(num),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(self.labels[target_idx], dtype=torch.float32),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer or GRU sequence model for next-attempt correctness.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--features", default=None)
    parser.add_argument("--model-type", choices=["transformer", "gru"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--mc-passes", type=int, default=None)
    parser.add_argument("--max-train-sequences", type=int, default=None)
    parser.add_argument("--max-val-sequences", type=int, default=None)
    parser.add_argument("--max-test-sequences", type=int, default=None)
    parser.add_argument(
        "--checkpoint-metric",
        choices=["val_loss", "brier_score", "roc_auc", "pr_auc", "f1"],
        default=None,
        help="Validation metric used for selecting the saved checkpoint.",
    )
    return parser.parse_args()


def require_pandas() -> Any:
    if pd is None:
        raise ImportError(
            "pandas is required for sequence model training. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        )
    return pd


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_vocabs(df: Any, categorical_cols: list[str], max_vocab_size: int) -> dict[str, dict[str, int]]:
    train = df[df["split"] == "train"]
    vocabs: dict[str, dict[str, int]] = {}
    for col in categorical_cols:
        values = train[col].astype("string").fillna("unknown").value_counts().head(max_vocab_size - 2)
        vocabs[col] = {str(value): idx + 2 for idx, value in enumerate(values.index)}
    return vocabs


def encode_categoricals(df: Any, categorical_cols: list[str], vocabs: dict[str, dict[str, int]]) -> np.ndarray:
    encoded = np.zeros((len(df), len(categorical_cols)), dtype=np.int64)
    for idx, col in enumerate(categorical_cols):
        mapping = vocabs[col]
        encoded[:, idx] = df[col].astype("string").fillna("unknown").map(mapping).fillna(1).astype("int64").to_numpy()
    return encoded


def normalize_numeric(df: Any, numeric_cols: list[str]) -> tuple[np.ndarray, dict[str, list[float]]]:
    train = df[df["split"] == "train"]
    mean = train[numeric_cols].astype(float).mean().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    std = train[numeric_cols].astype(float).std().replace([np.inf, -np.inf], np.nan).fillna(1.0)
    std = std.mask(std < 1e-6, 1.0)
    values = df[numeric_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    normalized = ((values - mean) / std).clip(-10, 10).to_numpy(dtype=np.float32)
    return normalized, {"mean": mean.tolist(), "std": std.tolist(), "columns": numeric_cols}


def build_windows(
    df: Any,
    split: str,
    sequence_length: int,
    min_history: int,
    max_sequences: int | None,
    seed: int,
) -> list[tuple[int, int, int]]:
    windows: list[tuple[int, int, int]] = []
    splits = df["split"].to_numpy()
    for _, indices in df.groupby("uuid", sort=False).indices.items():
        indices = np.asarray(indices)
        for position in range(min_history, len(indices)):
            target_idx = int(indices[position])
            if splits[target_idx] != split:
                continue
            start_position = max(0, position - sequence_length)
            start_idx = int(indices[start_position])
            end_idx = int(indices[position])
            if end_idx - start_idx <= 0:
                continue
            windows.append((start_idx, end_idx, target_idx))
    if max_sequences and len(windows) > max_sequences:
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(windows), size=max_sequences, replace=False)
        windows = [windows[int(i)] for i in selected]
    return windows


def make_model(config: dict[str, Any], model_type: str, vocab_sizes: list[int], num_numeric: int, sequence_length: int) -> nn.Module:
    seq_cfg = config["sequence"]
    if model_type == "transformer":
        return TemporalTransformer(
            vocab_sizes=vocab_sizes,
            num_numeric_features=num_numeric,
            sequence_length=sequence_length,
            embedding_dim=int(seq_cfg["embedding_dim"]),
            d_model=int(seq_cfg["d_model"]),
            n_heads=int(seq_cfg["n_heads"]),
            num_layers=int(seq_cfg["num_layers"]),
            dropout=float(seq_cfg["dropout"]),
        )
    return TemporalGRU(
        vocab_sizes=vocab_sizes,
        num_numeric_features=num_numeric,
        embedding_dim=int(seq_cfg["embedding_dim"]),
        d_model=int(seq_cfg["d_model"]),
        num_layers=int(seq_cfg["num_layers"]),
        dropout=float(seq_cfg["dropout"]),
    )


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    model.train()
    losses = []
    for batch in tqdm(loader, desc="train", leave=False):
        categorical = batch["categorical"].to(device)
        numeric = batch["numeric"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(categorical, numeric, mask)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else np.nan


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, mc_passes: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true_batches = []
    pass_probs = []
    if mc_passes <= 1:
        model.eval()
        probs = []
        for batch in tqdm(loader, desc="predict", leave=False):
            categorical = batch["categorical"].to(device)
            numeric = batch["numeric"].to(device)
            mask = batch["mask"].to(device)
            logits = model(categorical, numeric, mask)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            y_true_batches.append(batch["label"].numpy())
        return np.concatenate(y_true_batches), np.concatenate(probs), np.zeros(sum(len(p) for p in probs))

    model.eval()
    enable_mc_dropout(model)
    cached_labels = None
    for _ in tqdm(range(mc_passes), desc="mc_dropout", leave=False):
        probs = []
        labels = []
        for batch in loader:
            categorical = batch["categorical"].to(device)
            numeric = batch["numeric"].to(device)
            mask = batch["mask"].to(device)
            logits = model(categorical, numeric, mask)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(batch["label"].numpy())
        pass_probs.append(np.concatenate(probs))
        if cached_labels is None:
            cached_labels = np.concatenate(labels)
    stacked = np.stack(pass_probs, axis=0)
    return cached_labels, stacked.mean(axis=0), stacked.std(axis=0)  # type: ignore[arg-type]


def risk_group(prob: np.ndarray, std: np.ndarray, high_risk: float, low_risk: float, uncertainty_quantile: float) -> list[str]:
    groups = []
    uncertainty_cut = float(np.quantile(std, uncertainty_quantile)) if len(std) else 0.0
    for p, s in zip(prob, std):
        if p < high_risk:
            base = "high_risk"
        elif p >= low_risk:
            base = "low_risk"
        else:
            base = "medium_risk"
        confidence = "uncertain" if s >= uncertainty_cut and uncertainty_cut > 0 else "confident"
        groups.append(f"{base}_{confidence}")
    return groups


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def metric_is_better(metric_name: str, score: float, best_score: float) -> bool:
    if np.isnan(score):
        return False
    if metric_name in {"val_loss", "brier_score"}:
        return score < best_score
    return score > best_score


def initial_best_score(metric_name: str) -> float:
    if metric_name in {"val_loss", "brier_score"}:
        return float("inf")
    return float("-inf")


def main() -> None:
    args = parse_args()
    pandas = require_pandas()
    config = load_config(args.config)
    ensure_project_dirs(config)
    logger = setup_logger("train_transformer")
    seed = int(config["sequence"]["random_seed"])
    seed_everything(seed)

    env = environment_summary()
    logger.info("Environment: %s", env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    features_path = Path(args.features) if args.features else configured_path(config, "processed_dir") / config["preprocessing"]["output_file"]
    if not features_path.is_absolute():
        features_path = ROOT / features_path
    logger.info("Reading features from %s", features_path)
    df = pandas.read_csv(features_path)
    if "timestamp" in df.columns:
        df["timestamp_sort"] = pandas.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values(["uuid", "timestamp_sort", "raw_row_id"], kind="mergesort").reset_index(drop=True)
    else:
        df = df.sort_values(["uuid", "raw_row_id"], kind="mergesort").reset_index(drop=True)

    sequence_length = args.sequence_length or int(config["sequence"]["sequence_length"])
    model_type = args.model_type or config["sequence"]["model_type"]
    epochs = args.epochs or int(config["sequence"]["epochs"])
    mc_passes = args.mc_passes or int(config["uncertainty"]["mc_dropout_passes"])
    checkpoint_metric = args.checkpoint_metric or config["sequence"].get("checkpoint_metric", "val_loss")
    max_train_sequences = args.max_train_sequences if args.max_train_sequences is not None else config["sequence"]["max_train_sequences"]
    max_val_sequences = args.max_val_sequences if args.max_val_sequences is not None else config["sequence"]["max_val_sequences"]
    max_test_sequences = args.max_test_sequences if args.max_test_sequences is not None else config["sequence"]["max_test_sequences"]
    max_train_sequences = None if max_train_sequences == 0 else max_train_sequences
    max_val_sequences = None if max_val_sequences == 0 else max_val_sequences
    max_test_sequences = None if max_test_sequences == 0 else max_test_sequences
    categorical_cols = [col for col in config["features"]["sequence_categorical_columns"] if col in df.columns]
    numeric_cols = [col for col in config["features"]["sequence_numeric_columns"] if col in df.columns]
    target_col = config["dataset"]["correct_col"]

    vocabs = build_vocabs(df, categorical_cols, int(config["sequence"]["max_vocab_size"]))
    categorical = encode_categoricals(df, categorical_cols, vocabs)
    numeric, numeric_stats = normalize_numeric(df, numeric_cols)
    labels = df[target_col].astype("float32").to_numpy()
    vocab_sizes = [max(mapping.values(), default=1) + 1 for mapping in vocabs.values()]

    train_windows = build_windows(
        df,
        "train",
        sequence_length,
        int(config["sequence"]["min_history_length"]),
        max_train_sequences,
        seed,
    )
    val_windows = build_windows(
        df,
        "val",
        sequence_length,
        int(config["sequence"]["min_history_length"]),
        max_val_sequences,
        seed,
    )
    test_windows = build_windows(
        df,
        "test",
        sequence_length,
        int(config["sequence"]["min_history_length"]),
        max_test_sequences,
        seed,
    )
    logger.info("Sequence windows: train=%s, val=%s, test=%s", len(train_windows), len(val_windows), len(test_windows))
    if not train_windows or not val_windows:
        raise ValueError("Not enough sequence windows to train/evaluate. Reduce min_history_length or build a larger feature set.")

    train_ds = SlidingWindowDataset(categorical, numeric, labels, train_windows, sequence_length)
    val_ds = SlidingWindowDataset(categorical, numeric, labels, val_windows, sequence_length)
    test_ds = SlidingWindowDataset(categorical, numeric, labels, test_windows, sequence_length)
    loader_kwargs = {
        "batch_size": int(config["sequence"]["batch_size"]),
        "num_workers": int(config["sequence"]["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = make_model(config, model_type, vocab_sizes, len(numeric_cols), sequence_length).to(device)
    pos_rate = float(labels[[target for _, _, target in train_windows]].mean())
    pos_weight = torch.tensor([(1.0 - pos_rate) / max(pos_rate, 1e-6)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["sequence"]["learning_rate"]),
        weight_decay=float(config["sequence"]["weight_decay"]),
    )

    metrics_rows: list[dict[str, Any]] = []
    best_score = initial_best_score(checkpoint_metric)
    best_epoch = None
    checkpoint_path = configured_path(config, "models_dir") / config["sequence"]["checkpoint_name"]
    if model_type == "gru":
        checkpoint_path = checkpoint_path.with_name(checkpoint_path.name.replace("transformer", "gru"))

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        y_val, p_val, _ = predict(model, val_loader, device, mc_passes=1)
        val_loss = float(nn.functional.binary_cross_entropy(torch.tensor(p_val), torch.tensor(y_val)).item())
        row = classification_metrics(y_val, p_val)
        row.update({"model": model_type, "split": "val", "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        metrics_rows.append(row)
        score = float(val_loss if checkpoint_metric == "val_loss" else row[checkpoint_metric])
        logger.info(
            "Epoch %s: train_loss=%.4f val_loss=%.4f val_auc=%s checkpoint_metric=%s score=%.6f",
            epoch,
            train_loss,
            val_loss,
            row["roc_auc"],
            checkpoint_metric,
            score,
        )
        if metric_is_better(checkpoint_metric, score, best_score):
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "model_type": model_type,
                    "categorical_cols": categorical_cols,
                    "numeric_cols": numeric_cols,
                    "vocabs": vocabs,
                    "numeric_stats": numeric_stats,
                    "vocab_sizes": vocab_sizes,
                    "sequence_length": sequence_length,
                    "checkpoint_metric": checkpoint_metric,
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                },
                checkpoint_path,
            )

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    y_test, p_test, std_test = predict(model, test_loader, device, mc_passes=mc_passes)
    test_metrics = classification_metrics(y_test, p_test)
    test_metrics.update(
        {
            "model": model_type,
            "split": "test",
            "epoch": f"best_{checkpoint.get('best_epoch', 'unknown')}",
            "mc_passes": mc_passes,
            "checkpoint_metric": checkpoint.get("checkpoint_metric", checkpoint_metric),
            "best_validation_score": checkpoint.get("best_score", best_score),
        }
    )
    metrics_rows.append(test_metrics)

    metrics_dir = configured_path(config, "metrics_dir")
    figures_dir = configured_path(config, "figures_dir")
    predictions = pandas.DataFrame(
        {
            "model": model_type,
            "split": "test",
            "y_true": y_test.astype(int),
            "y_prob": p_test,
            "predictive_std": std_test,
            "risk_group": risk_group(
                p_test,
                std_test,
                float(config["uncertainty"]["high_risk_threshold"]),
                float(config["uncertainty"]["low_risk_threshold"]),
                float(config["uncertainty"]["high_uncertainty_quantile"]),
            ),
        }
    )
    pandas.DataFrame(metrics_rows).to_csv(metrics_dir / f"{model_type}_sequence_metrics.csv", index=False)
    predictions.to_csv(metrics_dir / f"{model_type}_sequence_predictions.csv", index=False)
    save_roc_pr_curves(predictions, figures_dir, f"{model_type}_sequence")
    save_calibration_plot(predictions, figures_dir, f"{model_type}_sequence")
    save_confusion_matrix_plot(predictions, figures_dir, f"{model_type}_sequence")
    write_rows(
        metrics_dir / f"{model_type}_risk_group_counts.csv",
        predictions["risk_group"].value_counts().rename_axis("risk_group").reset_index(name="count").to_dict("records"),
    )
    (configured_path(config, "reports_dir") / f"{model_type}_model_card.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "sequence_length": sequence_length,
                "categorical_cols": categorical_cols,
                "numeric_cols": numeric_cols,
                "test_metrics": test_metrics,
                "mc_dropout_passes": mc_passes,
                "high_uncertainty_quantile": float(config["uncertainty"]["high_uncertainty_quantile"]),
                "checkpoint_metric": checkpoint.get("checkpoint_metric", checkpoint_metric),
                "best_validation_score": checkpoint.get("best_score", best_score),
                "best_epoch": checkpoint.get("best_epoch", best_epoch),
                "checkpoint": str(checkpoint_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Saved sequence metrics, predictions, uncertainty groups, and checkpoint")


if __name__ == "__main__":
    main()
