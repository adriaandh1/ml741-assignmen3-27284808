"""Smoke-test and report-ready diagnostic plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ml741_assignment3.training import RunResult


def plot_confusion_matrix(result: RunResult, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(result.metrics["confusion_matrix"], dtype=float)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals > 0)
    figure, axis = plt.subplots(figsize=(4.8, 4.0))
    image = axis.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    labels = sorted(np.unique(result.targets).tolist())
    axis.set_xticks(range(len(labels)), labels)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(result.method.replace("_", " ").title())
    figure.colorbar(image, ax=axis, label="Row-normalised proportion")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def plot_incremental_progress(result: RunResult, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, primary = plt.subplots(figsize=(6.8, 4.0))
    epochs = [int(record["global_epoch"]) for record in result.epoch_records]
    validation = [float(record["validation_macro_f1"]) for record in result.epoch_records]
    hidden = [int(record["hidden_units"]) for record in result.epoch_records]
    primary.plot(
        epochs,
        validation,
        color="#1f77b4",
        linewidth=1.8,
        label="Validation macro F1",
    )
    primary.set_xlabel("Global epoch")
    primary.set_ylabel("Validation macro F1", color="#1f77b4")
    primary.set_ylim(0, 1.02)
    secondary = primary.twinx()
    secondary.step(epochs, hidden, where="post", color="#d62728", label="Hidden units")
    secondary.set_ylabel("Hidden units", color="#d62728")
    for event in result.events:
        if event.get("event") == "class_added":
            stage = int(event["stage"])
            matching = [
                int(record["global_epoch"])
                for record in result.epoch_records
                if int(record["stage"]) == stage
            ]
            if matching:
                primary.axvline(min(matching), color="0.65", linestyle="--", linewidth=0.8)
    primary.set_title("Adaptive incremental smoke-test trajectory")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def plot_method_comparison(results: list[RunResult], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    labels = [result.method.replace("_", "\n") for result in results]
    values = [float(result.metrics["macro_f1"]) for result in results]
    figure, axis = plt.subplots(figsize=(7.0, 3.8))
    bars = axis.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b"])
    axis.set_ylabel("Test macro F1")
    axis.set_ylim(0, 1.02)
    axis.set_title("Synthetic smoke-test comparison")
    axis.bar_label(bars, fmt="%.3f", padding=3)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination
