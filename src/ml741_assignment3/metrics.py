"""Class-balanced metrics and incremental-learning diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_metrics(
    targets: Sequence[int] | NDArray[np.int64],
    predictions: Sequence[int] | NDArray[np.int64],
    *,
    labels: Sequence[int],
) -> dict[str, object]:
    """Calculate aggregate and per-class measures with explicit label support."""
    y_true = np.asarray(targets, dtype=np.int64)
    y_pred = np.asarray(predictions, dtype=np.int64)
    label_array = np.asarray(labels, dtype=np.int64)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError("targets and predictions must be matching one-dimensional arrays")
    if y_true.size == 0:
        raise ValueError("metrics require at least one observation")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_array,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": [
            {
                "label": int(label),
                "precision": float(class_precision),
                "recall": float(class_recall),
                "f1": float(class_f1),
                "support": int(class_support),
            }
            for label, class_precision, class_recall, class_f1, class_support in zip(
                label_array,
                precision,
                recall,
                f1,
                support,
                strict=True,
            )
        ],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_array).tolist(),
    }


def average_forgetting(stage_recalls: Sequence[dict[int, float]]) -> float:
    """Measure mean loss from each class's best earlier recall to final recall."""
    if len(stage_recalls) < 2:
        return 0.0
    final = stage_recalls[-1]
    losses: list[float] = []
    for label, final_recall in final.items():
        earlier = [stage[label] for stage in stage_recalls[:-1] if label in stage]
        if earlier:
            losses.append(max(0.0, max(earlier) - final_recall))
    return float(np.mean(losses)) if losses else 0.0
