"""Leakage-safe data preparation for multiclass experiments."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

Label = Hashable


@dataclass(frozen=True)
class DatasetSplits:
    """Prepared arrays, original indices, and a training-fitted scaler."""

    x_train: NDArray[np.float32]
    y_train: NDArray[np.int64]
    x_validation: NDArray[np.float32]
    y_validation: NDArray[np.int64]
    x_test: NDArray[np.float32]
    y_test: NDArray[np.int64]
    train_indices: NDArray[np.int64]
    validation_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    scaler_mean: NDArray[np.float64]
    scaler_scale: NDArray[np.float64]
    label_values: tuple[Label, ...]


@dataclass(frozen=True)
class TabularDataset:
    """Validated numeric features and an unmodified target vector."""

    features: NDArray[np.float64]
    targets: NDArray[np.object_]
    feature_names: tuple[str, ...]


def load_numeric_csv(
    path: str | Path,
    *,
    target_column: str,
    drop_columns: Sequence[str] = (),
) -> TabularDataset:
    """Load a numeric tabular classification problem from a local CSV file."""
    frame = pd.read_csv(path)
    if target_column not in frame.columns:
        raise ValueError(f"target column is absent: {target_column}")
    unknown_drops = set(drop_columns) - set(frame.columns)
    if unknown_drops:
        raise ValueError(f"drop columns are absent: {sorted(unknown_drops)}")
    feature_frame = frame.drop(columns=[target_column, *drop_columns])
    if feature_frame.shape[1] < 1:
        raise ValueError("at least one feature column is required")
    non_numeric = feature_frame.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"non-numeric feature columns require encoding: {non_numeric}")
    if feature_frame.isna().any().any() or frame[target_column].isna().any():
        raise ValueError("missing values require an explicit preprocessing decision")
    features = feature_frame.to_numpy(dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError("features contain non-finite values")
    return TabularDataset(
        features=features,
        targets=frame[target_column].to_numpy(dtype=object),
        feature_names=tuple(feature_frame.columns.astype(str)),
    )


def encode_labels(
    labels: Sequence[Label] | NDArray[np.object_],
) -> tuple[NDArray[np.int64], tuple[Label, ...]]:
    """Encode arbitrary labels in a deterministic global order."""
    values = tuple(sorted(set(labels), key=lambda value: (str(type(value)), str(value))))
    if len(values) < 2:
        raise ValueError("at least two target classes are required")
    mapping = {value: index for index, value in enumerate(values)}
    encoded = np.asarray([mapping[value] for value in labels], dtype=np.int64)
    return encoded, values


def minority_first_order(labels: Sequence[int] | NDArray[np.int64]) -> list[int]:
    """Return labels ordered by ascending frequency with deterministic ties."""
    counts = Counter(int(value) for value in labels)
    if len(counts) < 2:
        raise ValueError("at least two observed classes are required")
    return sorted(counts, key=lambda value: (counts[value], value))


def stratified_splits(
    features: NDArray[np.floating] | Sequence[Sequence[float]],
    labels: Sequence[Label] | NDArray[np.object_],
    *,
    seed: int,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.2,
    standardize: bool = True,
) -> DatasetSplits:
    """Create train, validation, and test partitions without preprocessing leakage."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 10:
        raise ValueError("features must be a two-dimensional array with at least 10 rows")
    if not np.isfinite(x).all():
        raise ValueError("features contain non-finite values")
    y, label_values = encode_labels(labels)
    if x.shape[0] != y.shape[0]:
        raise ValueError("feature and target row counts differ")
    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must lie between zero and one")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("test and validation fractions leave no training data")

    indices = np.arange(x.shape[0], dtype=np.int64)
    development_indices, test_indices = train_test_split(
        indices,
        test_size=test_fraction,
        random_state=seed,
        stratify=y,
    )
    relative_validation_fraction = validation_fraction / (1.0 - test_fraction)
    train_indices, validation_indices = train_test_split(
        development_indices,
        test_size=relative_validation_fraction,
        random_state=seed + 1,
        stratify=y[development_indices],
    )

    scaler = StandardScaler(with_mean=True, with_std=True)
    if standardize:
        scaler.fit(x[train_indices])
        x_train = scaler.transform(x[train_indices])
        x_validation = scaler.transform(x[validation_indices])
        x_test = scaler.transform(x[test_indices])
        scaler_mean = scaler.mean_.copy()
        scaler_scale = scaler.scale_.copy()
    else:
        x_train = x[train_indices].copy()
        x_validation = x[validation_indices].copy()
        x_test = x[test_indices].copy()
        scaler_mean = np.zeros(x.shape[1], dtype=np.float64)
        scaler_scale = np.ones(x.shape[1], dtype=np.float64)

    return DatasetSplits(
        x_train=x_train.astype(np.float32),
        y_train=y[train_indices],
        x_validation=x_validation.astype(np.float32),
        y_validation=y[validation_indices],
        x_test=x_test.astype(np.float32),
        y_test=y[test_indices],
        train_indices=train_indices.copy(),
        validation_indices=validation_indices.copy(),
        test_indices=test_indices.copy(),
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        label_values=label_values,
    )


def controlled_long_tail_indices(
    labels: Sequence[int] | NDArray[np.int64],
    *,
    maximum_to_minimum_ratio: float,
    seed: int,
    class_order: Sequence[int] | None = None,
) -> NDArray[np.int64]:
    """Downsample a training target to a reproducible exponential long tail."""
    y = np.asarray(labels, dtype=np.int64)
    classes = list(class_order) if class_order is not None else sorted(np.unique(y).tolist())
    if len(classes) != len(set(classes)) or set(classes) != set(np.unique(y).tolist()):
        raise ValueError("class_order must contain every observed class exactly once")
    if not np.isfinite(maximum_to_minimum_ratio) or maximum_to_minimum_ratio < 1:
        raise ValueError("maximum_to_minimum_ratio must be at least one")
    maximum = min(int(np.sum(y == label)) for label in classes)
    if maximum < 2:
        raise ValueError("every class requires at least two observations")
    targets = np.geomspace(maximum, maximum / maximum_to_minimum_ratio, len(classes))
    targets = np.maximum(2, np.floor(targets).astype(int))
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label, target in zip(classes, targets, strict=True):
        available = np.flatnonzero(y == label)
        count = min(int(target), available.size)
        selected.extend(rng.choice(available, size=count, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def make_smoke_dataset(
    *, seed: int = 741, samples: int = 480
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Build a small, learnable, imbalanced four-class dataset."""
    features, labels = make_classification(
        n_samples=samples,
        n_features=10,
        n_informative=8,
        n_redundant=1,
        n_classes=4,
        n_clusters_per_class=1,
        weights=[0.50, 0.25, 0.15, 0.10],
        class_sep=1.25,
        flip_y=0.01,
        random_state=seed,
    )
    return features.astype(np.float64), labels.astype(np.int64)
