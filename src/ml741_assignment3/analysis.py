"""Pure, tested statistical operations for the frozen final experiment."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_matrix(frame: pd.DataFrame, datasets: list, methods: list, seeds: list) -> None:
    keys = ["dataset", "method", "seed"]
    require(not frame.duplicated(keys).any(), "duplicate run")
    observed = set(map(tuple, frame[keys].itertuples(index=False, name=None)))
    expected = set(itertools.product(datasets, methods, seeds))
    require(observed == expected, "missing or unexpected run")


def paired_values(frame: pd.DataFrame, a: str, b: str, metric: str) -> np.ndarray:
    selected = frame[frame.method.isin([a, b])]
    require(not selected.duplicated(["method", "seed"]).any(), "duplicate pair")
    pivot = selected.pivot(index="seed", columns="method", values=metric).sort_index()
    require(a in pivot and b in pivot and not pivot.isna().any().any(), "unpaired seeds")
    return (pivot[a] - pivot[b]).to_numpy(float)


def bootstrap_mean(
    values, *, seed: int = 2741, repetitions: int = 10000
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    require(
        values.ndim == 1 and len(values) >= 2 and np.isfinite(values).all(), "invalid sample"
    )
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(axis=1)
    return tuple(float(v) for v in np.quantile(means, [0.025, 0.975]))


def sign_flip(values) -> float:
    values = np.asarray(values, dtype=float)
    require(1 <= len(values) <= 20 and np.isfinite(values).all(), "invalid sign-flip sample")
    signs = np.asarray(list(itertools.product((-1, 1), repeat=len(values))))
    observed = abs(values.mean())
    return float(np.mean(np.abs((signs * values).mean(axis=1)) >= observed - 1e-12))


def holm(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "invalid p-values")
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty_like(p)
    result[order] = adjusted
    return result


def frequency_groups(counts: list[int]) -> tuple[list[int], list[int]]:
    require(len(counts) >= 3 and min(counts) > 0, "invalid class counts")
    n = math.ceil(len(counts) / 3)
    minority = sorted(range(len(counts)), key=lambda c: (counts[c], c))[:n]
    majority = sorted(range(len(counts)), key=lambda c: (-counts[c], c))[:n]
    return minority, majority


def summary_statistics(values) -> dict:
    values = np.asarray(values, dtype=float)
    low, high = bootstrap_mean(values)
    return dict(
        n=len(values),
        mean=float(values.mean()),
        sd=float(values.std(ddof=1)),
        median=float(np.median(values)),
        q1=float(np.quantile(values, 0.25)),
        q3=float(np.quantile(values, 0.75)),
        minimum=float(values.min()),
        maximum=float(values.max()),
        ci_low=low,
        ci_high=high,
    )


def representative_seed(frame: pd.DataFrame) -> int:
    require(len(frame) > 0, "empty condition")
    ordered = frame.sort_values(["macro_f1", "seed"])
    return int(ordered.iloc[(len(ordered) - 1) // 2].seed)
