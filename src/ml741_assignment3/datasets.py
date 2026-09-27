"""UCI acquisition and auditable, training-only preparation for the real pilots."""

from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml741_assignment3.artifacts import file_sha256, write_json
from ml741_assignment3.data import DatasetSplits, controlled_long_tail_indices, encode_labels

SOURCES = {
    "dry_bean": "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip",
    "landsat": "https://archive.ics.uci.edu/static/public/146/statlog+landsat+satellite.zip",
    "optdigits": (
        "https://archive.ics.uci.edu/static/public/80/"
        "optical+recognition+of+handwritten+digits.zip"
    ),
}


def acquire(name: str, raw: Path) -> tuple[np.ndarray, np.ndarray, int | None, dict]:
    """Download once, parse in memory (never extract archive paths), fingerprint."""
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / f"{name}.zip"
    if not path.exists():
        request = urllib.request.Request(SOURCES[name], headers={"User-Agent": "ML741/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read()
        # Validate the complete response before installing a cached archive.
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if archive.testzip() is not None:
                raise ValueError("corrupt source archive")
        path.write_bytes(payload)
    with zipfile.ZipFile(path) as archive:
        if name == "dry_bean":
            member = next(p for p in archive.namelist() if p.endswith(".arff"))
            records, _ = arff.loadarff(io.StringIO(archive.read(member).decode()))
            columns = records.dtype.names
            x = np.column_stack([records[c] for c in columns[:-1]]).astype(float)
            y = np.asarray([v.decode() for v in records[columns[-1]]])
            boundary = None
        else:
            train_file, test_file = (
                ("sat.trn", "sat.tst")
                if name == "landsat"
                else ("optdigits.tra", "optdigits.tes")
            )
            arrays = []
            for filename in (train_file, test_file):
                member = next(p for p in archive.namelist() if p.split("/")[-1] == filename)
                arrays.append(
                    np.loadtxt(
                        io.BytesIO(archive.read(member)),
                        delimiter="," if name == "optdigits" else None,
                    )
                )
            boundary = len(arrays[0])
            all_rows = np.vstack(arrays)
            x, y = all_rows[:, :-1], all_rows[:, -1].astype(int)
    return x, y, boundary, {"url": SOURCES[name], "archive_sha256": file_sha256(path)}


def prepare(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 741,
    official_boundary: int | None = None,
    long_tail_ratio: float | None = None,
) -> tuple[DatasetSplits, dict]:
    """Deduplicate identical features; reserve official holdouts preferentially.

    Conflicting labels on identical features are rejected, not silently resolved.
    With official splits, retain the first held-out copy and remove any training
    copies. Otherwise retain the first copy before stratified 60/20/20 splitting.
    No row is moved out of an official holdout into development.
    """
    x = np.asarray(x, dtype=np.float64)
    y, label_values = encode_labels(labels)
    if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError("invalid features or targets")
    if official_boundary is not None and not 0 < official_boundary < len(x):
        raise ValueError("invalid official boundary")
    _, inverse = np.unique(x, axis=0, return_inverse=True)
    groups: dict[int, list[int]] = {}
    for i, group in enumerate(inverse):
        groups.setdefault(int(group), []).append(i)
    keep = []
    cross = 0
    for indices in groups.values():
        if len(set(y[indices])) != 1:
            raise ValueError("identical features have conflicting labels")
        heldout = (
            [] if official_boundary is None else [i for i in indices if i >= official_boundary]
        )
        cross += bool(heldout and any(i < official_boundary for i in indices))
        keep.append(heldout[0] if heldout else indices[0])
    keep = np.asarray(sorted(keep), dtype=np.int64)
    if official_boundary is None:
        development, test = train_test_split(
            keep, test_size=0.2, stratify=y[keep], random_state=seed
        )
    else:
        development, test = keep[keep < official_boundary], keep[keep >= official_boundary]
    train, validation = train_test_split(
        development, test_size=0.25, stratify=y[development], random_state=seed + 1
    )
    before_tail = np.bincount(y[train], minlength=len(label_values))
    if long_tail_ratio is not None:
        selected = controlled_long_tail_indices(
            y[train],
            maximum_to_minimum_ratio=long_tail_ratio,
            seed=seed + 2,
            class_order=list(range(len(label_values))),
        )
        train = train[selected]
    scaler = StandardScaler().fit(x[train])
    splits = DatasetSplits(
        scaler.transform(x[train]).astype(np.float32),
        y[train],
        scaler.transform(x[validation]).astype(np.float32),
        y[validation],
        scaler.transform(x[test]).astype(np.float32),
        y[test],
        train,
        validation,
        test,
        scaler.mean_,
        scaler.scale_,
        label_values,
    )
    counts = {
        partition: np.bincount(y[idx], minlength=len(label_values)).tolist()
        for partition, idx in (("train", train), ("validation", validation), ("test", test))
    }
    if min(counts["test"]) < 20 or min(counts["validation"]) < 20:
        raise ValueError("insufficient per-class evaluation support")
    audit = {
        "original_rows": len(x),
        "features": x.shape[1],
        "classes": len(label_values),
        "labels": [str(v) for v in label_values],
        "duplicate_rows_removed": len(x) - len(keep),
        "cross_official_duplicate_groups": cross,
        "split_seed": seed,
        "official_boundary": official_boundary,
        "long_tail_ratio": long_tail_ratio,
        "train_counts_before_long_tail": before_tail.tolist(),
        "counts": counts,
        "train_imbalance_ratio": max(counts["train"]) / min(counts["train"]),
        "constant_training_features": int(np.sum(np.var(x[train], axis=0) == 0)),
        "missing_values": 0,
        "feature_duplicate_overlap_between_partitions": 0,
    }
    return splits, audit


def prepare_all(root: Path) -> dict[str, dict]:
    audits = {}
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        x, y, boundary, source = acquire(name, root / "data" / "raw")
        splits, audit = prepare(
            x,
            y,
            official_boundary=boundary,
            long_tail_ratio=10 if name == "optdigits" else None,
        )
        audit.update(source)
        destination = processed / f"{name}.npz"
        np.savez_compressed(destination, **asdict(splits))
        audit["prepared_sha256"] = file_sha256(destination)
        audits[name] = audit
        print(name, audit, flush=True)
    write_json(root / "results" / "pilot" / "dataset_audit.json", audits)
    return audits


def load_prepared(path: Path, *, include_test: bool = False) -> DatasetSplits:
    """Pilots do not even load the held-out feature/target arrays."""
    with np.load(path, allow_pickle=False) as archive:
        values = {
            key: (None if key in ("x_test", "y_test") and not include_test else archive[key])
            for key in DatasetSplits.__dataclass_fields__
        }
    values["label_values"] = tuple(values["label_values"].tolist())
    return DatasetSplits(**values)
