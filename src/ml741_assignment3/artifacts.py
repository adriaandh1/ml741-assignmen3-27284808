"""Transparent, machine-readable experiment artifact bundles."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml741_assignment3.model import AdditiveGrowingNetwork
from ml741_assignment3.training import RunResult


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in record.items()
                }
            )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_manifest() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "packages": {
            name: version(name)
            for name in ("pandas", "scikit-learn", "scipy", "matplotlib", "PyYAML")
        },
        "source_sha256": {
            path.name: file_sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
    }


def save_run_bundle(
    result: RunResult,
    output_directory: str | Path,
    *,
    configuration: dict[str, Any],
) -> Path:
    """Save metrics, histories, predictions, provenance, and a checkpoint."""
    destination = Path(output_directory) / result.method
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "summary.json", result.summary())
    write_json(destination / "configuration.json", configuration)
    write_json(destination / "environment.json", environment_manifest())
    write_json(destination / "events.json", result.events)
    write_records(destination / "epoch_history.csv", result.epoch_records)
    write_records(destination / "stage_history.csv", result.stage_records)
    write_records(
        destination / "predictions.csv",
        [
            {"row": index, "target": int(target), "prediction": int(prediction)}
            for index, (target, prediction) in enumerate(
                zip(result.targets, result.predictions, strict=True)
            )
        ],
    )
    checkpoint = destination / "checkpoints" / "final.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": result.model_state,
            "shape": result.model_shape,
            "class_order": result.class_order,
        },
        checkpoint,
    )
    checksummed = [
        destination / "summary.json",
        destination / "configuration.json",
        destination / "environment.json",
        destination / "events.json",
        destination / "epoch_history.csv",
        destination / "stage_history.csv",
        destination / "predictions.csv",
        checkpoint,
    ]
    write_json(
        destination / "checksums.json",
        {path.relative_to(destination).as_posix(): file_sha256(path) for path in checksummed},
    )
    return destination


def load_model_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[AdditiveGrowingNetwork, list[int]]:
    """Reconstruct a growing network and label order from a saved checkpoint."""
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    shape = payload["shape"]
    model = AdditiveGrowingNetwork(
        int(shape["input_features"]),
        int(shape["output_classes"]),
        activation=str(shape["activation"]),
    ).to(device)
    model.add_hidden_units(
        int(shape["hidden_units"]),
        preserve_function=True,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, [int(label) for label in payload["class_order"]]
