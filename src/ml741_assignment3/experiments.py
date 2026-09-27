"""Protocol integrity and exact training-work accounting."""

from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

from ml741_assignment3.artifacts import file_sha256


def training_work(epoch_records: list[dict], train_counts: list[int], batch_size: int) -> dict:
    """Count actual minibatch updates and presentations, including rolled-back epochs."""
    presentations = [sum(train_counts[c] for c in row["seen_classes"]) for row in epoch_records]
    return {
        "gradient_updates": sum(math.ceil(n / batch_size) for n in presentations),
        "training_example_presentations": sum(presentations),
        "training_epochs": len(epoch_records),
    }


def verify_lock(root: Path) -> tuple[dict, dict]:
    lock = json.loads((root / "configs/protocol_lock.json").read_text())
    for relative, expected in lock["sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"protocol fingerprint mismatch: {relative}")
    config = yaml.safe_load((root / "configs/final.yaml").read_text())
    if config["status"] != "locked":
        raise ValueError("final protocol is not locked")
    for name, values in config["datasets"].items():
        for relative, expected in (
            (f"data/raw/{name}.zip", values["data"]["archive_sha256"]),
            (f"data/processed/{name}.npz", values["data"]["prepared_sha256"]),
        ):
            if file_sha256(root / relative) != expected:
                raise ValueError(f"dataset fingerprint mismatch: {relative}")
    return config, lock
