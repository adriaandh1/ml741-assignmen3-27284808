"""Verify every pilot bundle against validation data without opening test arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_pilot import configurations

from ml741_assignment3.artifacts import (
    file_sha256,
    load_model_checkpoint,
    write_json,
    write_records,
)
from ml741_assignment3.datasets import load_prepared
from ml741_assignment3.experiments import training_work, verify_lock
from ml741_assignment3.metrics import classification_metrics
from ml741_assignment3.training import train_adaptive_incremental


def main() -> None:
    root = Path.cwd()
    config, _ = verify_lock(root)
    torch.set_num_threads(1)
    work = []
    verified = 0
    for dataset in config["datasets"]:
        splits = load_prepared(root / "data/processed" / f"{dataset}.npz")
        for path in sorted((root / "results/pilot/runs" / dataset).glob("*/*/*/summary.json")):
            destination = path.parent
            summary = json.loads(path.read_text())
            run_config = json.loads(path.with_name("configuration.json").read_text())
            hashes = json.loads(path.with_name("checksums.json").read_text())
            assert all(file_sha256(destination / p) == h for p, h in hashes.items())
            assert summary["metrics"]["evaluation_partition"] == "validation"
            assert not run_config["training"]["evaluate_test"]
            model, order = load_model_checkpoint(destination / "checkpoints/final.pt")
            with torch.no_grad():
                pred = np.asarray(order)[model(torch.from_numpy(splits.x_validation)).argmax(1)]
            stored = pd.read_csv(destination / "predictions.csv")
            assert np.array_equal(stored.target, splits.y_validation)
            assert np.array_equal(stored.prediction, pred)
            metric = classification_metrics(splits.y_validation, pred, labels=sorted(order))
            assert metric["macro_f1"] == summary["metrics"]["macro_f1"]
            assert set(order) == set(range(len(splits.label_values)))
            history = pd.read_csv(destination / "epoch_history.csv")
            assert np.isfinite(history[["train_loss", "validation_loss"]]).all().all()
            history["seen_classes"] = history.seen_classes.map(json.loads)
            counts = run_config["data"]["counts"]["train"]
            work.append(
                {
                    "dataset": dataset,
                    "candidate": run_config["candidate"],
                    "method": summary["method"],
                    "seed": summary["seed"],
                    **training_work(
                        history.to_dict("records"), counts, run_config["training"]["batch_size"]
                    ),
                }
            )
            verified += 1
    # Real-data determinism check, not merely checkpoint serialization.
    candidate = config["datasets"]["dry_bean"]["selected_candidate"]
    training, controller, _ = configurations(candidate, 741)
    splits = load_prepared(root / "data/processed/dry_bean.npz")
    result = train_adaptive_incremental(splits, config=training, controller_config=controller)
    destination = root / "results/pilot/runs/dry_bean" / candidate / "741"
    destination /= "adaptive_capacity_incremental"
    model, _ = load_model_checkpoint(destination / "checkpoints/final.pt")
    assert all(
        torch.equal(tensor, model.state_dict()[key])
        for key, tensor in result.model_state.items()
    )
    assert result.events == json.loads((destination / "events.json").read_text())
    assert verified == 72
    write_records(root / "results/pilot/training_work.csv", work)
    write_json(
        root / "results/pilot/verification.json",
        {
            "verified_bundles": verified,
            "checksum_validation": "pass",
            "checkpoint_validation_predictions": "exact_match_all_72",
            "finite_losses": "pass",
            "all_classes_reached": "pass",
            "real_data_repeat": (
                "Dry Bean selected adaptive seed 741: bitwise identical weights/events"
            ),
            "test_arrays_loaded": False,
        },
    )
    print(f"PASS: {verified} pilot bundles and exact real-data adaptive rerun")


if __name__ == "__main__":
    main()
