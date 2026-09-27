"""Phase 3 launcher: preflight by default; held-out evaluation requires --execute."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ml741_assignment3.artifacts import file_sha256, save_run_bundle, write_json
from ml741_assignment3.controller import ControllerConfig
from ml741_assignment3.datasets import load_prepared
from ml741_assignment3.experiments import training_work, verify_lock
from ml741_assignment3.training import (
    TrainingConfig,
    train_adaptive_incremental,
    train_all_class_baseline,
    train_fixed_incremental,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    config, lock = verify_lock(root)
    total = len(config["datasets"]) * len(config["methods"]) * len(config["seeds"])
    if total != config["expected_runs"]:
        raise ValueError("unexpected run matrix")
    print(f"Protocol verified: {total} paired runs, test evaluation {args.execute}", flush=True)
    if not args.execute:
        return
    torch.set_num_threads(1)
    for name, values in config["datasets"].items():
        splits = load_prepared(root / "data/processed" / f"{name}.npz", include_test=True)
        for seed in config["seeds"]:
            training = TrainingConfig(**values["training"], seed=seed)
            controller = ControllerConfig(**values["controller"])
            for method in config["methods"]:
                parent = root / "results/final/runs" / name / str(seed)
                destination = parent / method
                run_config = {
                    "dataset": name,
                    "seed": seed,
                    "settings": values,
                    "protocol_lock_sha256": file_sha256(root / "configs/protocol_lock.json"),
                }
                if destination.exists():
                    stored = json.loads((destination / "configuration.json").read_text())
                    checksums = json.loads((destination / "checksums.json").read_text())
                    if stored != run_config or any(
                        file_sha256(destination / p) != h for p, h in checksums.items()
                    ):
                        raise ValueError(f"invalid existing run: {destination}")
                    continue
                print(f"START {name} {seed} {method}", flush=True)
                if method == "all_class_baseline":
                    result = train_all_class_baseline(
                        splits, hidden_units=values["hidden_units"], config=training
                    )
                elif method == "fixed_capacity_incremental":
                    result = train_fixed_incremental(
                        splits, hidden_units=values["hidden_units"], config=training
                    )
                elif method == "adaptive_capacity_incremental":
                    result = train_adaptive_incremental(
                        splits, config=training, controller_config=controller
                    )
                else:
                    raise ValueError(f"unknown method: {method}")
                result.metrics.update(
                    training_work(
                        result.epoch_records,
                        values["data"]["counts"]["train"],
                        training.batch_size,
                    )
                )
                save_run_bundle(result, parent, configuration=run_config)
                write_json(destination / "protocol_lock.json", lock)
                print(f"DONE {name} {seed} {method}", flush=True)


if __name__ == "__main__":
    main()
