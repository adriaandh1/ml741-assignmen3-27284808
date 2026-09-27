"""Bounded validation-only Phase 2 search; restart skips verified complete runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from ml741_assignment3.artifacts import file_sha256, save_run_bundle, write_json, write_records
from ml741_assignment3.controller import ControllerConfig
from ml741_assignment3.datasets import SOURCES, load_prepared, prepare_all
from ml741_assignment3.training import (
    TrainingConfig,
    train_adaptive_incremental,
    train_all_class_baseline,
    train_fixed_incremental,
)

# Small matched search, not a Cartesian hyperparameter sweep. Same candidate
# controls for every method. Thresholds represent explicit target training fit,
# not a requirement that adaptive growth must outperform the baseline.
CANDIDATES = {
    "tanh_fast": dict(
        learning_rate=0.01,
        activation="tanh",
        hidden_units=16,
        weight_decay=0.0001,
        underfit_train_f1=0.95,
    ),
    "tanh_medium": dict(
        learning_rate=0.003,
        activation="tanh",
        hidden_units=16,
        weight_decay=0.0001,
        underfit_train_f1=0.95,
    ),
    "relu_fast": dict(
        learning_rate=0.01,
        activation="relu",
        hidden_units=16,
        weight_decay=0.0001,
        underfit_train_f1=0.95,
    ),
    "tanh_small": dict(
        learning_rate=0.01,
        activation="tanh",
        hidden_units=8,
        weight_decay=0.001,
        underfit_train_f1=0.90,
    ),
}
METHODS = ("all_class_baseline", "fixed_capacity_incremental", "adaptive_capacity_incremental")


def configurations(candidate: str, seed: int) -> tuple[TrainingConfig, ControllerConfig, int]:
    values = CANDIDATES[candidate].copy()
    hidden = values.pop("hidden_units")
    threshold = values.pop("underfit_train_f1")
    training = TrainingConfig(
        **values,
        seed=seed,
        batch_size=128,
        max_epochs=60,
        patience=8,
        min_delta=0.001,
        consolidation_epochs=15,
        evaluate_test=False,
    )
    controller = ControllerConfig(
        min_epochs=8,
        patience=8,
        min_delta=0.001,
        underfit_train_f1=threshold,
        underfit_max_gap=0.08,
        overfit_gap=0.15,
        max_hidden_units=16,
        max_epochs_per_stage=60,
    )
    return training, controller, hidden


def run(root: Path, datasets: list[str], candidates: list[str], seeds: list[int]) -> None:
    torch.set_num_threads(1)
    audit = json.loads((root / "results/pilot/dataset_audit.json").read_text())
    for name in datasets:
        path = root / "data/processed" / f"{name}.npz"
        if file_sha256(path) != audit[name]["prepared_sha256"]:
            raise ValueError("prepared data changed")
        splits = load_prepared(path)
        for candidate in candidates:
            for seed in seeds:
                training, controller, hidden = configurations(candidate, seed)
                configuration = {
                    "dataset": name,
                    "candidate": candidate,
                    "training": asdict(training),
                    "controller": asdict(controller),
                    "hidden_units": hidden,
                    "data": audit[name],
                }
                parent = root / "results/pilot/runs" / name / candidate / str(seed)
                for method in METHODS:
                    destination = parent / method
                    if destination.exists():
                        stored = json.loads((destination / "configuration.json").read_text())
                        if stored != configuration:
                            raise ValueError(f"configuration changed: {destination}")
                        hashes = json.loads((destination / "checksums.json").read_text())
                        if any(file_sha256(destination / p) != h for p, h in hashes.items()):
                            raise ValueError(f"corrupt bundle: {destination}")
                        continue
                    print(f"START {name} {candidate} {seed} {method}", flush=True)
                    if method == METHODS[0]:
                        result = train_all_class_baseline(
                            splits, hidden_units=hidden, config=training
                        )
                    elif method == METHODS[1]:
                        result = train_fixed_incremental(
                            splits, hidden_units=hidden, config=training
                        )
                    else:
                        result = train_adaptive_incremental(
                            splits, config=training, controller_config=controller
                        )
                    if result.metrics["evaluation_partition"] != "validation":
                        raise ValueError("pilot evaluated a forbidden partition")
                    save_run_bundle(result, parent, configuration=configuration)
                    print(
                        f"DONE F1={result.metrics['macro_f1']:.4f} "
                        f"H={result.final_hidden_units} seconds={result.runtime_seconds:.1f}",
                        flush=True,
                    )
                    summarise(root)


def summarise(root: Path) -> list[dict]:
    rows = []
    for path in sorted((root / "results/pilot/runs").glob("*/*/*/*/summary.json")):
        summary = json.loads(path.read_text())
        config = json.loads(path.with_name("configuration.json").read_text())
        epochs = path.with_name("epoch_history.csv").read_text().splitlines()
        rows.append(
            {
                "dataset": config["dataset"],
                "candidate": config["candidate"],
                "seed": summary["seed"],
                "method": summary["method"],
                "macro_f1": summary["metrics"]["macro_f1"],
                "balanced_accuracy": summary["metrics"]["balanced_accuracy"],
                "accuracy": summary["metrics"]["accuracy"],
                "hidden_units": summary["final_hidden_units"],
                "parameters": summary["parameter_count"],
                "epochs": len(epochs) - 1,
                "runtime_seconds": summary["runtime_seconds"],
                "evaluation_partition": summary["metrics"]["evaluation_partition"],
            }
        )
    write_records(root / "results/pilot/search_results.csv", rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--datasets", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument(
        "--candidates", nargs="+", choices=list(CANDIDATES), default=list(CANDIDATES)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[741])
    args = parser.parse_args()
    root = Path.cwd()
    if args.prepare:
        prepare_all(root)
    write_json(
        root / "results/pilot/search_design.json",
        {
            "candidates": CANDIDATES,
            "screening_seeds": [741],
            "confirmation_seeds": [742, 743],
            "rule": "screen on mean validation macro F1 across three methods; confirm top two",
            "test_evaluation": False,
        },
    )
    run(root, args.datasets, args.candidates, args.seeds)
