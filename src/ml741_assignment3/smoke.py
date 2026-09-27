"""End-to-end synthetic smoke experiment."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ml741_assignment3.artifacts import save_run_bundle, write_json
from ml741_assignment3.config import load_yaml, require_keys
from ml741_assignment3.controller import ControllerConfig
from ml741_assignment3.data import make_smoke_dataset, stratified_splits
from ml741_assignment3.plots import (
    plot_confusion_matrix,
    plot_incremental_progress,
    plot_method_comparison,
)
from ml741_assignment3.training import (
    RunResult,
    TrainingConfig,
    train_adaptive_incremental,
    train_all_class_baseline,
    train_fixed_incremental,
)


def _default_configuration(seed: int) -> dict[str, Any]:
    return {
        "kind": "synthetic_smoke_test",
        "seed": seed,
        "samples": 480,
        "baseline_hidden_units": 4,
        "fixed_hidden_units": 4,
        "training": {
            "learning_rate": 0.02,
            "weight_decay": 0.0001,
            "batch_size": 32,
            "max_epochs": 18,
            "patience": 4,
            "min_delta": 0.002,
            "activation": "tanh",
            "seed": seed,
            "evaluate_test": True,
        },
        "controller": {
            "min_epochs": 3,
            "patience": 3,
            "min_delta": 0.003,
            "underfit_train_f1": 0.98,
            "underfit_max_gap": 0.20,
            "overfit_gap": 0.30,
            "max_hidden_units": 3,
            "max_epochs_per_stage": 16,
        },
    }


def run_smoke(
    output_directory: str | Path,
    *,
    seed: int | None = None,
    config_path: str | Path | None = None,
) -> list[RunResult]:
    """Run all methods on synthetic data and create an inspectable artifact bundle."""
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    raw_config = (
        load_yaml(config_path) if config_path is not None else _default_configuration(741)
    )
    require_keys(
        raw_config,
        "kind",
        "samples",
        "baseline_hidden_units",
        "fixed_hidden_units",
        "training",
        "controller",
    )
    seed = int(raw_config.get("seed", 741)) if seed is None else seed
    raw_config["seed"] = seed
    raw_config["training"]["seed"] = seed
    features, targets = make_smoke_dataset(seed=seed, samples=int(raw_config["samples"]))
    splits = stratified_splits(features, targets, seed=seed)
    write_json(
        output / "data_manifest.json",
        {
            "source": "sklearn.make_classification",
            "source_sha256": hashlib.sha256(features.tobytes() + targets.tobytes()).hexdigest(),
            "seed": seed,
            "train_indices": splits.train_indices,
            "validation_indices": splits.validation_indices,
            "test_indices": splits.test_indices,
            "scaler_mean": splits.scaler_mean,
            "scaler_scale": splits.scaler_scale,
            "label_values": splits.label_values,
        },
    )
    training = TrainingConfig(**raw_config["training"])
    controller = ControllerConfig(**raw_config["controller"])
    results = [
        train_all_class_baseline(
            splits,
            hidden_units=int(raw_config["baseline_hidden_units"]),
            config=training,
        ),
        train_fixed_incremental(
            splits,
            hidden_units=int(raw_config["fixed_hidden_units"]),
            config=training,
        ),
        train_adaptive_incremental(
            splits,
            config=training,
            controller_config=controller,
        ),
    ]
    configuration: dict[str, Any] = {
        **raw_config,
        "training": asdict(training),
        "controller": asdict(controller),
        "test_partition_used_for_development": True,
    }
    for result in results:
        save_run_bundle(result, output / "runs", configuration=configuration)
        plot_confusion_matrix(
            result,
            output / "figures" / f"confusion_matrix_{result.method}.png",
        )
    adaptive = next(
        result for result in results if result.method == "adaptive_capacity_incremental"
    )
    plot_incremental_progress(adaptive, output / "figures" / "adaptive_progress.png")
    plot_method_comparison(results, output / "figures" / "method_comparison.png")
    write_json(
        output / "summary.json",
        {
            "configuration": configuration,
            "methods": [result.summary() for result in results],
        },
    )
    return results
