"""Run the frozen validation-only adaptive-controller sensitivity check."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import torch
import yaml

from ml741_assignment3.controller import ControllerConfig
from ml741_assignment3.datasets import load_prepared
from ml741_assignment3.experiments import training_work
from ml741_assignment3.training import TrainingConfig, train_adaptive_incremental

SEEDS = (741, 742, 743)
VARIANTS = {
    "nominal": {},
    "fit_090": {"underfit_train_f1": 0.90},
    "fit_098": {"underfit_train_f1": 0.98},
    "patience_6": {"min_epochs": 6, "patience": 6},
    "tolerance_003": {"min_delta": 0.003},
    "overfit_010": {"overfit_gap": 0.10},
    "growth_cap_8": {"max_hidden_units": 8},
    "epoch_cap_45": {"max_epochs_per_stage": 45},
}


def main() -> None:
    root = Path.cwd()
    final = yaml.safe_load((root / "configs/final.yaml").read_text())
    output = root / "results/controller-sensitivity"
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    rows: list[dict] = []
    for dataset, values in final["datasets"].items():
        splits = load_prepared(root / "data/processed" / f"{dataset}.npz")
        base_training = {
            key: value
            for key, value in values["training"].items()
            if key != "evaluate_test"
        }
        base_training["evaluate_test"] = False
        nominal = ControllerConfig(**values["controller"])
        for variant, changes in VARIANTS.items():
            controller = replace(nominal, **changes)
            for seed in SEEDS:
                training = TrainingConfig(**base_training, seed=seed)
                result = train_adaptive_incremental(
                    splits,
                    config=training,
                    controller_config=controller,
                )
                if result.metrics["evaluation_partition"] != "validation":
                    raise RuntimeError("sensitivity analysis accessed the test partition")
                work = training_work(
                    result.epoch_records,
                    values["data"]["counts"]["train"],
                    training.batch_size,
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "seed": seed,
                        "validation_macro_f1": result.metrics["macro_f1"],
                        "hidden_units": result.final_hidden_units,
                        "parameters": result.parameter_count,
                        **work,
                    }
                )
                print(
                    f"{dataset} {variant} {seed}: "
                    f"F1={result.metrics['macro_f1']:.4f}, H={result.final_hidden_units}",
                    flush=True,
                )

    runs = pd.DataFrame(rows).sort_values(["dataset", "variant", "seed"])
    summary = (
        runs.groupby(["dataset", "variant"], as_index=False)
        .agg(
            validation_macro_f1_mean=("validation_macro_f1", "mean"),
            validation_macro_f1_sd=("validation_macro_f1", "std"),
            hidden_units_mean=("hidden_units", "mean"),
            hidden_units_min=("hidden_units", "min"),
            hidden_units_max=("hidden_units", "max"),
            parameters_mean=("parameters", "mean"),
            training_epochs_mean=("training_epochs", "mean"),
            gradient_updates_mean=("gradient_updates", "mean"),
        )
        .sort_values(["dataset", "variant"])
    )
    runs.to_csv(output / "runs.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "design.json").write_text(
        json.dumps(
            {
                "purpose": "validation-only one-factor controller sensitivity check",
                "datasets": list(final["datasets"]),
                "seeds": list(SEEDS),
                "variants": VARIANTS,
                "nominal_controller": asdict(
                    ControllerConfig(**final["datasets"]["dry_bean"]["controller"])
                ),
                "test_evaluation": False,
                "reporting_rule": (
                    "report score and capacity ranges; do not alter the locked final protocol"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
