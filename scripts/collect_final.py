"""Verify all final bundles, reconstruct metrics, and write canonical analysis tables."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
from sklearn.metrics import f1_score

from ml741_assignment3.analysis import frequency_groups, require, validate_matrix
from ml741_assignment3.artifacts import (
    environment_manifest,
    file_sha256,
    load_model_checkpoint,
    write_json,
    write_records,
)
from ml741_assignment3.controller import ControllerConfig
from ml741_assignment3.data import minority_first_order
from ml741_assignment3.datasets import load_prepared
from ml741_assignment3.experiments import training_work, verify_lock
from ml741_assignment3.metrics import average_forgetting, classification_metrics
from ml741_assignment3.training import (
    TrainingConfig,
    train_adaptive_incremental,
    train_all_class_baseline,
    train_fixed_incremental,
)


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    config, lock = verify_lock(root)
    torch.set_num_threads(1)
    rows, classes, stages, recalls, events, ledger = [], [], [], [], [], []
    fingerprints = {}
    expected_paths = {
        root / "results/final/runs" / dataset / str(seed) / method
        for dataset in config["datasets"]
        for seed in config["seeds"]
        for method in config["methods"]
    }
    observed_paths = set((root / "results/final/runs").glob("*/*/*"))
    require(observed_paths == expected_paths, "incomplete or unexpected final run directories")
    for dataset, settings in config["datasets"].items():
        splits = load_prepared(root / "data/processed" / f"{dataset}.npz", include_test=True)
        counts = settings["data"]["counts"]["train"]
        minority, majority = frequency_groups(counts)
        for seed in config["seeds"]:
            for method in config["methods"]:
                identity = dict(dataset=dataset, method=method, seed=seed)
                path = root / "results/final/runs" / dataset / str(seed) / method
                hashes = read(path / "checksums.json")
                for relative, expected in hashes.items():
                    require(
                        file_sha256(path / relative) == expected, f"checksum: {path / relative}"
                    )
                require(read(path / "protocol_lock.json") == lock, "wrong embedded protocol")
                run_config = read(path / "configuration.json")
                require(
                    run_config
                    == {
                        "dataset": dataset,
                        "seed": seed,
                        "settings": settings,
                        "protocol_lock_sha256": file_sha256(
                            root / "configs/protocol_lock.json"
                        ),
                    },
                    "configuration mismatch",
                )
                environment = read(path / "environment.json")
                for name, expected in lock["environment"]["source_sha256"].items():
                    require(
                        environment["source_sha256"][name] == expected, "training source drift"
                    )
                require(
                    environment["packages"] == lock["environment"]["packages"], "package drift"
                )
                for package in ("numpy", "torch", "python"):
                    require(
                        environment[package] == lock["environment"][package], "runtime drift"
                    )
                summary = read(path / "summary.json")
                require(
                    all(summary[k] == identity[k] for k in ("method", "seed")),
                    "identity mismatch",
                )
                require(
                    summary["metrics"]["evaluation_partition"] == "test", "not test results"
                )
                model, order = load_model_checkpoint(path / "checkpoints/final.pt")
                expected_order = (
                    sorted(set(splits.y_train))
                    if method == "all_class_baseline"
                    else minority_first_order(splits.y_train)
                )
                require(order == expected_order == summary["class_order"], "wrong output order")
                with torch.no_grad():
                    logits = model(torch.from_numpy(splits.x_test))
                    predictions = np.asarray(order)[logits.argmax(1).numpy()]
                saved = pd.read_csv(path / "predictions.csv")
                require(
                    np.array_equal(saved.row, np.arange(len(splits.y_test))), "row positions"
                )
                require(np.array_equal(saved.target, splits.y_test), "target mismatch")
                require(np.array_equal(saved.prediction, predictions), "prediction mismatch")
                metrics = classification_metrics(
                    splits.y_test, predictions, labels=sorted(order)
                )
                for key, value in metrics.items():
                    require(value == summary["metrics"][key], f"metric mismatch {key}")
                require(
                    abs(
                        f1_score(splits.y_test, predictions, average="macro")
                        - metrics["macro_f1"]
                    )
                    < 1e-12,
                    "independent F1",
                )
                local_targets = np.asarray([order.index(int(y)) for y in splits.y_test])
                # Stable logit-space CE: probability-based log_loss clips very
                # confident errors and can silently understate their loss.
                logits_array = logits.double().numpy()
                ce = float(
                    np.mean(
                        logsumexp(logits_array, axis=1)
                        - logits_array[np.arange(len(local_targets)), local_targets]
                    )
                )
                torch_ce = torch.nn.functional.cross_entropy(
                    logits.double(), torch.from_numpy(local_targets)
                ).item()
                require(abs(ce - torch_ce) < 1e-9, "cross-entropy mismatch")
                require(
                    model.parameter_count() == summary["parameter_count"], "parameter count"
                )
                require(model.hidden_units == summary["final_hidden_units"], "hidden count")
                history = pd.read_csv(path / "epoch_history.csv")
                history["seen_classes"] = history.seen_classes.map(json.loads)
                require(
                    np.isfinite(history[["train_loss", "validation_loss"]]).all().all(),
                    "nonfinite history",
                )
                require(
                    history.global_epoch.tolist() == list(range(1, len(history) + 1)),
                    "epoch sequence",
                )
                work = training_work(
                    history.to_dict("records"), counts, settings["training"]["batch_size"]
                )
                require(
                    all(summary["metrics"][k] == v for k, v in work.items()), "work mismatch"
                )
                stage_frame = pd.read_csv(path / "stage_history.csv").fillna("")
                stage_recalls = []
                for _, stage in stage_frame.iterrows():
                    per_class = json.loads(stage.per_class)
                    stage_recalls.append({item["label"]: item["recall"] for item in per_class})
                    phase = stage.phase if "phase" in stage_frame else ""
                    phase = phase or "class_stage"
                    stage_history = history[history.stage == stage.stage]
                    require(len(stage_history) == stage.epochs, "stage epochs mismatch")
                    budget = (
                        settings["training"]["consolidation_epochs"]
                        if phase == "consolidation"
                        else settings["training"]["max_epochs"]
                    )
                    require(0 < stage.epochs <= budget, "stage budget violation")
                    if phase == "consolidation":
                        require(stage.epochs == budget, "incomplete consolidation")
                    stages.append(
                        {
                            **identity,
                            "stage": int(stage.stage),
                            "phase": phase,
                            "macro_f1": stage.macro_f1,
                            "epochs": stage.epochs,
                            "hidden_units": int(stage_history.hidden_units.iloc[-1]),
                            "termination": stage.termination,
                            "seen_classes": [item["label"] for item in per_class],
                        }
                    )
                    for item in per_class:
                        recalls.append(
                            {**identity, "stage": int(stage.stage), "phase": phase, **item}
                        )
                require(set(stage_recalls[-1]) == set(order), "not all classes reached")
                forgetting = average_forgetting(stage_recalls)
                require(
                    forgetting == summary["metrics"]["average_forgetting"],
                    "forgetting mismatch",
                )
                event_list = read(path / "events.json")
                growth = [
                    event for event in event_list if event["event"] == "hidden_unit_added"
                ]
                if method == "adaptive_capacity_incremental":
                    require(history.hidden_units.iloc[0] == 0, "adaptive did not start linear")
                    require(
                        model.hidden_units <= settings["controller"]["max_hidden_units"],
                        "capacity cap",
                    )
                    require(len(growth) == model.hidden_units, "growth event mismatch")
                    for i, event in enumerate(growth):
                        require(event["hidden_units_before"] == i, "growth before-count")
                        require(event["hidden_units_after"] == i + 1, "growth after-count")
                        epoch = history[history.global_epoch == event["global_epoch"]].iloc[0]
                        require(epoch.controller_action == "grow", "growth without controller")
                else:
                    require(
                        model.hidden_units == settings["hidden_units"], "fixed width changed"
                    )
                if method != "all_class_baseline":
                    added = [e["class"] for e in event_list if e["event"] == "class_added"]
                    require(added == order[2:], "class introduction sequence")
                    require(len(stage_frame) == len(order), "missing class/consolidation stage")
                for event in event_list:
                    events.append({**identity, **event})
                by_class = {item["label"]: item for item in metrics["per_class"]}
                for label, item in by_class.items():
                    classes.append(
                        {
                            **identity,
                            **item,
                            "class_name": str(splits.label_values[label]),
                            "train_count": counts[label],
                            "minority": label in minority,
                            "majority": label in majority,
                        }
                    )
                rows.append(
                    {
                        **identity,
                        **{
                            k: metrics[k]
                            for k in (
                                "macro_f1",
                                "accuracy",
                                "balanced_accuracy",
                                "macro_precision",
                            )
                        },
                        "cross_entropy": ce,
                        "minority_recall": np.mean([by_class[c]["recall"] for c in minority]),
                        "majority_recall": np.mean([by_class[c]["recall"] for c in majority]),
                        "hidden_units": model.hidden_units,
                        "parameters": model.parameter_count(),
                        "runtime_seconds": summary["runtime_seconds"],
                        **work,
                        "validation_forgetting": forgetting
                        if method != "all_class_baseline"
                        else None,
                    }
                )
                write_records(
                    path / "test_row_audit.csv",
                    [
                        {
                            "row": i,
                            "source_row": int(source),
                            "target": int(target),
                            "prediction": int(pred),
                        }
                        for i, (source, target, pred) in enumerate(
                            zip(splits.test_indices, splits.y_test, predictions, strict=True)
                        )
                    ],
                )
                ledger.append(
                    {
                        **identity,
                        "status": "verified",
                        "attempts_observed": 1,
                        "checkpoint_sha256": file_sha256(path / "checkpoints/final.pt"),
                    }
                )
                for artifact in sorted(path.rglob("*")):
                    if artifact.is_file():
                        fingerprints[artifact.relative_to(root).as_posix()] = file_sha256(
                            artifact
                        )
    validate_matrix(
        pd.DataFrame(rows), list(config["datasets"]), config["methods"], config["seeds"]
    )
    output = root / "results/summaries"
    for name, records in (
        ("runs", rows),
        ("per_class", classes),
        ("stages", stages),
        ("stage_recalls", recalls),
        ("events", events),
        ("run_ledger", ledger),
    ):
        write_records(output / f"{name}.csv", records)
    repeat_results = []
    if args.repeat:
        values = config["datasets"]["dry_bean"]
        splits = load_prepared(root / "data/processed/dry_bean.npz", include_test=True)
        training = TrainingConfig(**values["training"], seed=1001)
        for method in config["methods"]:
            if method == "all_class_baseline":
                result = train_all_class_baseline(
                    splits, hidden_units=values["hidden_units"], config=training
                )
            elif method == "fixed_capacity_incremental":
                result = train_fixed_incremental(
                    splits, hidden_units=values["hidden_units"], config=training
                )
            else:
                result = train_adaptive_incremental(
                    splits,
                    config=training,
                    controller_config=ControllerConfig(**values["controller"]),
                )
            path = root / "results/final/runs/dry_bean/1001" / method
            model, _ = load_model_checkpoint(path / "checkpoints/final.pt")
            require(
                all(
                    torch.equal(v, model.state_dict()[k]) for k, v in result.model_state.items()
                ),
                "rerun weights",
            )
            require(result.events == read(path / "events.json"), "rerun events")
            require(
                np.array_equal(
                    result.predictions, pd.read_csv(path / "predictions.csv").prediction
                ),
                "rerun predictions",
            )
            repeat_results.append(
                dict(method=method, seed=1001, dataset="dry_bean", result="bitwise_match")
            )
        write_json(output / "repeatability.json", repeat_results)
    write_json(
        root / "results/final/completion_manifest.json",
        {
            "expected_runs": 90,
            "verified_runs": len(rows),
            "failures": [],
            "retries": [],
            "sha256": fingerprints,
            "canonical_sha256": {
                f"results/summaries/{name}.csv": file_sha256(output / f"{name}.csv")
                for name in (
                    "runs",
                    "per_class",
                    "stages",
                    "stage_recalls",
                    "events",
                    "run_ledger",
                )
            },
            "protocol_lock_sha256": file_sha256(root / "configs/protocol_lock.json"),
            "environment": environment_manifest(),
            "git_head": subprocess.check_output(
                ["git", "-C", str(root.parent), "rev-parse", "HEAD"], text=True
            ).strip(),
            "git_dirty": bool(
                subprocess.check_output(
                    ["git", "-C", str(root.parent), "status", "--porcelain"], text=True
                )
            ),
        },
    )
    print(
        "Verified 90 final bundles; wrote canonical tables. Repeat checks:", len(repeat_results)
    )


if __name__ == "__main__":
    main()
