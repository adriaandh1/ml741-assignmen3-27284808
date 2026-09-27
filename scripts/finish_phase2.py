"""Confirm top two screened settings, plot validation evidence, and lock Phase 3."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from run_pilot import METHODS, configurations, run, summarise

from ml741_assignment3.artifacts import environment_manifest, file_sha256, write_json


def main() -> None:
    root = Path.cwd()
    lock_path = root / "configs/protocol_lock.json"
    if lock_path.exists():
        raise FileExistsError("protocol already locked; do not silently replace it")
    frame = pd.DataFrame(summarise(root))
    screen = frame[frame.seed == 741].groupby(["dataset", "candidate"]).macro_f1.mean()
    if len(screen) != 12 or len(frame[frame.seed == 741]) != 36:
        raise ValueError("complete the 36-run screening first")
    shortlist = {}
    for dataset in sorted(frame.dataset.unique()):
        shortlist[dataset] = (
            screen.loc[dataset].sort_values(ascending=False).head(2).index.tolist()
        )
    write_json(root / "results/pilot/confirmation_shortlist.json", shortlist)
    for dataset, candidates in shortlist.items():
        run(root, [dataset], candidates, [742, 743])
    frame = pd.DataFrame(summarise(root))
    selected = {}
    for dataset, candidates in shortlist.items():
        subset = frame[(frame.dataset == dataset) & frame.candidate.isin(candidates)]
        scores = subset.groupby("candidate").macro_f1.mean().sort_values(ascending=False)
        selected[dataset] = str(scores.index[0])
    selected_frame = pd.concat(
        [
            frame[(frame.dataset == dataset) & (frame.candidate == candidate)]
            for dataset, candidate in selected.items()
        ]
    )
    selected_frame.to_csv(root / "results/pilot/selected_runs.csv", index=False)
    aggregates = selected_frame.groupby(["dataset", "method"]).agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_sd=("macro_f1", "std"),
        hidden_mean=("hidden_units", "mean"),
        hidden_min=("hidden_units", "min"),
        hidden_max=("hidden_units", "max"),
        runtime_mean=("runtime_seconds", "mean"),
        parameter_mean=("parameters", "mean"),
    )
    aggregates.to_csv(root / "results/pilot/selected_summary.csv")
    audit = json.loads((root / "results/pilot/dataset_audit.json").read_text())
    figures = root / "results/figures/pilot"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#34699a", "#e5a33d", "#45936d"]
    names = list(selected)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, dataset in zip(axes, names, strict=True):
        data = aggregates.loc[dataset].reindex(METHODS)
        ax.bar(range(3), data.macro_f1_mean, yerr=data.macro_f1_sd, color=colors, capsize=4)
        ax.set(
            xticks=range(3),
            xticklabels=["All-class", "Fixed inc.", "Adaptive"],
            ylim=(0, 1),
            title=dataset.replace("_", " ").title(),
            ylabel="Validation macro F1",
        )
        for i, score in enumerate(data.macro_f1_mean):
            ax.text(i, score + 0.03, f"{score:.3f}", ha="center")
    fig.suptitle("Pilot only: mean ± sample SD over 3 optimisation seeds")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(figures / f"validation_comparison.{extension}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, dataset in zip(axes, names, strict=True):
        info = audit[dataset]
        ax.bar(info["labels"], info["counts"]["train"], color=colors[0])
        ax.tick_params(axis="x", rotation=55)
        ax.set(title=dataset.replace("_", " ").title(), ylabel="Retained training examples")
    fig.tight_layout()
    fig.savefig(figures / "training_class_counts.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex="col")
    for column, dataset in enumerate(names):
        candidate = selected[dataset]
        for seed in (741, 742, 743):
            path = root / "results/pilot/runs" / dataset / candidate / str(seed) / METHODS[2]
            history = pd.read_csv(path / "epoch_history.csv")
            stages = pd.read_csv(path / "stage_history.csv")
            if len(json.loads(stages.iloc[-1].per_class)) != audit[dataset]["classes"]:
                raise ValueError("adaptive run did not reach all classes")
            axes[0, column].plot(
                history.global_epoch, history.validation_macro_f1, label=str(seed), alpha=0.8
            )
            axes[1, column].step(history.global_epoch, history.hidden_units, where="post")
        axes[0, column].set(title=dataset, ylabel="Seen-class validation F1", ylim=(0, 1.02))
        axes[1, column].set(
            xlabel="Cumulative training epoch", ylabel="Hidden units", ylim=(0, 17)
        )
    axes[0, 0].legend(title="Seed", fontsize=8)
    fig.suptitle("Adaptive pilot trajectories (class sets change between stages)")
    fig.tight_layout()
    fig.savefig(figures / "adaptive_trajectories.png", dpi=180)
    plt.close(fig)
    final = {
        "phase": "final",
        "status": "locked",
        "version": 1,
        "primary_dataset": "dry_bean",
        "methods": list(METHODS),
        "seeds": list(range(1001, 1011)),
        "split_seed": 741,
        "paired_splits": True,
        "repetitions": 10,
        "expected_runs": 90,
        "primary_metric": "macro_f1",
        "datasets": {},
        "uncertainty_scope": (
            "optimisation variability conditional on one fixed split per dataset"
        ),
        "test_evaluation": "only with explicit --execute flag after lock preflight",
    }
    for dataset, candidate in selected.items():
        training, controller, hidden = configurations(candidate, 1001)
        controls = asdict(training)
        controls.pop("seed")
        controls["evaluate_test"] = True
        final["datasets"][dataset] = {
            "selected_candidate": candidate,
            "training": controls,
            "controller": asdict(controller),
            "hidden_units": hidden,
            "data": audit[dataset],
        }
    # YAML is generated configuration, not a manually authored source edit.
    (root / "configs/final.yaml").write_text(yaml.safe_dump(final, sort_keys=False))
    write_json(
        root / "results/pilot/selection.json",
        {
            "selected": selected,
            "total_runs": len(frame),
            "selection_rule": "highest mean validation macro F1 across 3 methods and 3 seeds",
            "primary_choice": (
                "Dry Bean: natural imbalance, 7 classes, ample support, no spatial caveat"
            ),
            "test_predictions_computed": False,
        },
    )
    paths = list((root / "src/ml741_assignment3").glob("*.py"))
    paths += list((root / "scripts").glob("*.py"))
    paths += [
        root / "configs/final.yaml",
        root / "requirements.lock.txt",
        root / "results/pilot/dataset_audit.json",
        root / "results/pilot/search_design.json",
        root / "results/pilot/search_results.csv",
        root / "results/pilot/selection.json",
    ]
    write_json(
        lock_path,
        {
            "locked_at_utc": datetime.now(UTC).isoformat(),
            "protocol_version": 1,
            "sha256": {p.relative_to(root).as_posix(): file_sha256(p) for p in sorted(paths)},
            "environment": environment_manifest(),
            "test_evaluation_performed": False,
        },
    )
    print(aggregates.to_string(), flush=True)
    print("LOCKED", selected, flush=True)


if __name__ == "__main__":
    main()
