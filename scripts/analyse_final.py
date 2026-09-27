"""Deterministic report evidence from verified final bundles; never trains a model."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml741_assignment3.analysis import (
    holm,
    paired_values,
    representative_seed,
    require,
    sign_flip,
    summary_statistics,
    validate_matrix,
)
from ml741_assignment3.artifacts import file_sha256, write_json
from ml741_assignment3.experiments import verify_lock

METHODS = ["all_class_baseline", "fixed_capacity_incremental", "adaptive_capacity_incremental"]
LABELS = ["All-class", "Fixed inc.", "Adaptive"]
COLORS = ["#34699a", "#d58b22", "#328364"]
DATASETS = ["dry_bean", "landsat", "optdigits"]
TITLES = {"dry_bean": "Dry Bean", "landsat": "Landsat", "optdigits": "Optical Digits"}


def save_figure(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        directory / f"{name}.svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "ML741 reproducible analysis"},
    )
    plt.close(fig)


def export_table(frame, directory, name):
    frame.to_csv(directory / f"{name}.csv", index=False)
    (directory / f"{name}.tex").write_text(
        frame.to_latex(index=False, escape=True, float_format="%.4f")
    )


def main():
    root = Path.cwd()
    config, _ = verify_lock(root)
    completion = json.loads((root / "results/final/completion_manifest.json").read_text())
    require(completion["verified_runs"] == 90, "incomplete verification")
    for relative, expected in completion["sha256"].items():
        require(file_sha256(root / relative) == expected, f"changed run artifact: {relative}")
    for relative, expected in completion["canonical_sha256"].items():
        require(
            file_sha256(root / relative) == expected, f"changed canonical table: {relative}"
        )
    folder = root / "results/summaries"
    figures = root / "results/figures/final"
    tables = root / "results/tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(folder / "runs.csv")
    per_class = pd.read_csv(folder / "per_class.csv")
    stage_recalls = pd.read_csv(folder / "stage_recalls.csv")
    stages = pd.read_csv(folder / "stages.csv")
    validate_matrix(frame, DATASETS, METHODS, config["seeds"])
    metrics = [
        "macro_f1",
        "accuracy",
        "balanced_accuracy",
        "cross_entropy",
        "minority_recall",
        "majority_recall",
        "parameters",
        "hidden_units",
        "runtime_seconds",
        "gradient_updates",
        "training_example_presentations",
        "training_epochs",
        "validation_forgetting",
    ]
    statistics = []
    for (dataset, method), group in frame.groupby(["dataset", "method"]):
        for metric in metrics:
            values = group[metric].dropna().to_numpy()
            if len(values):
                statistics.append(
                    dict(
                        dataset=dataset,
                        method=method,
                        metric=metric,
                        **summary_statistics(values),
                    )
                )
    stats = pd.DataFrame(statistics)
    stats.to_csv(folder / "aggregate_statistics.csv", index=False)
    comparisons = []
    for dataset in DATASETS:
        subset = frame[frame.dataset == dataset]
        for a, b, name in [
            (METHODS[1], METHODS[0], "fixed_minus_baseline"),
            (METHODS[2], METHODS[0], "adaptive_minus_baseline"),
            (METHODS[2], METHODS[1], "adaptive_minus_fixed"),
        ]:
            difference = paired_values(subset, a, b, "macro_f1")
            comparisons.append(
                {
                    "dataset": dataset,
                    "comparison": name,
                    **summary_statistics(difference),
                    "wins": int(sum(difference > 0)),
                    "ties": int(sum(difference == 0)),
                    "losses": int(sum(difference < 0)),
                    "p_raw": sign_flip(difference) if b == METHODS[0] else None,
                }
            )
    comparisons = pd.DataFrame(comparisons)
    mask = comparisons.p_raw.notna()
    require(mask.sum() == 6, "wrong testing family")
    comparisons.loc[mask, "p_holm"] = holm(comparisons.loc[mask, "p_raw"])
    comparisons.to_csv(folder / "paired_comparisons.csv", index=False)
    class_summary = (
        per_class.groupby(["dataset", "method", "label", "class_name"], sort=True)
        .agg(
            recall_mean=("recall", "mean"),
            recall_sd=("recall", "std"),
            precision_mean=("precision", "mean"),
            f1_mean=("f1", "mean"),
            test_support=("support", "first"),
            train_count=("train_count", "first"),
            minority=("minority", "first"),
            majority=("majority", "first"),
        )
        .reset_index()
    )
    class_summary.to_csv(folder / "class_summary.csv", index=False)
    class_differences = []
    for dataset in DATASETS:
        for label in sorted(per_class[per_class.dataset == dataset].label.unique()):
            subset = per_class[(per_class.dataset == dataset) & (per_class.label == label)]
            for method in METHODS[1:]:
                diff = paired_values(subset, method, METHODS[0], "recall")
                class_differences.append(
                    dict(
                        dataset=dataset,
                        label=label,
                        class_name=subset.class_name.iloc[0],
                        method=method,
                        mean_recall_difference=float(diff.mean()),
                        minimum=float(diff.min()),
                        maximum=float(diff.max()),
                    )
                )
    pd.DataFrame(class_differences).to_csv(folder / "class_differences.csv", index=False)
    representatives = []
    for (dataset, method), group in frame.groupby(["dataset", "method"]):
        representatives.append(
            dict(dataset=dataset, method=method, seed=representative_seed(group))
        )
    reps = pd.DataFrame(representatives)
    reps.to_csv(folder / "representative_runs.csv", index=False)
    stages.groupby(["dataset", "method", "termination"]).size().rename("count").to_csv(
        folder / "termination_counts.csv"
    )
    # Endpoint consolidation changes: class-wise and aggregate, not immediate forgetting.
    consolidation = []
    for (dataset, method, seed), group in stages.groupby(["dataset", "method", "seed"]):
        if method == METHODS[0]:
            continue
        ordered = group.sort_values("stage")
        consolidation.append(
            dict(
                dataset=dataset,
                method=method,
                seed=seed,
                macro_f1_change=ordered.iloc[-1].macro_f1 - ordered.iloc[-2].macro_f1,
            )
        )
    pd.DataFrame(consolidation).to_csv(folder / "consolidation_changes.csv", index=False)
    main_table = []
    for (dataset, method), group in frame.groupby(["dataset", "method"]):
        main_table.append(
            {
                "Dataset": TITLES[dataset],
                "Method": LABELS[METHODS.index(method)],
                "Macro F1": f"{group.macro_f1.mean():.4f} +/- {group.macro_f1.std():.4f}",
                "Accuracy": group.accuracy.mean(),
                "Balanced accuracy": group.balanced_accuracy.mean(),
                "Cross-entropy": group.cross_entropy.mean(),
            }
        )
    export_table(pd.DataFrame(main_table), tables, "performance")
    export_table(
        comparisons[["dataset", "comparison", "mean", "ci_low", "ci_high", "p_holm"]],
        tables,
        "paired_differences",
    )
    for name, columns in (
        ("class_groups", ["minority_recall", "majority_recall"]),
        ("cost", ["parameters", "hidden_units", "gradient_updates", "runtime_seconds"]),
    ):
        export_table(
            frame.groupby(["dataset", "method"])[columns].mean().reset_index(), tables, name
        )
    dataset_rows, setting_rows = [], []
    for dataset, values in config["datasets"].items():
        audit = values["data"]
        dataset_rows.append(
            dict(
                dataset=dataset,
                features=audit["features"],
                classes=audit["classes"],
                train=sum(audit["counts"]["train"]),
                validation=sum(audit["counts"]["validation"]),
                test=sum(audit["counts"]["test"]),
                imbalance=audit["train_imbalance_ratio"],
            )
        )
        for section in ("training", "controller"):
            for setting, value in values[section].items():
                setting_rows.append(
                    dict(dataset=dataset, section=section, setting=setting, value=str(value))
                )
        setting_rows.append(
            dict(
                dataset=dataset,
                section="architecture",
                setting="fixed_hidden_units",
                value=str(values["hidden_units"]),
            )
        )
    export_table(pd.DataFrame(dataset_rows), tables, "datasets")
    export_table(pd.DataFrame(setting_rows), tables, "settings")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "ml741-final-v1",
        }
    )
    # Raw seeds and distributions, no opaque mean-only bars.
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained")
    for ax, dataset in zip(axes, DATASETS, strict=True):
        subset = frame[frame.dataset == dataset]
        for i, method in enumerate(METHODS):
            values = subset[subset.method == method].sort_values("seed").macro_f1.to_numpy()
            ax.scatter(i + np.linspace(-0.13, 0.13, len(values)), values, color=COLORS[i], s=18)
            ax.plot([i - 0.2, i + 0.2], [values.mean()] * 2, color="black", lw=1.5)
        ax.set(
            title=TITLES[dataset], xticks=range(3), xticklabels=LABELS, ylabel="Test macro F1"
        )
    save_figure(fig, figures, "performance_seeds")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained")
    for ax, dataset in zip(axes, DATASETS, strict=True):
        subset = comparisons[
            (comparisons.dataset == dataset)
            & (comparisons.comparison != "adaptive_minus_fixed")
        ]
        for i, (_, row) in enumerate(subset.iterrows()):
            ax.errorbar(
                i,
                100 * row["mean"],
                yerr=[[100 * (row["mean"] - row.ci_low)], [100 * (row.ci_high - row["mean"])]],
                fmt="o",
                color=COLORS[i + 1],
                capsize=4,
            )
        ax.axhline(0, color="gray", lw=1, ls="--")
        ax.set(
            title=TITLES[dataset],
            xticks=[0, 1],
            xticklabels=LABELS[1:],
            ylabel="Macro-F1 difference (percentage points)",
        )
    fig.suptitle("Paired mean differences from all-class; 95% seed-bootstrap intervals")
    save_figure(fig, figures, "paired_differences")
    # Distribution and recall figures per dataset remain readable in report columns.
    for dataset in DATASETS:
        info = config["datasets"][dataset]["data"]
        labels = info["labels"]
        fig, ax = plt.subplots(figsize=(5.5, 3.1), layout="constrained")
        ax.bar(labels, info["counts"]["train"], color=COLORS[0])
        ax.tick_params(axis="x", rotation=40 if dataset == "dry_bean" else 0)
        ax.set(title=TITLES[dataset], ylabel="Retained training examples")
        save_figure(fig, figures, f"class_distribution_{dataset}")
        fig, ax = plt.subplots(figsize=(6, 2.7), layout="constrained")
        array = np.asarray(
            [
                class_summary[
                    (class_summary.dataset == dataset) & (class_summary.method == method)
                ]
                .sort_values("label")
                .recall_mean
                for method in METHODS
            ]
        )
        im = ax.imshow(array, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for i in range(3):
            for j in range(len(labels)):
                ax.text(
                    j,
                    i,
                    f"{array[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if array[i, j] > 0.65 else "black",
                    fontsize=8,
                )
        ax.set(
            xticks=range(len(labels)),
            xticklabels=labels,
            yticks=range(3),
            yticklabels=LABELS,
            title=f"{TITLES[dataset]}: mean test recall (10 seeds)",
        )
        ax.tick_params(axis="x", rotation=40 if dataset == "dry_bean" else 0)
        fig.colorbar(im, ax=ax, label="Recall", shrink=0.8)
        save_figure(fig, figures, f"recall_heatmap_{dataset}")
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), layout="constrained")
        for i, (ax, method) in enumerate(zip(axes, METHODS, strict=True)):
            seed = int(reps[(reps.dataset == dataset) & (reps.method == method)].seed.iloc[0])
            path = root / "results/final/runs" / dataset / str(seed) / method
            summary = json.loads((path / "summary.json").read_text())
            confusion = np.asarray(summary["metrics"]["confusion_matrix"], dtype=float)
            confusion /= confusion.sum(axis=1, keepdims=True)
            ax.imshow(confusion, vmin=0, vmax=1, cmap="Blues")
            ax.set(
                xticks=range(len(labels)),
                xticklabels=labels,
                yticks=range(len(labels)),
                yticklabels=labels,
                title=f"{LABELS[i]} (seed {seed})",
                xlabel="Predicted",
                ylabel="True",
            )
            ax.tick_params(axis="both", labelsize=6)
            ax.tick_params(axis="x", rotation=90)
        fig.suptitle(f"{TITLES[dataset]}: row-normalised confusion, lower-median F1 runs")
        save_figure(fig, figures, f"confusions_{dataset}")
        fig, axes = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True, layout="constrained")
        for seed in config["seeds"]:
            path = root / "results/final/runs" / dataset / str(seed) / METHODS[2]
            history = pd.read_csv(path / "epoch_history.csv")
            axes[0].plot(history.global_epoch, history.validation_macro_f1, alpha=0.5, lw=0.8)
            axes[1].step(history.global_epoch, history.hidden_units, where="post", alpha=0.5)
        axes[0].set(
            title=f"{TITLES[dataset]}: all 10 adaptive runs", ylabel="Seen-class validation F1"
        )
        axes[1].set(ylabel="Hidden units", xlabel="Cumulative training epoch")
        save_figure(fig, figures, f"growth_{dataset}")
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), layout="constrained", sharey=True)
        for ax, method in zip(axes, METHODS[1:], strict=True):
            subset = stage_recalls[
                (stage_recalls.dataset == dataset) & (stage_recalls.method == method)
            ]
            for label in range(len(labels)):
                trace = subset[subset.label == label].groupby("stage").recall.mean()
                ax.plot(trace.index + 1, trace.values, marker=".", label=labels[label])
            ax.set(
                title=LABELS[METHODS.index(method)],
                xlabel="Stage (last = consolidation)",
                ylabel="Mean validation recall",
                ylim=(0, 1.03),
            )
        axes[-1].legend(fontsize=6, ncol=2, loc="lower left")
        fig.suptitle(f"{TITLES[dataset]}: stage-end class retention; absent classes omitted")
        save_figure(fig, figures, f"retention_{dataset}")
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), layout="constrained")
    for col, dataset in enumerate(DATASETS):
        for i, method in enumerate(METHODS):
            subset = frame[(frame.dataset == dataset) & (frame.method == method)]
            axes[0, col].scatter(
                subset.parameters, subset.macro_f1, label=LABELS[i], color=COLORS[i], s=18
            )
            axes[1, col].scatter(
                subset.gradient_updates, subset.macro_f1, color=COLORS[i], s=18
            )
        axes[0, col].set(title=TITLES[dataset], xlabel="Parameters", ylabel="Test macro F1")
        axes[1, col].set(xlabel="Gradient updates", ylabel="Test macro F1")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=3)
    save_figure(fig, figures, "cost_tradeoffs")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained")
    for ax, dataset in zip(axes, DATASETS, strict=True):
        for i, method in enumerate(METHODS):
            subset = frame[(frame.dataset == dataset) & (frame.method == method)]
            ax.scatter(
                subset.runtime_seconds, subset.macro_f1, color=COLORS[i], label=LABELS[i], s=18
            )
        ax.set(
            title=TITLES[dataset],
            xlabel="Training/evaluation runtime (s)",
            ylabel="Test macro F1",
        )
    axes[0].legend(fontsize=7)
    save_figure(fig, figures, "runtime_tradeoffs")
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    boxes = [
        (
            0.5,
            0.91,
            "Sort training classes: rarest first\nStart with 2 outputs, 0 hidden units",
        ),
        (
            0.5,
            0.73,
            "Train on all retained examples of seen classes\n"
            "Until plateau, overfit or stage budget",
        ),
        (0.5, 0.54, "At a plateau: underfit + capacity available?"),
        (0.22, 0.33, "Yes: restore best model\nAdd 1 hidden unit\nReset Adam; resume training"),
        (
            0.78,
            0.33,
            "No / overfit / budget reached:\nRestore best model\nIntroduce next class, if any",
        ),
        (0.78, 0.09, "All classes seen: consolidate\nSelect on validation; evaluate test"),
    ]
    for x, y, text in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=.5", fc="#edf3f8", ec=COLORS[0]),
        )
    for start, end in [
        ((0.5, 0.85), (0.5, 0.79)),
        ((0.5, 0.67), (0.5, 0.58)),
        ((0.4, 0.50), (0.22, 0.41)),
        ((0.6, 0.50), (0.78, 0.41)),
        ((0.78, 0.25), (0.78, 0.15)),
    ]:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->"))
    ax.plot([0.02, 0.005, 0.005, 0.23], [0.33, 0.33, 0.73, 0.73], color="gray", lw=1)
    ax.annotate(
        "", xy=(0.25, 0.73), xytext=(0.20, 0.73), arrowprops=dict(arrowstyle="->", color="gray")
    )
    ax.plot([0.98, 0.995, 0.995, 0.80], [0.33, 0.33, 0.73, 0.73], color="gray", lw=1, ls="--")
    ax.annotate(
        "", xy=(0.77, 0.73), xytext=(0.84, 0.73), arrowprops=dict(arrowstyle="->", color="gray")
    )
    save_figure(fig, figures, "algorithm_flow")
    # Frozen numerical conclusion inputs; prose handoff cites these directly.
    print(pd.DataFrame(main_table).to_string(index=False))
    print(
        comparisons[["dataset", "comparison", "mean", "ci_low", "ci_high", "p_holm"]].to_string(
            index=False
        )
    )
    paths = sorted(
        list(folder.glob("*.csv")) + list(figures.glob("*")) + list(tables.glob("*"))
    )
    write_json(
        root / "results/final/analysis_manifest.json",
        {
            "sha256": {path.relative_to(root).as_posix(): file_sha256(path) for path in paths},
            "analysis_source_sha256": {
                str(path.relative_to(root)): file_sha256(path)
                for path in [
                    root / "src/ml741_assignment3/analysis.py",
                    root / "scripts/collect_final.py",
                    root / "scripts/analyse_final.py",
                    root / "configs/analysis.yaml",
                ]
            },
            "final_collection_sha256": file_sha256(
                root / "results/final/completion_manifest.json"
            ),
        },
    )


if __name__ == "__main__":
    main()
