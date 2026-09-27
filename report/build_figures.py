"""Report-only presentation of frozen results; no training or evidence rewriting."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report/figures"
METHODS = ["all_class_baseline", "fixed_capacity_incremental", "adaptive_capacity_incremental"]
NAMES = ["All-class", "Fixed incremental", "Adaptive"]
COLOURS = ["#285f86", "#bc7b22", "#327658"]
DATASETS = ["dry_bean", "landsat", "optdigits"]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    runs = pd.read_csv(ROOT / "results/summaries/runs.csv")
    paired = pd.read_csv(ROOT / "results/summaries/paired_comparisons.csv")
    recall = pd.read_csv(ROOT / "results/summaries/class_summary.csv")
    stage_recall = pd.read_csv(ROOT / "results/summaries/stage_recalls.csv")
    # Widths equal the inclusion width; ten-point text is not shrunk in LaTeX.
    for dataset in DATASETS:
        fig, ax = plt.subplots(figsize=(3.48, 2.22))
        fig.subplots_adjust(left=0.20, right=0.97, top=0.95, bottom=0.23)
        for i, method in enumerate(METHODS):
            values = (
                runs[(runs.dataset == dataset) & (runs.method == method)]
                .sort_values("seed")
                .macro_f1
            )
            ax.scatter(i + np.linspace(-0.12, 0.12, 10), values, color=COLOURS[i], s=16)
            ax.hlines(values.mean(), i - 0.23, i + 0.23, colors="black", lw=1.2)
        ax.set(
            xticks=range(3),
            xticklabels=["All-class", "Fixed", "Adaptive"],
            ylabel=r"Test macro $F_1$",
        )
        save(fig, f"performance_{dataset}")
        fig, ax = plt.subplots(figsize=(3.48, 2.35))
        fig.subplots_adjust(left=0.20, right=0.97, top=0.95, bottom=0.23)
        subset = paired[
            (paired.dataset == dataset) & (paired.comparison != "adaptive_minus_fixed")
        ]
        for i, row in enumerate(subset.itertuples()):
            ax.errorbar(
                i,
                row.mean * 100,
                yerr=[[100 * (row.mean - row.ci_low)], [100 * (row.ci_high - row.mean)]],
                fmt="o",
                color=COLOURS[i + 1],
                capsize=5,
            )
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set(
            xticks=[0, 1],
            xticklabels=["Fixed", "Adaptive"],
            ylabel=r"Macro $F_1$ difference (points)",
        )
        save(fig, f"paired_{dataset}")
    for dataset in ["dry_bean", "optdigits"]:
        data = recall[recall.dataset == dataset]
        labels = (
            data[data.method == METHODS[0]].sort_values("label").class_name.astype(str).tolist()
        )
        fig, ax = plt.subplots(figsize=(7.16, 1.35 if dataset == "optdigits" else 1.55))
        fig.subplots_adjust(
            left=0.19, right=0.98, top=0.91, bottom=0.31 if dataset == "optdigits" else 0.48
        )
        values = np.asarray(
            [data[data.method == m].sort_values("label").recall_mean for m in METHODS]
        )
        ax.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(3):
            for j in range(len(labels)):
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if values[i, j] > 0.65 else "black",
                )
        ax.set(
            xticks=range(len(labels)), xticklabels=labels, yticks=range(3), yticklabels=NAMES
        )
        ax.tick_params(axis="x", rotation=30 if dataset == "dry_bean" else 0)
        save(fig, f"recall_{dataset}")
    # Endpoint means align by class stage, not by unequal epoch counts.
    stages = pd.read_csv(ROOT / "results/summaries/stages.csv")
    fig, ax = plt.subplots(figsize=(3.48, 2.45))
    fig.subplots_adjust(left=0.18, right=0.97, top=0.96, bottom=0.24)
    for i, dataset in enumerate(DATASETS):
        subset = stages[
            (stages.dataset == dataset)
            & (stages.method == METHODS[2])
            & (stages.phase == "class_stage")
        ]
        means = subset.groupby("stage").hidden_units.mean()
        ax.plot(
            means.index + 2,
            means.values,
            marker=["o", "s", "^"][i],
            label=["Dry Bean", "Landsat", "Optical Digits"][i],
            color=COLOURS[i],
        )
    ax.set(xlabel="Classes introduced", ylabel="Mean hidden units", xticks=[2, 4, 6, 8, 10])
    ax.legend(frameon=False, loc="upper right")
    save(fig, "capacity")
    fig, ax = plt.subplots(figsize=(3.48, 2.45))
    fig.subplots_adjust(left=0.18, right=0.97, top=0.74, bottom=0.24)
    for i, method in enumerate(METHODS[1:]):
        for label, style in [(8, "--"), (9, "-")]:
            subset = stage_recall[
                (stage_recall.dataset == "optdigits")
                & (stage_recall.method == method)
                & (stage_recall.label == label)
            ]
            means = subset.groupby("stage").recall.mean()
            ax.plot(
                means.index + 1,
                means.values,
                linestyle=style,
                marker=["s", "o"][i],
                markevery=2,
                color=COLOURS[i + 1],
                label=f"{['Fixed', 'Adaptive'][i]}, digit {label}",
            )
    ax.set(
        xlabel="Stage",
        ylabel="Validation recall",
        ylim=(0.68, 1.01),
        xticks=[1, 3, 5, 7, 9, 10],
    )
    ax.legend(
        frameon=False,
        fontsize=9,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        columnspacing=0.9,
        handlelength=2.0,
    )
    save(fig, "retention")
    print("Built report figures from frozen summaries.")


if __name__ == "__main__":
    main()
