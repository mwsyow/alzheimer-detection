"""Overlay configuration-level rotating CV results on the historical benchmark."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


MODELS = ["Simple3DCNN", "ResNet10 scratch", "ResNet10 MedicalNet",
          "EfficientNet-B0", "DenseNet121 pretrained"]
LABELS = ["Simple3DCNN", "ResNet10\nscratch", "ResNet10\nMedicalNet",
          "EfficientNet-B0", "DenseNet121\npretrained"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=Path("reports/benchmark_test_set_results.csv"))
    parser.add_argument("--comparisons", type=Path, nargs="+", required=True,
                        help="Comparison directories containing aggregate.csv and runs.csv")
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark_old_vs_rotating_cv.png"))
    args = parser.parse_args()
    old = pd.read_csv(args.old)
    newer = {}
    rows = []
    for directory in args.comparisons:
        frame = pd.read_csv(directory / "aggregate.csv", header=[0, 1], index_col=list(range(5)))
        runs = pd.read_csv(directory / "runs.csv")
        if len(frame) != 1:
            raise ValueError(f"Select exactly one configuration: {directory}")
        if set(runs.cv_random_seed) != {7, 42, 1337, 2024} or len(runs) != 4:
            raise ValueError(f"Expected four matching partition seeds: {directory}")
        if set(runs.threshold_mode) != {"per-fold"}:
            raise ValueError(f"Expected per-fold threshold results: {directory}")
        architecture = runs.architecture.iloc[0]
        adaptation = runs.adaptation_mode.iloc[0]
        name = {
            "Simple3DCNN": "Simple3DCNN",
            "EfficientNetBN": "EfficientNet-B0",
            "DenseNet121": "DenseNet121 pretrained",
        }.get(architecture)
        if architecture == "ResNet10":
            name = "ResNet10 MedicalNet" if "medicalnet" in adaptation else "ResNet10 scratch"
        if name is None or name in newer:
            raise ValueError(f"Unknown or duplicate model: {architecture}, {adaptation}")
        newer[name] = frame.iloc[0]

    fig, ax = plt.subplots(figsize=(16, 8.8))
    fig.subplots_adjust(left=.07, right=.985, bottom=.12, top=.78)
    width, shift = .27, .095
    colors = ["#247f9f", "#40ad90"]
    greys = ["#b9bec3", "#d3d6d8"]
    for i, model in enumerate(MODELS):
        historic = old[old.model == model]
        for j, (old_metric, metric) in enumerate([("auroc", "roc_auc"), ("balanced_accuracy", "balanced_accuracy")]):
            x = i + (j - .5) * .34
            mean, sd = historic[old_metric].mean(), historic[old_metric].std(ddof=1)
            ax.bar(x + shift, mean, width, color=greys[j], zorder=2,
                   yerr=sd, capsize=4, error_kw={"ecolor": "#8a9096", "elinewidth": 1.5, "zorder": 3})
            ax.text(x + shift, mean + sd + .012, f"{mean:.3f}", ha="center", fontsize=10, color="#858c92")
            rows.append(dict(model=model, protocol="old_fixed_test", metric=metric, mean=mean, sd=sd))
            if model in newer:
                mean = float(newer[model][(metric, "mean")])
                sd = float(newer[model][(metric, "std")])
                ax.bar(x, mean, width, color=colors[j], zorder=4,
                       yerr=sd, capsize=4, error_kw={"ecolor": "#172c3b", "elinewidth": 1.6, "zorder": 5})
                ax.text(x, mean + sd + .012, f"{mean:.3f}", ha="center", fontsize=10, color="#17344b", zorder=6)
                rows.append(dict(model=model, protocol="rotating_oof_per_fold", metric=metric, mean=mean, sd=sd))
    ax.set_xticks(np.arange(5) + shift / 2, LABELS, fontsize=12)
    ax.set_ylim(.45, 1.015)
    ax.set_yticks(np.arange(.5, 1.01, .1))
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("Held-out score", fontsize=13, color="#627682")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#d9e2e7", linewidth=.9)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    fig.text(.07, .95, "Old vs New CV Strategy", fontsize=23, weight="bold", color="#18364c")
    fig.text(.07, .902, "Mean ± sample SD across CV seeds 7, 42, 1337, and 2024", fontsize=13, color="#627682")
    fig.legend(handles=[Patch(color=colors[0], label="New AUROC"), Patch(color=colors[1], label="New balanced accuracy"),
                        Patch(color=greys[0], label="Old AUROC"), Patch(color=greys[1], label="Old balanced accuracy")],
               loc="upper left", bbox_to_anchor=(.065, .873), ncol=4, frameon=False, fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf"]:
        fig.savefig(args.output.with_suffix(suffix), dpi=220, facecolor="white")
    pd.DataFrame(rows).to_csv(args.output.with_suffix(".csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
