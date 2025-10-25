#!/usr/bin/env python
"""
Plot TinyVariant TRM performance across full feature and ablation runs.

Usage example:
    python scripts/plot_trm_ablation.py \
        --full outputs/clinvar_trm_metrics.json \
        --no-phenotype outputs/clinvar_long_phenotype_ablation_20251024-215110_metrics.json \
        --no-provenance outputs/clinvar_long_provenance_ablation_20251025-074717_metrics.json \
        --output docs/figures/clinvar_trm_ablation_comparison.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text())
    return {
        "accuracy": float(data.get("accuracy", 0.0)),
        "roc_auc": float(data.get("roc_auc", 0.0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True, help="Metrics JSON for full-feature TRM run.")
    parser.add_argument("--no-phenotype", type=Path, required=True, help="Metrics JSON for phenotype ablation run.")
    parser.add_argument("--no-provenance", type=Path, required=True, help="Metrics JSON for provenance ablation run.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save plot (PNG/SVG). If omitted, shows the figure interactively.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    runs = {
        "Full features": load_metrics(args.full),
        "No phenotype": load_metrics(args.no_phenotype),
        "No provenance": load_metrics(args.no_provenance),
    }

    labels = list(runs.keys())
    auc_values = np.array([runs[label]["roc_auc"] for label in labels])
    acc_values = np.array([runs[label]["accuracy"] for label in labels])

    x = np.arange(len(labels))
    width = 0.8

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    colors = ["#f3a45b", "#8fbce6", "#f5cc5b"]

    axes[0].bar(x, auc_values, width, color=colors)
    axes[0].set_title("ROC AUC")
    axes[0].set_ylim(auc_values.min() - 0.002, auc_values.max() + 0.002)
    axes[0].set_xticks(x, labels, rotation=20, ha="right")

    axes[1].bar(x, acc_values, width, color=colors)
    axes[1].set_title("Accuracy")
    axes[1].set_ylim(acc_values.min() - 0.002, acc_values.max() + 0.002)
    axes[1].set_xticks(x, labels, rotation=20, ha="right")

    for ax, values in zip(axes, [auc_values, acc_values]):
        for idx, val in enumerate(values):
            ax.text(
                x[idx],
                val + 0.0005,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("TinyVariant TRM Ablation Comparison (ClinVar 50k)")
    fig.tight_layout()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=200)
        print(f"Saved ablation comparison plot to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
