#!/usr/bin/env python
"""
Compare evaluation metrics between TinyVariant TRM checkpoints and baseline models.

Usage example:
    python scripts/plot_eval_comparison.py \
        --trm outputs/clinvar_trm_metrics.json \
        --baseline outputs/clinvar_logreg_metrics.json \
        --output docs/figures/clinvar_metric_comparison.png
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
    parser.add_argument(
        "--trm",
        type=Path,
        required=True,
        help="JSON metrics produced by tools/evaluate_clinvar_checkpoint.py (--output ...).",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="JSON metrics from baseline training (e.g., tools/train_baseline_logreg.py --output ...).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the bar chart (PNG/SVG). Defaults to inline display.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="ClinVar Evaluation Comparison",
        help="Plot title (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.trm.exists():
        raise FileNotFoundError(f"TRM metrics file not found: {args.trm}")
    if not args.baseline.exists():
        raise FileNotFoundError(f"Baseline metrics file not found: {args.baseline}")

    trm_metrics = load_metrics(args.trm)
    baseline_metrics = load_metrics(args.baseline)

    metrics = ["accuracy", "roc_auc"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    trm_values = [trm_metrics[m] for m in metrics]
    baseline_values = [baseline_metrics[m] for m in metrics]

    ax.bar(x - width / 2, baseline_values, width, label="Baseline (LogReg)", color="#8fbce6")
    ax.bar(x + width / 2, trm_values, width, label="TinyVariant TRM", color="#f3a45b")

    ax.set_xticks(x, [m.replace("_", " ").title() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(args.title)
    ax.legend()

    for idx, value in enumerate(baseline_values):
        ax.text(x[idx] - width / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    for idx, value in enumerate(trm_values):
        ax.text(x[idx] + width / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=200)
        print(f"Saved comparison plot to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
