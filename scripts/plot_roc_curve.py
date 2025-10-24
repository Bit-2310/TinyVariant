#!/usr/bin/env python
"""
Plot ROC curve from TinyVariant evaluation predictions.

First generate prediction records using:
    python tools/evaluate_clinvar_checkpoint.py \
        --config .../all_config.yaml \
        --checkpoint .../step_xxxx \
        --output outputs/clinvar_trm_metrics.json \
        --save-preds outputs/clinvar_trm_predictions.jsonl

Then plot:
    python scripts/plot_roc_curve.py \
        --preds outputs/clinvar_trm_predictions.jsonl \
        --output docs/figures/clinvar_trm_roc.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    labels = []
    with path.open("r") as f:
        for line in f:
            record = json.loads(line)
            scores.append(float(record["score"]))
            labels.append(int(record["label"]))
    if not scores:
        raise ValueError(f"No prediction records found in {path}")
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def compute_roc(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    # Convert labels to binary {0,1} relative to pathogenic class (assumed positive=1).
    if set(np.unique(labels)) - {0, 1}:
        raise ValueError("Labels must be encoded as 0 (benign) or 1 (pathogenic).")

    order = np.argsort(-scores)
    scores = scores[order]
    labels = labels[order]

    tps = np.cumsum(labels == 1)
    fps = np.cumsum(labels == 0)

    total_pos = tps[-1]
    total_neg = fps[-1]

    tpr = tps / total_pos if total_pos > 0 else np.zeros_like(tps, dtype=np.float64)
    fpr = fps / total_neg if total_neg > 0 else np.zeros_like(fps, dtype=np.float64)

    # Prepend origin (0,0) for plotting.
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    auc = np.trapezoid(tpr, fpr)
    return fpr, tpr, auc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preds",
        type=Path,
        required=True,
        help="JSONL predictions produced by evaluate_clinvar_checkpoint.py --save-preds ...",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the ROC plot (PNG/SVG). Defaults to inline display.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="TinyVariant TRM ROC Curve",
        help="Plot title (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scores, labels = load_predictions(args.preds)
    fpr, tpr, auc = compute_roc(scores, labels)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"TinyVariant TRM (AUC={auc:.3f})", color="#f37f5b")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#cccccc", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(args.title)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=200)
        print(f"Saved ROC curve to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
