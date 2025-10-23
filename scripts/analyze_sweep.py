#!/usr/bin/env python
"""Aggregate ClinVar sweep results and report the best run."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SWEEP_ROOT = Path("checkpoints/Clinvar_trm-ACT-torch")


def load_run_metrics(run_dir: Path) -> dict:
    metrics_file = run_dir / "ClinVarEvaluator_metrics.json"
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text())
    else:
        metrics = {}
    cfg = json.loads((run_dir / "all_config.yaml").read_text())

    return {
        "run": run_dir.name,
        "roc_auc": metrics.get("ClinVar/roc_auc", float("nan")),
        "accuracy": metrics.get("ClinVar/accuracy", float("nan")),
        "hidden_size": cfg["arch"]["hidden_size"],
        "L_layers": cfg["arch"]["L_layers"],
        "L_cycles": cfg["arch"]["L_cycles"],
        "lr": cfg["lr"],
        "checkpoint": str(run_dir / "step_1560"),
        "config": str(run_dir / "all_config.yaml"),
    }


def main() -> None:
    runs = sorted([d for d in SWEEP_ROOT.glob("clinvar_sweep_*") if d.is_dir()])
    if not runs:
        print("No sweep results found under", SWEEP_ROOT)
        return

    records = [load_run_metrics(run) for run in runs if (run / 'all_config.yaml').exists()]
    df = pd.DataFrame(records)
    df = df.dropna(subset=["roc_auc", "accuracy"])
    if df.empty:
        print("No metrics found; ensure sweep has finished.")
        return

    best = df.sort_values("roc_auc", ascending=False).iloc[0]
    print("Best run:")
    print(best)
    df.to_csv("sweep_summary.csv", index=False)
    print("Summary saved to sweep_summary.csv")


if __name__ == "__main__":
    main()
