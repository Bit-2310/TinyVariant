# TinyVariant: tiny recursion meets variant pathogenicity

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-red.svg)
![Status](https://img.shields.io/badge/status-proof_of_concept-orange.svg)
![Hardware](https://img.shields.io/badge/hardware-RTX%203050-lightgrey.svg)

Hello! This repo started life as a fork of Samsung SAIT Montréal’s
**Tiny Recursive Models** project, the ARC-AGI puzzle solver. I wanted to see
if that clever recursive reasoning core could help with something far more
biological: guessing whether a DNA variant is pathogenic. Rather than rewrite
everything from scratch, I’m keeping the original TRM plumbing and layering a
bioinformatics proof of concept on top.

**Objective — prove that a slimmed-down TRM can break 60 % accuracy on a
ClinVar pathogenic vs benign classification task using nothing fancier than
an RTX 3050.**

> The core training loop, halting logic, and attention layout follow Samsung
> SAIT Montréal’s Tiny Recursive Models. We keep their architecture intact
> and swap the ARC puzzle inputs for ClinVar-derived feature tokens.

That’s it. If the approach shows promise, the plan is to grow TinyVariant into
a variant-analysis toolkit; if not, we pivot with a clear conscience.

---

## Where things stand right now

- ✅ Confirmed the base TRM code runs locally (no more adam-atan2 headaches).
- ✅ Built a mini ARC dataset (800 puzzles) to make sure the training loop,
  logging, and checkpointing play nicely with 4 GB of VRAM.
- ✅ Logged the whole bring-up story in `biotrm_progress_log.txt`.
- 🔄 Next up: swap in ClinVar data, craft a VariantTRM model, and push for that
  60 % accuracy milestone.

---

## Quickstart (current state)

1. Activate the working environment:
   ```bash
   conda activate trm_env
   ```
2. Build the tiny ARC sanity dataset (already done once, but reproducible):
   ```bash
   PYTHONPATH=$(pwd) python -m dataset.build_arc_dataset \
       --input-file-prefix kaggle/combined/arc-agi \
       --output-dir data/arc1concept-mini \
       --subsets training evaluation \
       --test-set-name evaluation \
       --num-aug 0
   ```
3. Run the trimmed TRM config that fits on the RTX 3050:
   ```bash
   DISABLE_COMPILE=1 python pretrain.py \
       arch=trm \
       data_paths="[data/arc1concept-mini]" \
       epochs=2 eval_interval=2 \
       global_batch_size=4 \
       arch.hidden_size=128 arch.puzzle_emb_ndim=128 \
       arch.num_heads=2 arch.expansion=2 \
       arch.L_layers=1 arch.L_cycles=2 arch.H_cycles=1 \
       +run_name=debug_run_tiny
   ```
   You’ll see WandB logs and a checkpoint under `checkpoints/`.

4. Rebuild the ClinVar dataset with phenotype/provenance context (5 k per class by default) and train the VariantTRM run (50 epochs, evaluator logs every 5 epochs):
   ```bash
   # Rebuild dataset after any feature changes (ClinSig columns are excluded by default)
   python tools/prepare_clinvar_dataset.py --max-per-class 5000
   python tools/build_clinvar_trm_dataset.py

   WANDB_MODE=offline DISABLE_COMPILE=1 \
   python pretrain.py --config-name cfg_clinvar_long +run_name=clinvar_long
   ```
   Add `+early_stop_patience=5` (and optional `+early_stop_metric`) to enable early stopping once the validation ROC AUC plateaus.  
   Variant sequences now contain 25 tokens covering gene/allele features, three phenotype buckets, evidence sources, submitter/evaluation buckets, five position digits, and a label slot.

5. Baseline comparison (logistic regression on the same split):
   ```bash
   python tools/train_baseline_logreg.py \
       --input data/clinvar/processed/clinvar_missense_balanced.tsv \
       --output outputs/clinvar_logreg_metrics.json
   ```

6. Sanity test (ensures no ClinicalSignificance leakage):
   ```bash
   python -m pytest tests/test_clinvar_dataset.py
   ```
   (Requires the balanced ClinVar TSV under `data/clinvar/processed/clinvar_missense_balanced.tsv`.)

7. Evaluate checkpoints (CPU or CUDA):
   ```bash
   python tools/evaluate_clinvar_checkpoint.py \
       --config checkpoints/Clinvar_trm-ACT-torch/clinvar_long_20251024-175518/all_config.yaml \
      --checkpoint checkpoints/Clinvar_trm-ACT-torch/clinvar_long_20251024-175518/step_1248 \
      --device cuda \
      --output outputs/clinvar_trm_metrics.json \
      --save-preds outputs/clinvar_trm_predictions.jsonl
   ```
   (Set `--device cpu` if GPUs are unavailable.)

8. Generate documentation plots (optional):
   ```bash
   python scripts/plot_eval_comparison.py \
       --trm outputs/clinvar_trm_metrics.json \
       --baseline outputs/clinvar_logreg_metrics.json \
       --output docs/figures/clinvar_metric_comparison.png

   python scripts/plot_roc_curve.py \
       --preds outputs/clinvar_trm_predictions.jsonl \
       --output docs/figures/clinvar_trm_roc.png
   ```
   (Both scripts default to displaying the figure if `--output` is omitted.)

9. Hyperparameter sweep example:
   ```bash
   WANDB_MODE=offline DISABLE_COMPILE=1 \
   python pretrain.py --config-name clinvar_sweep --multirun
   python scripts/analyze_sweep.py
   ```
   (Sweeps hidden size, L_layers, L_cycles, and learning rate; each run is stored under
   `checkpoints/Clinvar_trm-ACT-torch/<override_dirname>`.)

---

## Roadmap snapshot

1. **ClinVar data prep** — download, filter to missense variants, create a
   5 k example dataset (2.5 k pathogenic / 2.5 k benign).
2. **VariantTRM** — adapt the TRM core for binary classification with lightweight
   embeddings (position, ref/alt, gene).
3. **Training loop** — start with tiny batches, mixed precision if needed, and
   push toward >60 % accuracy.
4. **Baselines & comparisons** — logistic regression, MLP, random forest.
5. **Decision time** — if recursion helps and metrics are encouraging, polish
   for a write-up; if not, pivot to the next biology challenge.

Detailed steps live in the TinyVariant proof-of-concept plan (see below).

---

## Files to watch

- `biotrm_progress_log.txt` — running commentary of everything that’s been
  tried and learned so far.
- `data/arc1concept-mini/` — tiny ARC dataset used for sanity checks.
- `biotrm/` (coming soon) — home for TinyVariant-specific data and model code.

---

## Long-form plan

For the full breakdown of tasks, success metrics, and deliverables, see the
TinyVariant POC plan in the repo. Highlights:

```
Phase 1  Setup & ClinVar data ingestion
Phase 2  Minimal VariantTRM model + training loop
Phase 3  Feature upgrades & hyperparameter search
Phase 4  Baselines, ablations, error analysis
Phase 5  Documentation + go/no-go decision
```

---

Questions, ideas, or words of caution? Pop them into the issue tracker (once
the repo rename to `TinyVariant` lands) or reach out directly. This is very
much a lab notebook in motion. Stay tuned for updates.
