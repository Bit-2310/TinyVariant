# TinyVariant – persistent context (updated)

## Purpose
- Apply Samsung SAIT’s Tiny Recursive Model (TRM) architecture to ClinVar pathogenic vs benign classification as a stepping stone toward richer, disease-aware variant interpretation.
- Current model predicts the binary ClinSigSimple label (0 benign / 1 pathogenic). We now plan to incorporate phenotype context to make the proof of concept biologically meaningful.

## Dataset
- Raw source: `data/clinvar/raw/variant_summary.txt.gz` (ClinVar tab-delimited export).
- Filtered and balanced via `tools/prepare_clinvar_dataset.py`:
  - Keep GRCh38 germline single-nucleotide missense variants.
  - ClinSigSimple ∈ {0,1}, review status in {single submitter, multiple submitters w/o conflicts, expert panel, practice guideline}.
  - Balanced using `--max-per-class` (default 5000 per class → 10k total, scalable upward as VRAM/time allow).
- Processed TSV: `data/clinvar/processed/clinvar_missense_balanced.tsv` now carries normalized phenotype terms/IDs, submitter counts/categories, ISO-normalized `LastEvaluatedISO`, and provenance buckets alongside the core variant annotations.

## TRM Tokenization (`tools/build_clinvar_trm_dataset.py`)
Sequence length = 25 tokens:
1. CLS token
2. `GENE:<bucketed gene>` (`<RARE>` for freq<5)
3. `CHR:<chromosome>`
4. `REV:<review status>`
5. `NUC:<ref base>`
6. `NUC:<alt base>`
7. `AA:<from amino acid>`
8. `AA:<to amino acid>`
9. `AAPROP:<property of from AA>` (nonpolar/positive/polar/negative/stop)
10. `AAPROP:<property of to AA>`
11. `AACHANGE:<same or transition>`
12. `CONSEQ:<parsed consequence>`
13. `PHENO_PRIMARY:<top phenotype term or <other>/<none>>`
14. `PHENO_SECONDARY:<second phenotype term or <none>>`
15. `PHENO_SOURCE:<phenotype ID namespace bucket>`
16. `PHENO_COUNT:<bucketed phenotype count>`
17. `SUBMITTERS:<bucketed submitter count>`
18. `EVAL:<bucketed evaluation recency>`
19–23. Five digits for protein position (zero-padded)
24. LABEL slot (`LABEL_BENIGN` or `LABEL_PATHOGENIC`)

## Model & Training
- Architecture: `TinyRecursiveReasoningModel_ACTV1` (unchanged). Recursive passes over latents `z_H`/`z_L`; halting logic via `q_head`.
- Loss head: `VariantClassificationHead` → cross entropy on final token.
- Training script: `pretrain.py`
  - Hydra configuration (`cfg_clinvar_long` as base).
  - Early stopping supported (`+early_stop_patience`, metric defaults to `ClinVar/roc_auc`).
  - Logging: wandb offline, checkpointing every eval.
- Baseline: `tools/train_baseline_logreg.py` mirrors the same features (one-hot + standardized position).

## Current Performance (5k/class dataset with phenotype context)
- TRM (`cfg_clinvar_long`, 50 epochs, early stop=5): accuracy ≈ 0.819, ROC AUC ≈ 0.884.
- Logistic regression baseline (80/20 split): accuracy ≈ 0.860, ROC AUC ≈ 0.930.

## Hyperparameter Sweep
- `config/clinvar_sweep.yaml` (hidden_size ∈ {256,384}, L_layers ∈ {2,3}, L_cycles ∈ {2,3}, lr ∈ {5e-4, 3e-4}).
- Run: `WANDB_MODE=offline DISABLE_COMPILE=1 python pretrain.py --config-name clinvar_sweep --multirun`
- Analyze results (once runs complete): `python scripts/analyze_sweep.py` → prints best ROC AUC and writes `sweep_summary.csv`.

## Tests
- `tests/test_clinvar_dataset.py`:
  - Ensures ClinSig-derived tokens do not enter the vocabulary.
  - Confirms sequence length = 24.
  - Verifies amino-acid property buckets, gene bucket logic, and that phenotype/evidence tokens are emitted when available.
  - Requires the balanced TSV built by preprocessing.
- Run with `python -m pytest tests/test_clinvar_dataset.py` (pytest must be installed in `trm_env`).

## Outstanding Work / Next Steps
1. **Phenotype utilization**
   - Explore richer phenotype encoding (e.g., HPO graph embeddings, multi-hot projections) and assess impact on ROC AUC.
2. **Additional features**
   - Integrate conservation/constraint features (gnomAD, phyloP) and structural annotations; revisit embedding strategy if vocab grows.
3. **Evaluation strategy**
   - Run stratified-by-gene/phenotype splits or cross-validation to gauge generalization; track calibration metrics.
4. **Housekeeping**
   - Silence regex deprecation warnings; add data-version stamps to outputs and document feature metadata consumption.

## Usage Reference (in `trm_env`)
```bash
# Rebuild dataset (default 5k per class; increase as needed)
python tools/prepare_clinvar_dataset.py --max-per-class 5000
python tools/build_clinvar_trm_dataset.py

# End-to-end convenience script (local only)
./run_clinvar_pipeline.sh --max-per-class 5000

# Run baseline
python tools/train_baseline_logreg.py \
    --input data/clinvar/processed/clinvar_missense_balanced.tsv \
    --output outputs/clinvar_logreg_metrics.json

# Train TRM (with early stopping)
WANDB_MODE=offline DISABLE_COMPILE=1 \
python pretrain.py --config-name cfg_clinvar_long +run_name=clinvar_long_5k +early_stop_patience=5

# Evaluate checkpoint + capture metrics/predictions
python tools/evaluate_clinvar_checkpoint.py --device cpu \
    --config checkpoints/Clinvar_trm-ACT-torch/clinvar_long_5k/all_config.yaml \
    --checkpoint checkpoints/Clinvar_trm-ACT-torch/clinvar_long_5k/step_1404 \
    --output outputs/clinvar_trm_metrics.json \
    --save-preds outputs/clinvar_trm_predictions.jsonl
# After every run/major analysis, append a summary to TinyVariant_log.md (date, command, key metrics).

# Ablations (optional)
python tools/build_clinvar_trm_dataset.py --phenotype-ablation
python tools/build_clinvar_trm_dataset.py --provenance-ablation

# Plot helpers (optional for documentation)
python scripts/plot_eval_comparison.py \
    --trm outputs/clinvar_trm_metrics.json \
    --baseline outputs/clinvar_logreg_metrics.json \
    --output docs/figures/clinvar_metric_comparison.png

python scripts/plot_roc_curve.py \
    --preds outputs/clinvar_trm_predictions.jsonl \
    --output docs/figures/clinvar_trm_roc.png

# Hyperparameter sweep and summary
WANDB_MODE=offline DISABLE_COMPILE=1 \
python pretrain.py --config-name clinvar_sweep --multirun
python scripts/analyze_sweep.py

# Tests
python -m pytest tests/test_clinvar_dataset.py

# Larger dataset build (50k+ per class)
python tools/prepare_clinvar_dataset.py --max-per-class 50000
python tools/build_clinvar_trm_dataset.py
```

## Notes
- Repo is currently several commits ahead of origin; push when ready.
- Local reference files (`Instructions.md`, ClinVar paper PDF) remain untracked.
- Keep wandb offline in the sandbox to avoid service failures.
