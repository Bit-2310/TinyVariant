2025-10-19, 05:30 : System prep
- Verified repo structure and key components of the TRM codebase.
- Installed PyYAML, hydra-core, argdantic, and adam-atan2-pytorch into `trm_env`.
- Adjusted `pretrain.py` to use `AdamAtan2` with a non-zero LR and recorded the dependency in `requirements.txt`.

2025-10-19, 05:45 : ARC dataset work
- Generated the full ARC dataset (arc1concept-aug-1000) to validate augmentation behaviour.
- Added `dataset/__init__.py` and produced `data/arc1concept-mini` (800 puzzles, 801 IDs) for sanity checks.

2025-10-19, 06:00 : Training sanity checks (RTX 3050, 4 GB)
- Ran the tiny TRM config with `DISABLE_COMPILE=1`, `global_batch_size=4`, `hidden_size=128`, `num_heads=2`, `expansion=2`, `L_layers=1`, `L_cycles=2`, `H_cycles=1`.
- Confirmed the loop trains end-to-end, saving checkpoints and logging to WandB (`debug_run_tiny`).

2025-10-19, 06:10 : Next focus (TinyVariant POC)
- Ready to implement ClinVar ingestion and stand up the VariantTRM pipeline per the proof-of-concept plan.

2025-10-19, 06:30 : ClinVar dataset prep
- Added `VERSION` (0.1.0) and pulled pandas into `requirements.txt` for preprocessing work.
- Authored `tools/prepare_clinvar_dataset.py` to filter GRCh38 germline missense SNVs with high-confidence labels.
- Downloaded `variant_summary.txt.gz` (378 MB) via `data/download_data.sh`.
- Produced a balanced 5 k-sample dataset under `data/clinvar/processed` with versioned stats.

2025-10-19, 06:45 : ClinVar TRM dataset
- Tokenized the balanced variants into 13-token sequences covering gene, chromosome, alleles, amino-acid change, and position digits.
- Emitted train/test splits in `data/clinvar/processed/clinvar_trm` along with `vocab.json` and `identifiers.json`.
- Verified splits at 4 k train / 1 k test with label tokens `LABEL_BENIGN` (3) and `LABEL_PATHOGENIC` (4).

2025-10-19, 07:00 : ClinVar training setup
- Added `cfg_clinvar.yaml` and the `variant_trm` arch to point `pretrain.py` at the ClinVar dataset.
- Implemented `VariantClassificationHead` to supervise the final token via cross entropy on the binary labels.
- Attempted an offline smoke run; GPU wasn’t exposed inside the sandbox at the time.

2025-10-22, 19:30 : ClinVar smoke run on GPU
- Ran the 1-epoch `cfg_clinvar` smoke test locally on Pop!_OS (RTX 3050) with offline WandB logging.
- Training completed successfully; checkpoint and logs saved under `checkpoints/Clinvar_trm_ACT-torch/clinvar_smoke`.
- Metrics show the wiring works, but longer training and baselines are still pending for the full POC.

2025-10-22, 20:00 : Next steps (10k dataset & baselines)
- To scale the data, rerun `tools/prepare_clinvar_dataset.py --max-per-class 5000` followed by `tools/build_clinvar_trm_dataset.py` (yields 10 k balanced variants).
- Plan: train with higher epochs/batch size using `cfg_clinvar` overrides and compare against a logistic regression baseline.
- ClinVar evaluator and `evaluate_clinvar_checkpoint.py` will track accuracy/AUC for every run.

2025-10-22, 20:10 : Training configuration for extended run
- Added `cfg_clinvar_long.yaml` (50 epochs, LR 5e-4, batch 256) for 10 k experiments.
- Command: `WANDB_MODE=offline DISABLE_COMPILE=1 python pretrain.py --config-name cfg_clinvar_long +run_name=clinvar_long`
- Evaluator logs `ClinVar/accuracy` and `ClinVar/roc_auc` every 5 epochs; checkpoints saved after each evaluation.

2025-10-22, 20:20 : Logistic regression baseline helper
- Added `tools/train_baseline_logreg.py` (one-hot features + standardized position) as a quick comparison.
- Command: `python tools/train_baseline_logreg.py --input data/clinvar/processed/clinvar_missense_balanced.tsv`
- Outputs accuracy/ROC AUC to stdout (optional `--output` saves JSON); split matches the TRM preprocessing seed.

2025-10-22, 20:30 : First ClinVar training + baseline results
- Ran `cfg_clinvar_long` for 50 epochs (10 k data). Final evaluation: accuracy ≈ 0.7339, ROC AUC ≈ 0.8097.
- Logistic regression baseline (`tools/train_baseline_logreg.py`) achieved accuracy 0.7660, ROC AUC 0.8435.
- Baseline currently outperforms VariantTRM; next steps: feature engineering, architecture tweaks, or hyperparameter search.

2025-10-22, 21:15 : Added review status and ClinSig tokens
- Updated `tools/build_clinvar_trm_dataset.py` to insert review-status and clinical-significance tokens (sequence length now 15).
- Logistic baseline now one-hot encodes those fields as well.
- Reminder: regenerate data via
  ```bash
  python tools/prepare_clinvar_dataset.py --max-per-class 5000
  python tools/build_clinvar_trm_dataset.py
  ```
  before rerunning training/baseline.

2025-10-22, 22:20 : Results with enriched features
- `cfg_clinvar_long` (50 epochs, 10 k data): accuracy 0.9165, ROC AUC 0.9771.
- Logistic regression baseline: accuracy 0.9690, ROC AUC 0.9849 (feature leakage suspected).
- CPU evaluation script now supports loading checkpoints without CUDA.

2025-10-22, 22:40 : Removed ClinSig-derived leakage
- Stripped all ClinicalSignificance-based tokens from dataset builder and baseline.
- Logistic baseline (8k/2k split) now reports accuracy ≈0.746, ROC AUC ≈0.838.
- Request: regenerate data and rerun `cfg_clinvar_long` for fresh TRM metrics.

2025-10-22, 21:30 : TRM architecture alignment
- Re-read Samsung SAIT’s Tiny Recursive Models documentation to confirm architecture parity.
- Noted in README that we inherit TRM’s recursive reasoning core and only swap the input tokens/labels.
- Future changes: respect their embedding/halting conventions when adding new features.

2025-10-22, 19:50 : ClinVar evaluation utilities
- Added `tools/evaluate_clinvar_checkpoint.py` to score checkpoints with accuracy and ROC AUC on the ClinVar test split.
- Introduced `evaluators/clinvar.py` and wired it into `cfg_clinvar.yaml` so evaluation metrics log automatically during training.
- Updated the README Quickstart with a sample evaluation command.
2025-10-22, 23:30 : Amino-acid property features
- Added AA property buckets (nonpolar/charged/etc.) and change-class tokens to the TRM dataset.
- Logistic baseline (80k/20k split) after rebuild: accuracy 0.823, ROC AUC 0.907.
- Reran `cfg_clinvar_long` on the 50k-per-class dataset: accuracy 0.841, ROC AUC 0.915.
- Baseline remains at 0.823 / 0.907 with the enriched features.

2025-10-23, 00:30 : Added no-leakage test
- Created `tests/test_clinvar_dataset.py` to ensure ClinSig tokens never appear in the TRM vocab and sequence length matches expectations.
- README now documents how to run the quick pytest check.
