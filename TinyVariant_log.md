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

2025-10-23, 00:45 : Hyperparameter sweep config
- Added `config/clinvar_sweep.yaml` to drive Hydra multiruns over hidden size, L layers, L cycles, and learning rate.
- README documents the sweep command (`python pretrain.py --config-name clinvar_sweep --multirun`).

2025-10-23, 19:00 : Integrated sweep analyzer
- Updated `config/clinvar_sweep.yaml` to rely on Hydra override directories.
- Added `scripts/analyze_sweep.py` and documented the workflow (run sweep + analyzer).

2025-10-23, 19:20 : Added early stopping support
- `PretrainConfig` now accepts `early_stop_patience`, `early_stop_metric`, and `early_stop_delta`.
- Training loop stops when the chosen metric fails to improve beyond `early_stop_delta` for `patience` evaluations.
- README documents usage via `+early_stop_patience=...`.

2025-10-24, 09:10 : ClinVar preprocessing enrichment
- Expanded `tools/prepare_clinvar_dataset.py` to retain phenotype terms/IDs, submitter counts, ISO-normalized review dates, and derived provenance buckets.
- Normalized phenotype strings (dedupe, placeholder filtering) and tracked coverage stats in `clinvar_missense_balanced_stats.json`.
- Verified pipeline with a 50-per-class smoke build under `data/clinvar/processed/tmp_balanced`.

2025-10-24, 09:40 : Phenotype context in TRM + baselines
- Updated `tools/build_clinvar_trm_dataset.py` to emit phenotype/source/submitter/evaluation tokens (sequence length 24) and persist feature metadata in dataset manifests.
- Synced logistic regression baseline with the new context features via shared preprocessing helpers.
- Extended pytest coverage to assert phenotype tokens are present and leak-free.

2025-10-24, 10:15 : Baseline runner ergonomics
- Added `tools/__init__.py` and path bootstrapping so `python tools/train_baseline_logreg.py` works without manual `PYTHONPATH` tweaks.
- Successfully reran the logistic regression baseline on the refreshed 5k/5k dataset (accuracy ≈0.861, ROC AUC ≈0.930).

2025-10-24, 17:25 : VariantTRM run with phenotype context
- Trained `cfg_clinvar_long` on the 5k/5k dataset with `+early_stop_patience=5` (offline WandB run `clinvar_long_5k`).
- Final evaluation: accuracy ≈0.819, ROC AUC ≈0.884; checkpoints saved under `checkpoints/Clinvar_trm-ACT-torch/clinvar_long_5k/`.
- Noted the performance gap vs. the logistic baseline, highlighting the need for further feature/model work.

2025-10-24, 17:45 : Expanded ClinVar dataset tests
- Augmented `tests/test_clinvar_dataset.py` to verify phenotype/provenance buckets, feature metadata, and updated sequence length (now 25).
- All ClinVar tests pass against the refreshed 5k/5k dataset (`python -m pytest tests/test_clinvar_dataset.py` → 5 passed).

2025-10-24, 18:10 : Added tertiary phenotype slots
- Extended `tools/build_clinvar_trm_dataset.py` to encode up to three phenotype tokens (sequence length 25) and preserved slot metadata.
- Updated the logistic baseline, docs, and tests to stay in sync; new baseline run: accuracy ≈0.860, ROC AUC ≈0.930.

2025-10-24, 18:55 : ClinVar dataset rebuild + run recap
- Regenerated the balanced ClinVar table (5k per class) and rebuilt the TRM dataset so phenotype/provenance fields populate all tokens.
- Trained `cfg_clinvar_long` with `+early_stop_patience=5` (`clinvar_long_20251024-175518`); early stopping fired at step 1248, final eval accuracy 0.819, ROC AUC 0.8859.
- Logistic baseline rerun on the refreshed dataset: accuracy 0.860, ROC AUC 0.9300 (8k/2k split).
- Evaluated `step_1248` with both CPU and CUDA modes after fixing `tools/evaluate_clinvar_checkpoint.py` to create carries on-device; scores match training (accuracy 0.8195, ROC AUC 0.8858).

2025-10-24, 19:20 : Plotting helpers for documentation
- Extended `tools/evaluate_clinvar_checkpoint.py` with `--save-preds` so per-example scores can be captured alongside summary metrics.
- Added `scripts/plot_eval_comparison.py` to visualize TinyVariant vs. baseline accuracy/AUC from saved JSON metrics.
- Added `scripts/plot_roc_curve.py` to render ROC curves from prediction JSONL files (pairs with the new evaluation flag).
- README Quickstart now documents the evaluation output flags and plotting workflow for future write-ups.

2025-10-24, 19:40 : Feature ablation toggles + scaling notes
- Added `--phenotype-ablation` and `--provenance-ablation` flags to `tools/build_clinvar_trm_dataset.py` so we can rebuild datasets with phenotype tokens or provenance buckets zeroed out.
- Feature metadata now records the ablation state; downstream baselines/TRM runs pick up the altered dataset automatically.
- README/Instructions document how to run ablation rebuilds and how to scale preprocessing to 50k+ examples via `--max-per-class`.

2025-10-24, 21:05 : 50k-per-class ClinVar run results
- Rebuilt the balanced dataset with `--max-per-class 50000` (100k examples total; ~61k carry phenotype annotations) and refreshed the TRM arrays.
- Logistic baseline on the 80k/20k split reached accuracy 0.8955, ROC AUC 0.9591.
- Trained `cfg_clinvar_long` with early stopping (run `clinvar_long_20251024-194100`); stopped at step 9372 with eval accuracy 0.8828 and ROC AUC 0.9450.
- Updated `tools/evaluate_clinvar_checkpoint.py` to move carries to the active device so GPU evaluation succeeds; metrics for `step_9372` are stored in `outputs/clinvar_trm_metrics.json` and the ROC curve in `docs/figures/clinvar_trm_roc.png`.

2025-10-24, 21:55 : Phenotype ablation baseline
- Rebuilt TRM dataset with `--phenotype-ablation` (phenotype slots forced to `<none>`).
- TRM run `clinvar_long_phenotype_ablation_20251024-215110` (W&B disabled, single-worker loader) early-stopped at step 14058: eval accuracy 0.8741, ROC AUC 0.9417.
- Evaluated checkpoint `step_14058` on CPU (GPU evaluation timing out) → metrics saved to `outputs/clinvar_long_phenotype_ablation_20251024-215110_metrics.json`, predictions in `_predictions.jsonl`.

2025-10-25, 07:55 : Provenance ablation baseline
- Rebuilt TRM dataset with `--provenance-ablation` (submitter/evaluation buckets set to the lowest bucket).
- TRM run `clinvar_long_provenance_ablation_20251025-074717` (WANDB offline, single-worker loader) completed full schedule; eval accuracy 0.8755, ROC AUC 0.9438.
- Evaluated checkpoint `step_15620` on CPU → metrics in `outputs/clinvar_long_provenance_ablation_20251025-074717_metrics.json`, predictions in `_predictions.jsonl`.
