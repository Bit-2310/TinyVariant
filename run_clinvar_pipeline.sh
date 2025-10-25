#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_clinvar_pipeline.sh [options]

Options:
  -m, --max-per-class N   Number of examples per class for preprocessing (default: 5000)
  -n, --run-name NAME     Override Hydra run name for TRM training (default: timestamped)
      --cpu               Force evaluation on CPU (training still uses available GPU)
      --phenotype-ablation   Drop phenotype tokens when building the TRM dataset
      --provenance-ablation  Drop submitter/evaluation provenance tokens when building the TRM dataset
      --train-arg ARG      Additional override passed to pretrain.py (repeatable)
      --skip-baseline      Skip logistic regression baseline step
      --skip-train         Skip TRM training (implies --skip-eval --skip-plots)
      --skip-eval          Skip checkpoint evaluation
      --skip-plots         Skip plotting helpers
      --help               Show this message

Examples:
  ./run_clinvar_pipeline.sh -m 50000 --train-arg '+early_stop_patience=8'
  ./run_clinvar_pipeline.sh --phenotype-ablation --skip-plots
EOF
}

MAX_PER_CLASS=5000
RUN_NAME=""
DEVICE="cuda"
PHENO_ABLATION=false
PROV_ABLATION=false
SKIP_BASELINE=false
SKIP_TRAIN=false
SKIP_EVAL=false
SKIP_PLOTS=false
TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--max-per-class)
      MAX_PER_CLASS=$2
      shift 2
      ;;
    -n|--run-name)
      RUN_NAME=$2
      shift 2
      ;;
    --cpu)
      DEVICE="cpu"
      shift
      ;;
    --phenotype-ablation)
      PHENO_ABLATION=true
      shift
      ;;
    --provenance-ablation)
      PROV_ABLATION=true
      shift
      ;;
    --train-arg)
      TRAIN_ARGS+=("$2")
      shift 2
      ;;
    --skip-baseline)
      SKIP_BASELINE=true
      shift
      ;;
    --skip-train)
      SKIP_TRAIN=true
      SKIP_EVAL=true
      SKIP_PLOTS=true
      shift
      ;;
    --skip-eval)
      SKIP_EVAL=true
      shift
      ;;
    --skip-plots)
      SKIP_PLOTS=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -d data ]]; then
  echo "This script must be run from the TinyVariant repository root." >&2
  exit 1
fi

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="clinvar_long_$(date +%Y%m%d-%H%M%S)"
fi

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

PYTHON_CMD=(python)
if ! python - <<'PY' >/dev/null 2>&1
import torch
import numpy
PY
then
  if command -v conda >/dev/null 2>&1; then
    PYTHON_CMD=(conda run -n trm_env python)
    if ! "${PYTHON_CMD[@]}" - <<'PY' >/dev/null 2>&1
import torch
import numpy
PY
    then
      echo "Unable to locate required Python environment (torch/numpy missing)." >&2
      exit 1
    fi
  else
    echo "Python environment missing torch/numpy and conda is unavailable." >&2
    exit 1
  fi
fi

run_python() {
  "${PYTHON_CMD[@]}" "$@"
}

log "Step 1/6: Preparing ClinVar dataset (max_per_class=${MAX_PER_CLASS})"
run_python tools/prepare_clinvar_dataset.py --max-per-class "$MAX_PER_CLASS"

log "Step 2/6: Building TRM dataset"
BUILD_ARGS=()
$PHENO_ABLATION && BUILD_ARGS+=("--phenotype-ablation")
$PROV_ABLATION && BUILD_ARGS+=("--provenance-ablation")
run_python tools/build_clinvar_trm_dataset.py "${BUILD_ARGS[@]}"

if ! $SKIP_BASELINE; then
  log "Step 3/6: Training logistic regression baseline"
  run_python tools/train_baseline_logreg.py \
    --input data/clinvar/processed/clinvar_missense_balanced.tsv \
    --output outputs/clinvar_logreg_metrics.json | tee outputs/clinvar_logreg_metrics.log
else
  log "Skipping baseline step"
fi

if ! $SKIP_TRAIN; then
  log "Step 4/6: Training TinyVariant TRM run ${RUN_NAME}"
  WANDB_MODE=offline DISABLE_COMPILE=1 \
    "${PYTHON_CMD[@]}" pretrain.py --config-name cfg_clinvar_long \
    +run_name="${RUN_NAME}" +early_stop_patience=5 "${TRAIN_ARGS[@]}"
else
  log "Skipping TRM training"
fi

RUN_DIR="checkpoints/Clinvar_trm-ACT-torch/${RUN_NAME}"
LATEST_STEP=""
if ! $SKIP_EVAL; then
  if [[ ! -d "$RUN_DIR" ]]; then
    echo "Expected run directory ${RUN_DIR} not found. Cannot evaluate." >&2
    exit 1
  fi
  LATEST_STEP=$(ls "$RUN_DIR"/step_* 2>/dev/null | sort -V | tail -n 1 || true)
  if [[ -z "$LATEST_STEP" ]]; then
    echo "No checkpoint files found under ${RUN_DIR}." >&2
    exit 1
  fi
  log "Step 5/6: Evaluating checkpoint $(basename "$LATEST_STEP") on ${DEVICE}"
  run_python tools/evaluate_clinvar_checkpoint.py \
    --config "$RUN_DIR/all_config.yaml" \
    --checkpoint "$LATEST_STEP" \
    --device "$DEVICE" \
    --output outputs/clinvar_trm_metrics.json \
    --save-preds outputs/clinvar_trm_predictions.jsonl | tee outputs/clinvar_trm_metrics.log
else
  log "Skipping evaluation"
fi

if ! $SKIP_PLOTS; then
  log "Step 6/6: Generating plots"
  mkdir -p docs/figures
  if [[ -f outputs/clinvar_trm_metrics.json ]]; then
    if [[ -f outputs/clinvar_logreg_metrics.json ]]; then
      run_python scripts/plot_eval_comparison.py \
        --trm outputs/clinvar_trm_metrics.json \
        --baseline outputs/clinvar_logreg_metrics.json \
        --output docs/figures/clinvar_metric_comparison.png
    elif ! $SKIP_BASELINE; then
      log "  Warning: baseline metrics not found; skipping comparison plot"
    fi
    if [[ -f outputs/clinvar_trm_predictions.jsonl ]]; then
      run_python scripts/plot_roc_curve.py \
        --preds outputs/clinvar_trm_predictions.jsonl \
        --output docs/figures/clinvar_trm_roc.png
    else
      log "  Warning: TRM prediction file missing; skipping ROC plot"
    fi
  else
    log "  Warning: TRM metrics file missing; skipping plots"
  fi
else
  log "Skipping plotting"
fi

log "Pipeline complete. Outputs saved under outputs/ and docs/figures/."
