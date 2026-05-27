#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

# Number of Optuna trials (default 50; use 10 for a quick smoke test)
N_TRIALS="${N_TRIALS:-50}"

# Number of CV folds (default 5)
N_FOLDS="${N_FOLDS:-5}"

# Minimum recall target for the Medium|High risk tier boundary (default 0.50)
RECALL_TARGET="${RECALL_TARGET:-0.50}"

if [[ -d "$VENV" ]]; then
    source "$VENV/bin/activate"
fi

echo "Running Model V5: LightGBM + Optuna Full-Data Tuning + Platt Calibration"
echo "N_TRIALS=$N_TRIALS  N_FOLDS=$N_FOLDS  RECALL_TARGET=$RECALL_TARGET"
N_TRIALS="$N_TRIALS" N_FOLDS="$N_FOLDS" RECALL_TARGET="$RECALL_TARGET" \
    python "$SCRIPT_DIR/models/model_v5/isic_model_v5_optuna_lgbm_platt.py"
