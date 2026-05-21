#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

# Edit the ratios to run (benign:malignant), comma-separated
RATIOS="${RATIOS:-5,10,20}"

if [[ -d "$VENV" ]]; then
    source "$VENV/bin/activate"
fi

echo "Running Model V2: Honest Melanoma Risk Ranking (Balanced Bagging)..."
echo "Ratios: $RATIOS"
RATIOS="$RATIOS" python "$SCRIPT_DIR/models/model_v2/isic_model_v2_balanced_bagging.py"
