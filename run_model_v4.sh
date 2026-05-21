#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

# Edit the ratios to run (benign:malignant), comma-separated
RATIOS="${RATIOS:-5,10,15,20,25,30,500,600,700,1000}"

if [[ -d "$VENV" ]]; then
    source "$VENV/bin/activate"
fi

echo "Running Model V4: LightGBM Ratio-Balanced (SMOTE + undersampling)..."
echo "Ratios: $RATIOS"
RATIOS="$RATIOS" python "$SCRIPT_DIR/models/model_v4/isic_model_v4_ratio_balanced.py"
