#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

# Edit the ratios to run (benign:malignant), comma-separated
RATIOS="${RATIOS:-5,10,20}"

if [[ -d "$VENV" ]]; then
    source "$VENV/bin/activate"
fi

echo "Running Model V3: Repeated Patient-Group CV Balanced Bagging..."
echo "Ratios: $RATIOS"
RATIOS="$RATIOS" python "$SCRIPT_DIR/models/model_v3/isic_model_v3_repeated_group_cv.py"
