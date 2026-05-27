#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

if [[ -d "$VENV" ]]; then
    source "$VENV/bin/activate"
fi

echo "Running Model V1: Range-Based Malignant Similarity (Naive Bayes)..."
python "$SCRIPT_DIR/models/model_v1/isic_model_v1_naive_bayes_comparison.py"
