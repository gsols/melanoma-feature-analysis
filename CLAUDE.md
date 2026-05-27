# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Pipeline

All commands assume the virtual environment is active (`source .venv/bin/activate`) and you are in the project root.

```bash
# 1. Generate processed datasets (run once; required before any model)
python src/data_preprocessing_pipeline.py

# 2. Run individual models
bash run_model_v1.sh
bash run_model_v2.sh
bash run_model_v3.sh
bash run_model_v4.sh

# Run with custom ratios (V2, V3, V4)
RATIOS=5,10 bash run_model_v4.sh

# Smoke test for V3 (fewer folds/bags)
N_SPLITS=3 N_REPEATS=1 N_BAGS=3 N_ESTIMATORS=40 bash run_model_v3.sh

# Generate manual NB frequency tables (requires V1 to have run first)
python src/build_nb_classifier_dataset.py

# Run a model script directly
python models/model_v4/isic_model_v4_ratio_balanced.py
```

## Architecture Overview

### Data pipeline

```
datasetv1_raw_metadata.csv + datasetv2_labels.csv
  → data_preprocessing_pipeline.py
  → datasetv4_merged_cleaned_engineered.csv   ← primary dataset for all models
  → datasetv5_cleaned_all_malignant.csv       ← reference profile for V1
  → datasetv6_cleaned_all_benign.csv          ← training targets for V1
```

All models in `models/` read `datasetv4` directly via a path resolved relative to the script's own location (`_HERE = Path(__file__).resolve().parent`). V3 is the exception — it reads `datasetv3` and re-applies the same preprocessing inline.

### Model versioning

| Model | File | Algorithm | Distinguishing design |
|---|---|---|---|
| V1 | `model_v1/isic_model_v1_naive_bayes_comparison.py` | GaussianNB + CategoricalNB | Trains only on benign records; malignant used only for IQR reference windows |
| V2 | `model_v2/isic_model_v2_balanced_bagging.py` | LightGBM bagging | Balanced undersampling; configurable benign:malignant ratios |
| V3 | `model_v3/isic_model_v3_repeated_group_cv.py` | LightGBM bagging | Patient-group CV to prevent patient-level data leakage |
| V4 | `model_v4/isic_model_v4_ratio_balanced.py` | LightGBM | SMOTE oversampling + random undersampling; tests extreme ratios up to 1000:1 |

### Output structure pattern

Each model version writes to its own subdirectory under `outputs/`:

```
outputs/
  v1_outputs/graphs/ + reports/
  v2_outputs/v2_outputs_ratio{N}_1/graphs/ + reports/
  v3_outputs/v3_outputs_ratio{N}_1/graphs/ + reports/
  v4_outputs/v4_outputs_ratio{N}_1/graphs/ + reports/
  v4_outputs/v4_ratio_model_registry.csv   ← V4 cross-ratio summary
```

Standard output files per model run:
- `v{N}_readable_report.txt` — human-readable summary
- `v{N}_global_metrics.json` — machine-readable metrics
- `v{N}_oof_confusion_matrix.png`, `v{N}_oof_precision_recall_curve.png`, `v{N}_feature_importance.png`
- `v{N}_ratio_{R}_benign_risk_predictions.csv` — per-record risk scores

### Feature set (19 total in datasetv4)

11 raw TBP features: `age_approx`, `clin_size_long_diam_mm`, `tbp_lv_area_perim_ratio`, `tbp_lv_norm_border`, `tbp_lv_symm_2axis`, `tbp_lv_eccentricity`, `tbp_lv_color_std_mean`, `tbp_lv_norm_color`, `tbp_lv_deltaLBnorm`, `tbp_lv_radial_color_std_max`, `tbp_lv_nevi_confidence`

8 engineered features (computed in `data_preprocessing_pipeline.py` and replicated in `add_engineered_features()` in V3):

| Feature | Formula |
|---|---|
| `color_contrast_3d` | `sqrt(deltaA² + deltaB² + deltaL²)` |
| `elongation` | `clin_size_long_diam_mm / (tbp_lv_minorAxisMM + ε)` |
| `nevi_color_tension` | `tbp_lv_nevi_confidence - tbp_lv_color_std_mean` |
| `log_area` | `log1p(tbp_lv_areaMM2)` |
| `stdL_ratio` | `tbp_lv_stdL / (tbp_lv_stdLExt + ε)` |
| `compactness` | `4π × tbp_lv_areaMM2 / (tbp_lv_perimeterMM² + ε)` |
| `chroma_contrast` | `tbp_lv_C - tbp_lv_Cext` |
| `radial_color_ratio` | `tbp_lv_radial_color_std_max / (tbp_lv_color_std_mean + ε)` |

3 categorical columns also used by models: `sex`, `anatom_site_general`, `tbp_lv_location_simple`

### 170melanoma-main/ subfolder

This subfolder is not part of the final repository and can be ignored. Do not reference it or derive any context from it when working with the models in `models/`.

### Shell script pattern

Each `run_model_v{N}.sh` activates `.venv` if present, then passes environment variables (like `RATIOS`) through to the Python script. New model shell scripts should follow the same pattern as `run_model_v4.sh`.

## Adding a New Model Version

When adding a model (e.g., model_v5):
1. Create `models/model_v5/<descriptive_name>.py` — use `_HERE = Path(__file__).resolve().parent` for path resolution; point `INPUT_DATA` two levels up to `datasets/datasetv4_merged_cleaned_engineered.csv`
2. Create `models/model_v5/processed_data/` for intermediate CSVs
3. Write outputs to `outputs/v5_outputs/` using the same `reports/` + `graphs/` subdirectory structure
4. Create `run_model_v5.sh` following the pattern of `run_model_v4.sh`
5. Update `README.md` and `documentation.txt` with the new model entry
