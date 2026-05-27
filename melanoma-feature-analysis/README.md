# Melanoma Feature Analysis
### ISIC 2024 Metadata-Based Malignancy Risk System

A machine learning system that assigns every benign skin lesion record a **malignancy risk probability** — a score between 0 and 1 describing how closely that benign lesion resembles a confirmed malignant melanoma based on its measurable features.

---

## Core Concept

The system uses confirmed malignant cases only as a **reference profile**, never as training samples. For each of 19 skin features, it computes the interquartile range (IQR) across all confirmed malignant records. A benign lesion receives a similarity score: the fraction of its features that fall inside those malignant IQR windows.

- Score ≥ 0.60 → labeled **near-malignant**
- The Naive Bayes classifier then learns to predict this label and outputs a probability for every benign record

> This does **not** replace medical diagnosis. It is a research and screening aid.

---

## Project Structure

```
melanoma-feature-analysis/
│
├── datasets/
│   ├── datasetv1_raw_metadata.csv                    # Original ISIC 2024 metadata (raw)
│   ├── datasetv2_labels.csv                          # Malignant/benign labels per ISIC ID
│   ├── datasetv3_raw_metadata_labels_merged.csv      # Merged (all original columns)
│   ├── datasetv4_merged_cleaned_engineered.csv       # Cleaned + engineered features (primary)
│   ├── datasetv5_cleaned_all_malignant.csv           # Malignant records only
│   └── datasetv6_cleaned_all_benign.csv              # Benign records only
│
├── src/
│   ├── data_preprocessing_pipeline.py               # Generates datasets v3–v6
│   └── build_nb_classifier_dataset.py               # Builds manual NB frequency tables
│
├── models/
│   ├── model_v1/   isic_model_v1_naive_bayes_comparison.py   # Main model
│   ├── model_v2/   isic_model_v2_balanced_bagging.py
│   ├── model_v3/   isic_model_v3_repeated_group_cv.py
│   └── model_v4/   isic_model_v4_ratio_balanced.py
│
├── nb_classifier_model_v1/                          # Manual NB frequency tables
│   ├── nb_classifier_model_v1.xlsx                  # All features in one Excel workbook
│   └── <feature>_near_malignant_counts.csv          # One CSV per feature
│
├── outputs/
│   ├── v1_outputs/   graphs/ + reports/
│   ├── v2_outputs/   v2_outputs_ratio{5,10,20}_1/
│   ├── v3_outputs/   v3_outputs_ratio{5,10,20}_1/
│   └── v4_outputs/   v4_outputs_ratio<N>_1/
│
├── run_model_v1.sh
├── run_model_v2.sh
├── run_model_v3.sh
├── run_model_v4.sh
└── requirements.txt
```

---

## Prerequisites

- Python 3.10 or newer
- pip
- bash (Linux/macOS built-in; on Windows use Git Bash or WSL)

Verify Python is installed:
```bash
python --version
# or
python3 --version
```

---

## Environment Setup

```bash
# 1. Navigate to the project folder
cd /path/to/melanoma-feature-analysis

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
source .venv/bin/activate          # Linux / macOS
source .venv/Scripts/activate      # Windows (Git Bash)
.venv\Scripts\activate.bat         # Windows (Command Prompt)

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify (optional)
python -c "import sklearn, pandas, numpy, matplotlib, lightgbm, openpyxl; print('All packages OK')"
```

**Dependencies:** scikit-learn 1.8, pandas 3.0, numpy 2.4, matplotlib 3.10, seaborn 0.13, lightgbm 4.6, openpyxl 3.1, scipy 1.17

---

## Required Data Files

Before running anything, confirm these two files exist in `datasets/`:
- `datasetv1_raw_metadata.csv`
- `datasetv2_labels.csv`

All other dataset files (v3–v6) are generated automatically by the preprocessing pipeline.

---

## Running the Full Pipeline

Follow these steps in order for a first-time setup or a full regeneration from scratch.

### Step A — Activate the virtual environment
```bash
source .venv/bin/activate
```

### Step B — Generate processed datasets (run once)
```bash
python src/data_preprocessing_pipeline.py
```
Reads `datasetv1` and `datasetv2`, produces `datasetv3` through `datasetv6`. Expected output:
```
datasetv4_merged_cleaned_engineered.csv : 401,059 rows x 22 cols
datasetv5_cleaned_all_malignant.csv     : 393 rows
datasetv6_cleaned_all_benign.csv        : 400,666 rows
```

### Step C — Run Model V1 (main model)
```bash
bash run_model_v1.sh
```

### Step D — Generate manual NB frequency tables (optional, run after V1)
```bash
python src/build_nb_classifier_dataset.py
```

### Step E — Run comparative models (optional, any order)
```bash
bash run_model_v2.sh
bash run_model_v3.sh
bash run_model_v4.sh
```

---

## Models

### Model V1 — Naive Bayes Comparison (Main)
`models/model_v1/isic_model_v1_naive_bayes_comparison.py`

Trains and compares two Naive Bayes classifiers:
- **GaussianNB** — operates on raw numeric feature values
- **CategoricalNB** — operates on binned/categorized versions (e.g., age → "Child", "Young Adult", "Middle Age", "Senior")

Training uses only benign records. Malignant records define the IQR reference windows only.

**Outputs:**
- `outputs/v1_outputs/graphs/features/` — 19 feature distribution charts (benign vs. malignant)
- `outputs/v1_outputs/graphs/analysis/` — confusion matrices, ROC curve, PR curve, metrics bar chart, probability distributions
- `outputs/v1_outputs/reports/v1_readable_report.txt` — human-readable summary
- `outputs/v1_outputs/reports/v1_naive_bayes_predictions.csv` — per-record predictions sorted by probability

**Key metrics to look at:** F1, ROC-AUC (>0.85 is strong), PR-AUC

---

### Manual NB Classifier (`nb_classifier_model_v1`)
`src/build_nb_classifier_dataset.py`

Produces human-readable Naive Bayes frequency tables. For each of the 19 features, shows per-category counts and conditional probabilities P(category | near_malignant) and P(category | not near_malignant). Results are compiled into `nb_classifier_model_v1.xlsx` (one sheet per feature).

Run this **after** Model V1.

---

### Model V2 — Balanced Bagging (Comparative)
`models/model_v2/isic_model_v2_balanced_bagging.py`

Trains on both benign and malignant records using a balanced undersampling ensemble. Each bag contains all malignant training rows plus a random benign subset at a configurable ratio. Answers: how does a standard bagging approach compare to V1's IQR similarity approach?

**Default ratios:** 5:1, 10:1, 20:1

```bash
bash run_model_v2.sh            # default ratios
RATIOS=5,10 bash run_model_v2.sh  # custom ratios
```

---

### Model V3 — Repeated Patient-Group CV (Comparative)
`models/model_v3/isic_model_v3_repeated_group_cv.py`

Extends V2 with patient-aware cross-validation — patients are never split across train and test folds. This prevents performance inflation from multiple lesion images per patient and gives a more honest generalization estimate.

**Default ratios:** 5:1, 10:1, 20:1

```bash
bash run_model_v3.sh
# Smoke test (fewer folds/bags):
N_SPLITS=3 N_REPEATS=1 N_BAGS=3 N_ESTIMATORS=40 bash run_model_v3.sh
```

---

### Model V4 — LightGBM Ratio-Balanced (Experimental)
`models/model_v4/isic_model_v4_ratio_balanced.py`

Uses LightGBM with a two-pronged class balancing strategy:
- **SMOTE** — generates synthetic malignant training samples
- **Random undersampling** — reduces benign records to the target ratio

Tested across extreme ratios (up to 1000:1) to explore how much class imbalance LightGBM can tolerate.

**Default ratios:** 5, 10, 15, 20, 25, 30, 500, 600, 700, 1000

```bash
bash run_model_v4.sh                   # all default ratios (slow)
RATIOS=5,10,20 bash run_model_v4.sh   # quick run
```

A summary across all ratios is saved to `outputs/v4_outputs/v4_ratio_model_registry.csv`.

---

## Features Analyzed (19 total)

| Feature | Description |
|---|---|
| `age_approx` | Patient age |
| `clin_size_long_diam_mm` | Lesion diameter (mm) |
| `tbp_lv_area_perim_ratio` | Area-to-perimeter ratio |
| `tbp_lv_norm_border` | Border irregularity score |
| `tbp_lv_symm_2axis` | Shape symmetry score |
| `tbp_lv_eccentricity` | Eccentricity (0 = circle, 1 = line) |
| `tbp_lv_color_std_mean` | Internal color variation |
| `tbp_lv_norm_color` | Normalized color score |
| `tbp_lv_deltaLBnorm` | Lightness difference vs. surrounding skin |
| `tbp_lv_radial_color_std_max` | Radial color variation (max) |
| `tbp_lv_nevi_confidence` | Nevi (mole) confidence score |
| `color_contrast_3d` | 3D color contrast *(engineered)* |
| `elongation` | Shape elongation ratio *(engineered)* |
| `nevi_color_tension` | Nevi confidence minus color variation *(engineered)* |
| `log_area` | Log-transformed lesion area *(engineered)* |
| `stdL_ratio` | Lightness standard deviation ratio *(engineered)* |
| `compactness` | Shape compactness index *(engineered)* |
| `chroma_contrast` | Chroma (color saturation) contrast *(engineered)* |
| `radial_color_ratio` | Ratio of radial to mean color variation *(engineered)* |

---

## Dataset Pipeline

```
datasetv1_raw_metadata.csv  +  datasetv2_labels.csv
          │
          ▼  (data_preprocessing_pipeline.py)
datasetv3: merged (all columns)
          │
          ▼  fill missing values (median / mode)
          │
          ▼  clip outliers to safe ranges
          │
          ▼  engineer 8 new features
          │
datasetv4: 22 columns (19 numeric features + sex, anatomy site, location, label)
          ├──▶ datasetv5: malignant records only (393 rows) — V1 reference profile
          └──▶ datasetv6: benign records only (400,666 rows) — V1 training targets
```

---

## Understanding the Outputs

| File | Description |
|---|---|
| `confusion_matrix.png` | TP/TN/FP/FN grid |
| `roc_*.png` | ROC curve — AUC > 0.85 indicates strong performance |
| `pr_*.png` | Precision-Recall curve |
| `feature_importance.png` | Feature contribution ranking (V2, V3, V4 only) |
| `*_readable_report.txt` | Human-readable summary (metrics, confusion matrix, IQR windows) |
| `*_metrics.json` | Machine-readable metrics in JSON |
| `*_predictions.csv` | Per-record predictions sorted by probability (V1 only) |

**Comparing models:** look at ROC-AUC (overall discrimination), PR-AUC (finding true near-malignant cases), and F1 (precision/recall balance) side by side across V1–V4 reports.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `python: command not found` | Try `python3` instead; install from python.org if missing |
| `No such file or directory: datasetv4_...` | Run `python src/data_preprocessing_pipeline.py` first |
| `No such file or directory: v1_train_benign_similarity.csv` | Run `bash run_model_v1.sh` before `build_nb_classifier_dataset.py` |
| `ModuleNotFoundError: No module named 'sklearn'` | Activate the venv (`source .venv/bin/activate`) then `pip install -r requirements.txt` |
| `Permission denied` on `.sh` file | Run `chmod +x run_model_v1.sh` then retry |
| V4 is taking very long | Limit ratios: `RATIOS=5,10 bash run_model_v4.sh` |
| Output graphs folder is empty | Check terminal for Python errors — the script prints progress as it runs |
| `build_nb_classifier_dataset.py` produces empty tables | Run Model V1 fully first (`bash run_model_v1.sh`) |
