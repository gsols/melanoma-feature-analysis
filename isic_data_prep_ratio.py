# =============================================================================
# ISIC 2024 — Ratio-based Resampling (SMOTE + Undersampling)
# =============================================================================
# Creates balanced training datasets with configurable class ratios
# using SMOTE for oversampling malignant cases and random undersampling
# for benign cases.
#
# Approach: Extract 20% test set first (before resampling), then apply
# ratio-based resampling to the remaining 80% training data to avoid leakage.
#
# Output files:
#   ratio_data/test_ratio_shared.csv        — test set (shared across all ratios)
#   ratio_data/train_ratio_{ratio}.csv      — resampled training set
#   ratio_data/ratio_{ratio}_summary.txt    — resampling summary
#
# Install once:
#   pip install imbalanced-learn scikit-learn
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import os
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DATA = 'merged_cleaned.csv'  # Original ~401k data
OUTPUT_DIR  = 'ratio_data'

# Ratios to generate (benign:malignant)
# e.g., 20 means 20 benign for every 1 malignant
RATIOS = [5, 10, 15, 20, 25, 30]

# Test set: extract 20% before resampling (to avoid data leakage)
TEST_SIZE = 0.20

# Target total records per training dataset (~300k)
TARGET_TOTAL_ROWS = 300_000

RANDOM_STATE = 42


# =============================================================================
# STEP 0 — Create output directory
# =============================================================================

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory: {OUTPUT_DIR}")


# =============================================================================
# STEP 1 — Load original data and create test set
# =============================================================================

print("=" * 70)
print("ISIC 2024 — Ratio-based Resampling with SMOTE + Undersampling")
print("=" * 70)

print("\nSTEP 1 — Loading original data and splitting")
print("-" * 70)

df_full = pd.read_csv(INPUT_DATA)
print(f"  Full data: {df_full.shape[0]:,} rows x {df_full.shape[1]} columns")
print(f"    Malignant: {df_full['malignant'].sum():,}")
print(f"    Benign   : {(df_full['malignant']==0).sum():,}")
print(f"    Ratio    : {(df_full['malignant']==0).sum() / df_full['malignant'].sum():.1f}:1")

# Split: all malignant samples go to train, 20% benign goes to test
benign_mask = df_full['malignant'] == 0
df_benign = df_full[benign_mask].reset_index(drop=True)
df_malignant = df_full[~benign_mask].reset_index(drop=True)

# Split benign samples: 80% train, 20% test
df_benign_train, df_test = train_test_split(
    df_benign,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

# Combine: train = all malignant + 80% benign, test = 20% benign (all benign)
df_train = pd.concat([df_malignant, df_benign_train], ignore_index=True)

print(f"\n  After splitting (all malignant → train, 20% benign → test):")
print(f"    Train: {df_train.shape[0]:,} rows")
print(f"      Malignant: {df_train['malignant'].sum():,} (all samples)")
print(f"      Benign   : {(df_train['malignant']==0).sum():,} (80% of benign)")
print(f"      Ratio    : {(df_train['malignant']==0).sum() / df_train['malignant'].sum():.1f}:1")

print(f"\n    Test: {df_test.shape[0]:,} rows (SHARED across all ratio configurations)")
print(f"      Malignant: {df_test['malignant'].sum():,} (none)")
print(f"      Benign   : {(df_test['malignant']==0).sum():,} (20% of benign)")
print(f"      Ratio    : all benign (negative samples only)")


# =============================================================================
# STEP 2 — Prepare features and target (encode categorical variables)
# =============================================================================

print("\nSTEP 2 — Preparing features and target")
print("-" * 70)

X_train = df_train.drop(columns=['malignant']).copy()
X_test = df_test.drop(columns=['malignant']).copy()
y_train = df_train['malignant']
y_test = df_test['malignant']

# Identify and encode categorical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X_train.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns.tolist()

print(f"  Categorical columns: {len(categorical_cols)} → {categorical_cols}")
print(f"  Numeric columns    : {len(numeric_cols)}")

# One-hot encode categorical variables
if categorical_cols:
    print(f"  Encoding categorical variables...")
    X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

    # Ensure train and test have same columns after encoding
    train_cols = set(X_train.columns)
    test_cols = set(X_test.columns)

    # Add missing columns to test with 0
    for col in train_cols - test_cols:
        X_test[col] = 0

    # Remove extra columns from test
    X_test = X_test[X_train.columns]

    print(f"  ✓ Categorical variables encoded")

print(f"  Total feature columns: {X_train.shape[1]}")
print(f"  Target variable: 'malignant'")


# =============================================================================
# STEP 3 — Save shared test set (once, used by all ratio configurations)
# =============================================================================

print("\nSTEP 3 — Saving shared test set")
print("-" * 70)

test_filename = os.path.join(OUTPUT_DIR, 'test_ratio_shared.csv')
df_test.to_csv(test_filename, index=False)
print(f"  ✓ Saved: {test_filename}")
print(f"    This test set will be shared across all {len(RATIOS)} ratio configurations")


# =============================================================================
# STEP 4 — Generate resampled training datasets for each ratio
# =============================================================================

print("\nSTEP 4 — Generating resampled training datasets")
print("-" * 70)

for target_ratio in RATIOS:
    print(f"\n  Processing ratio {target_ratio}:1 (benign:malignant)")

    # Calculate target counts to achieve ~300k total rows with specified ratio
    # If ratio is R (benign:malignant), then:
    #   benign = R * malignant
    #   total = malignant + R * malignant = malignant * (R + 1)
    #   malignant = total / (R + 1)
    #   benign = total - malignant

    n_malignant_target = int(TARGET_TOTAL_ROWS / (target_ratio + 1))
    n_benign_target = TARGET_TOTAL_ROWS - n_malignant_target

    sampling_strategy = {
        1: n_malignant_target,      # oversample malignant to target
        0: n_benign_target          # undersample benign to target
    }

    # Create pipeline: SMOTE (oversample malignant) then undersample benign
    pipeline = Pipeline([
        ('smote', SMOTE(
            sampling_strategy='minority',  # oversample minority (malignant)
            random_state=RANDOM_STATE,
            k_neighbors=5  # use fewer neighbors since malignant samples are small
        )),
        ('undersample', RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=RANDOM_STATE
        ))
    ])

    # Apply resampling to training data only
    X_train_resampled, y_train_resampled = pipeline.fit_resample(X_train, y_train)

    # Reconstruct dataframe with resampled data
    df_train_resampled = X_train_resampled.copy()
    df_train_resampled['malignant'] = y_train_resampled.values

    # Shuffle and reset index
    df_train_resampled = df_train_resampled.sample(
        frac=1.0, random_state=RANDOM_STATE
    ).reset_index(drop=True)

    # Save train data
    train_filename = os.path.join(OUTPUT_DIR, f'train_ratio_{target_ratio}_1.csv')
    df_train_resampled.to_csv(train_filename, index=False)

    # Print summary
    n_malignant_resampled = y_train_resampled.sum()
    n_benign_resampled = (y_train_resampled == 0).sum()
    actual_ratio = n_benign_resampled / n_malignant_resampled if n_malignant_resampled > 0 else 0

    print(f"    ✓ Train rows   : {len(df_train_resampled):,}")
    print(f"      Malignant   : {n_malignant_resampled:,} (synthetic samples via SMOTE)")
    print(f"      Benign      : {n_benign_resampled:,} (undersampled)")
    print(f"      Actual ratio: {actual_ratio:.2f}:1")
    print(f"    ✓ Saved: {train_filename}")

    # Save summary for this ratio
    summary_filename = os.path.join(OUTPUT_DIR, f'ratio_{target_ratio}_1_summary.txt')
    with open(summary_filename, 'w') as f:
        f.write(f"Ratio {target_ratio}:1 (benign:malignant) Resampling Summary\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Data Split (from ~400k original data) - AVOIDS DATA LEAKAGE\n")
        f.write(f"  Full data            : ~400k rows\n")
        f.write(f"  Test set (20% benign): {len(df_test):,} rows (extracted FIRST)\n")
        f.write(f"  Train set            : {len(df_train):,} rows (remaining data)\n")
        f.write(f"    - All malignant samples (for training)\n")
        f.write(f"    - 80% of benign samples\n\n")
        f.write(f"Original Train Data (before resampling)\n")
        f.write(f"  Total rows   : {len(df_train):,}\n")
        f.write(f"  Malignant    : {y_train.sum():,} (all samples)\n")
        f.write(f"  Benign       : {(y_train==0).sum():,} (80% of benign)\n")
        f.write(f"  Ratio        : {(y_train==0).sum() / y_train.sum():.2f}:1\n\n")
        f.write(f"Resampled Train Data (ratio {target_ratio}:1)\n")
        f.write(f"  Target total : {TARGET_TOTAL_ROWS:,} rows\n")
        f.write(f"  Total rows   : {len(df_train_resampled):,}\n")
        f.write(f"  Malignant    : {n_malignant_resampled:,} (SMOTE synthetic oversampled)\n")
        f.write(f"  Benign       : {n_benign_resampled:,} (undersampled from {(y_train==0).sum():,})\n")
        f.write(f"  Ratio        : {actual_ratio:.2f}:1\n\n")
        f.write(f"Test Data (unchanged, shared across all ratios)\n")
        f.write(f"  Total rows   : {len(df_test):,}\n")
        f.write(f"  Malignant    : {y_test.sum():,} (none - all benign)\n")
        f.write(f"  Benign       : {(y_test==0).sum():,} (20% of original benign)\n")
        f.write(f"  Purpose      : Evaluate model specificity (no false positives)\n\n")
        f.write(f"Resampling Method\n")
        f.write(f"  Oversampling : SMOTE (synthetic minority class generation)\n")
        f.write(f"  Undersampling: RandomUnderSampler (majority class reduction)\n")
        f.write(f"  Random state : {RANDOM_STATE}\n\n")
        f.write(f"Data Leakage Prevention\n")
        f.write(f"  Test set extracted BEFORE resampling to prevent leakage.\n")
        f.write(f"  Only training data undergoes SMOTE/undersampling.\n")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("ALL DONE — Ratio-based Datasets Generated")
print("=" * 70)

print(f"\nConfiguration:")
print(f"  Original data       : ~400k rows")
print(f"  Test set (20% benign): {len(df_test):,} rows (all benign - no leakage)")
print(f"  Train set           : {len(df_train):,} rows (all malignant + 80% benign)")
print(f"  Target total rows   : {TARGET_TOTAL_ROWS:,} per ratio")
print(f"  Ratios generated    : {RATIOS}")

print(f"\nGenerated {len(RATIOS)} training datasets + 1 shared test set:")
print(f"  Shared test set: test_ratio_shared.csv ({len(df_test):,} benign samples)")
for ratio in RATIOS:
    n_mal = int(TARGET_TOTAL_ROWS / (ratio + 1))
    n_ben = TARGET_TOTAL_ROWS - n_mal
    print(f"  • {ratio}:1 ratio — train_ratio_{ratio}_1.csv ({n_ben:,} benign + {n_mal:,} malignant)")

print(f"\nAll files saved to: {OUTPUT_DIR}/")
print(f"\nTo use these datasets in your model:")
print(f"  1. Load: train_df = pd.read_csv('ratio_data/train_ratio_20_1.csv')")
print(f"  2. Test: test_df = pd.read_csv('ratio_data/test_ratio_shared.csv')")
print(f"  3. Train your model on train_df")
print(f"  4. Evaluate on test_df (same test set for all ratios)")
print(f"\nNote: All malignant samples are in training set.")
print(f"      Test set is all benign (negative samples only) to evaluate")
print(f"      the model's specificity. Malignant samples in training are")
print(f"      synthetically generated via SMOTE to achieve desired ratios.\n")

print("=" * 70)
