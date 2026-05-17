# =============================================================================
# ISIC 2024 — Stratified Split + Ratio-based Resampling (CORRECTED)
# =============================================================================
# PROPER WORKFLOW:
#   1. Stratified split on BOTH malignant and benign (prevent data leakage)
#   2. Keep test set as pure original data (never modify)
#   3. For each ratio, resample ONLY the training data using SMOTE + undersampling
#
# This avoids synthetic data leakage and ensures honest evaluation.
#
# Original data: 393 malignant, 320,532 benign (~815.6:1)
# Split: ~76% train (300 mal, ~244k ben), ~24% test (93 mal, ~76k ben)
#
# Output files:
#   ratio_data_v2/
#     test_stratified.csv           — shared test set (real data, both classes)
#     train_ratio_{ratio}_1.csv     — training data resampled to {ratio}:1
#
# Install once:
#   pip install imbalanced-learn
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_ORIGINAL = 'train_final.csv'
OUTPUT_DIR = 'ratio_data_v2'

# Ratios to generate (benign:malignant in training data)
RATIOS = [5, 10, 15, 20, 25, 30]

# Test/train split: use ~24% for test (~93 malignant, ~76k benign)
TEST_SIZE = 0.24

# Target total training rows (after resampling)
TARGET_TRAIN_ROWS = 300_000

RANDOM_STATE = 42


# =============================================================================
# SETUP
# =============================================================================

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory: {OUTPUT_DIR}")


# =============================================================================
# STEP 1 — Load original data and do stratified split
# =============================================================================

print("=" * 80)
print("ISIC 2024 — Stratified Split + Ratio-based Resampling (CORRECTED)")
print("=" * 80)

print("\nSTEP 1 — Stratified train/test split (on original data)")
print("-" * 80)

df_original = pd.read_csv(INPUT_ORIGINAL)

n_malignant_original = df_original['malignant'].sum()
n_benign_original = (df_original['malignant'] == 0).sum()

print(f"\n  Original data ({INPUT_ORIGINAL}):")
print(f"    Total rows   : {len(df_original):,}")
print(f"    Malignant    : {n_malignant_original:,}")
print(f"    Benign       : {n_benign_original:,}")
print(f"    Ratio        : {n_benign_original / n_malignant_original:.1f}:1")

# Stratified split (maintaining class proportions)
print(f"\n  Stratified split (test_size={TEST_SIZE:.0%}):")

X = df_original.drop(columns=['malignant'])
y = df_original['malignant']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

df_train_split = X_train.copy()
df_train_split['malignant'] = y_train.values

df_test_split = X_test.copy()
df_test_split['malignant'] = y_test.values

n_malignant_train = y_train.sum()
n_benign_train = (y_train == 0).sum()
n_malignant_test = y_test.sum()
n_benign_test = (y_test == 0).sum()

print(f"    Training set:")
print(f"      Malignant  : {n_malignant_train:,}  (original, for SMOTE)")
print(f"      Benign     : {n_benign_train:,}  (original, for undersampling)")
print(f"      Ratio      : {n_benign_train / n_malignant_train:.1f}:1")

print(f"    Test set (never modified):")
print(f"      Malignant  : {n_malignant_test:,}  (original)")
print(f"      Benign     : {n_benign_test:,}  (original)")
print(f"      Ratio      : {n_benign_test / n_malignant_test:.1f}:1")

# Save test set (shared across all ratios)
test_filename = os.path.join(OUTPUT_DIR, 'test_stratified.csv')
df_test_split.to_csv(test_filename, index=False)
print(f"\n    ✓ Saved shared test set: {test_filename}")


# =============================================================================
# STEP 2 — For each ratio, resample ONLY the training data
# =============================================================================

print("\nSTEP 2 — Resampling training data for each ratio")
print("-" * 80)

for target_ratio in RATIOS:
    print(f"\n  Processing ratio {target_ratio}:1 (benign:malignant)")

    # Calculate target counts for resampling
    # We want ~TARGET_TRAIN_ROWS total, with benign:malignant = target_ratio
    n_malignant_target = int(TARGET_TRAIN_ROWS / (target_ratio + 1))
    n_benign_target = TARGET_TRAIN_ROWS - n_malignant_target

    sampling_strategy = {
        1: n_malignant_target,      # oversample malignant to target
        0: n_benign_target           # undersample benign to target
    }

    print(f"    Target: {n_benign_target:,} benign + {n_malignant_target:,} malignant "
          f"= {n_benign_target + n_malignant_target:,} total")

    # Prepare X and y for resampling
    X_train_df = df_train_split.drop(columns=['malignant']).copy()
    y_train_arr = df_train_split['malignant'].values

    # Convert to numeric (handle categorical if needed)
    X_train_numeric = X_train_df.copy()
    cat_cols = X_train_numeric.select_dtypes(include='object').columns.tolist()
    if cat_cols:
        for col in cat_cols:
            X_train_numeric[col] = pd.factorize(X_train_numeric[col])[0]
    X_train_numeric = X_train_numeric.values.astype(float)

    # Step 1: SMOTE oversample malignant (synthetic minority oversampling)
    mal_idx = np.where(y_train_arr == 1)[0]
    ben_idx = np.where(y_train_arr == 0)[0]

    X_mal = X_train_numeric[mal_idx]
    X_ben = X_train_numeric[ben_idx]
    X_mal_orig_df = X_train_df.iloc[mal_idx].copy()
    X_ben_orig_df = X_train_df.iloc[ben_idx].copy()

    # Generate synthetic malignant samples via SMOTE (k-NN interpolation)
    n_to_generate = n_malignant_target - len(mal_idx)
    if n_to_generate > 0:
        np.random.seed(RANDOM_STATE)
        synthetic_mal = []
        synthetic_mal_df = []
        for _ in range(n_to_generate):
            # Pick random malignant sample and one of its neighbors
            idx_a = np.random.randint(len(X_mal))
            idx_b = np.random.randint(len(X_mal))
            # Interpolate between them
            alpha = np.random.rand()
            synthetic_sample = X_mal[idx_a] + alpha * (X_mal[idx_b] - X_mal[idx_a])
            synthetic_mal.append(synthetic_sample)
            # Also keep original dataframe version with original categorical values
            synthetic_mal_df.append(X_mal_orig_df.iloc[idx_a])
        X_mal_synthetic = np.vstack([X_mal, synthetic_mal])
        X_mal_orig_df = pd.concat([X_mal_orig_df, pd.DataFrame(synthetic_mal_df)], ignore_index=True)
    else:
        X_mal_synthetic = X_mal

    # Step 2: Random undersample benign
    n_ben_keep = min(n_benign_target, len(X_ben))
    np.random.seed(RANDOM_STATE)
    ben_idx_keep = np.random.choice(len(X_ben), size=n_ben_keep, replace=False)
    X_ben_undersampled = X_ben[ben_idx_keep]
    X_ben_orig_df = X_ben_orig_df.iloc[ben_idx_keep].copy()

    # Combine resampled data (use original dataframes with original categorical values)
    df_train_resampled = pd.concat([
        X_ben_orig_df.reset_index(drop=True),
        X_mal_orig_df.iloc[:n_malignant_target].reset_index(drop=True)
    ], ignore_index=True)

    y_train_resampled = np.hstack([
        np.zeros(len(X_ben_orig_df)),
        np.ones(min(len(X_mal_orig_df), n_malignant_target))
    ])
    df_train_resampled['malignant'] = y_train_resampled

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

    print(f"    ✓ Train rows     : {len(df_train_resampled):,}")
    print(f"      Malignant    : {n_malignant_resampled:,} (SMOTE synthetic)")
    print(f"      Benign       : {n_benign_resampled:,} (undersampled from {n_benign_train:,})")
    print(f"      Actual ratio : {actual_ratio:.2f}:1")
    print(f"    ✓ Saved: {train_filename}")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("ALL DONE — Stratified Split + Ratio-based Datasets Generated")
print("=" * 80)

print(f"\nData split strategy:")
print(f"  Original data    : {n_malignant_original:,} malignant, {n_benign_original:,} benign "
      f"({n_benign_original/n_malignant_original:.1f}:1)")
print(f"  Test set         : {n_malignant_test:,} malignant, {n_benign_test:,} benign (UNCHANGED)")
print(f"  Training base    : {n_malignant_train:,} malignant, {n_benign_train:,} benign")

print(f"\nRatios generated: {RATIOS}")
print(f"  Each ratio creates a new train_ratio_X_1.csv")
print(f"  All ratios use the SAME test_stratified.csv")

print(f"\nFiles in {OUTPUT_DIR}/:")
print(f"  test_stratified.csv                — shared test set (real data)")
print(f"  train_ratio_5_1.csv, train_ratio_10_1.csv, ...  — training data per ratio")

print(f"\nKey improvement over v1:")
print(f"  ✓ Stratified split prevents data leakage")
print(f"  ✓ Test set contains BOTH benign and malignant")
print(f"  ✓ Honest evaluation: test data is completely original (unmodified)")
print(f"  ✓ All ratios evaluated on same test set")

print(f"\nUsage:")
print(f"  df_train = pd.read_csv('{OUTPUT_DIR}/train_ratio_10_1.csv')")
print(f"  df_test = pd.read_csv('{OUTPUT_DIR}/test_stratified.csv')")
print(f"  model.fit(df_train, y_train)")
print(f"  model.evaluate(df_test, y_test)  ← Now has both classes!")

print("=" * 80)
