# =============================================================================
# ISIC 2024 — Data Cleaning, Feature Engineering & Dataset Splitting
# =============================================================================
# Output files:
#   merged_cleaned.csv        — full cleaned dataset (all records)
#   merged_cleaned_binned.csv — same + Low/Medium/High bin columns
#   datasetv1.csv             — malignant records only (393 rows)
#   datasetv2.csv             — benign records only (all 400,666)
#   datasetv3.csv             — full dataset with risk scores
#   feature_ranges.csv        — benign vs malignant stats per feature
#   train_final.csv           — training set (all 393 malignant + 80% benign)
#   test_final.csv            — test set (20% benign — NEVER seen by model)
#   patient_ids_train.csv     — patient IDs for training rows (for GroupKFold)
#
# Install once:
#   !pip install pandas numpy scikit-learn matplotlib seaborn scipy lightgbm
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

_HERE      = Path(__file__).resolve().parent
_DATASETS  = (_HERE / '../../datasets').resolve()
_PROCESSED = _HERE / 'processed_data'
_PROCESSED.mkdir(exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

FILE_FEATURES = str(_DATASETS / 'metadata.csv')
FILE_LABELS   = str(_DATASETS / 'labels.csv')

ANALYSIS_COLUMNS = [
    'age_approx',
    'sex',
    'anatom_site_general',
    'tbp_lv_location_simple',
    'clin_size_long_diam_mm',
    'tbp_lv_area_perim_ratio',
    'tbp_lv_norm_border',
    'tbp_lv_symm_2axis',
    'tbp_lv_eccentricity',
    'tbp_lv_color_std_mean',
    'tbp_lv_norm_color',
    'tbp_lv_deltaLBnorm',
    'tbp_lv_radial_color_std_max',
    'tbp_lv_nevi_confidence',
    'color_contrast_3d',
    'elongation',
    'nevi_color_tension',
    'log_area',
    'stdL_ratio',
    'compactness',
    'chroma_contrast',
    'radial_color_ratio',
    'malignant',
]

CONTINUOUS_COLS = [
    'age_approx',
    'clin_size_long_diam_mm',
    'tbp_lv_area_perim_ratio',
    'tbp_lv_norm_border',
    'tbp_lv_symm_2axis',
    'tbp_lv_eccentricity',
    'tbp_lv_color_std_mean',
    'tbp_lv_norm_color',
    'tbp_lv_deltaLBnorm',
    'tbp_lv_radial_color_std_max',
    'tbp_lv_nevi_confidence',
    'color_contrast_3d',
    'elongation',
    'nevi_color_tension',
    'log_area',
    'stdL_ratio',
    'compactness',
    'chroma_contrast',
    'radial_color_ratio',
]

CATEGORICAL_COLS = [
    'sex',
    'anatom_site_general',
    'tbp_lv_location_simple',
]

# =============================================================================
# RANGE CONFIGURATION — adjust bin edges here to experiment
# =============================================================================
# Each feature has a list of cut points that define Low / Medium / High bins.
# Change these numbers to try different range sizes and compare results.
#
# Format: [lower_bound, cut1, cut2, upper_bound]
#   Values below cut1         = Low
#   Values between cut1, cut2 = Medium
#   Values above cut2         = High
#
# VERSION LABEL — change this when you change ranges so output files
# are named differently and you can compare versions easily.

RANGE_VERSION = 'v1'   # change to v2, v3 etc. when you adjust ranges

CUSTOM_BINS = {
    # ABCDE "D" — 6mm is the clinical threshold
    'age_approx':               [0,    40,   65,   85],
    'clin_size_long_diam_mm':   [0,     6,   15,  150],

    # Shape (ABCDE A and B)
    'tbp_lv_area_perim_ratio':  [0,    12,   25,  100],
    'tbp_lv_norm_border':       [0,   2.5,    5,   10],
    'tbp_lv_symm_2axis':        [0,   0.2,  0.5,    1],
    'tbp_lv_eccentricity':      [0,   0.4,  0.7,    1],

    # Color (ABCDE C)
    'tbp_lv_color_std_mean':    [0,     1,    3,   10],
    'tbp_lv_norm_color':        [0,   2.5,    5,   10],
    'tbp_lv_deltaLBnorm':       [0,     5,   12,   30],
    'tbp_lv_radial_color_std_max': [0,  1,    4,   15],

    # AI score
    'tbp_lv_nevi_confidence':   [0,    33,   67,  100],

    # Engineered
    'color_contrast_3d':        [0,     5,   12,   50],
    'elongation':               [0,   1.2,    2,   10],
    'nevi_color_tension':       [-50,   0,   30,  100],
    'log_area':                 [0,     2,    4,   10],
    'stdL_ratio':               [0,   0.5,  1.5,   10],
    'compactness':              [0,   0.5,  0.8,    1],
    'chroma_contrast':          [-20,   0,    5,   30],
    'radial_color_ratio':       [0,   0.5,  1.5,   20],
}

TEST_SIZE    = 0.20       # 20% of benign records held out for testing
RANDOM_STATE = 42


# =============================================================================
# STEP 1 — Load both files
# =============================================================================

print("=" * 60)
print("STEP 1 — Loading files")
print("=" * 60)

df_features = pd.read_csv(FILE_FEATURES, dtype={
    'isic_id':                    'str',
    'patient_id':                 'str',
    'age_approx':                 'float32',
    'sex':                        'str',
    'anatom_site_general':        'str',
    'clin_size_long_diam_mm':     'float32',
    'image_type':                 'str',
    'tbp_tile_type':              'str',
    'tbp_lv_A':                   'float32',
    'tbp_lv_Aext':                'float32',
    'tbp_lv_B':                   'float32',
    'tbp_lv_Bext':                'float32',
    'tbp_lv_C':                   'float32',
    'tbp_lv_Cext':                'float32',
    'tbp_lv_H':                   'float32',
    'tbp_lv_Hext':                'float32',
    'tbp_lv_L':                   'float32',
    'tbp_lv_Lext':                'float32',
    'tbp_lv_areaMM2':             'float32',
    'tbp_lv_area_perim_ratio':    'float32',
    'tbp_lv_color_std_mean':      'float32',
    'tbp_lv_deltaA':              'float32',
    'tbp_lv_deltaB':              'float32',
    'tbp_lv_deltaL':              'float32',
    'tbp_lv_deltaLB':             'float32',
    'tbp_lv_deltaLBnorm':         'float32',
    'tbp_lv_eccentricity':        'float32',
    'tbp_lv_location':            'str',
    'tbp_lv_location_simple':     'str',
    'tbp_lv_minorAxisMM':         'float32',
    'tbp_lv_nevi_confidence':     'float32',
    'tbp_lv_norm_border':         'float32',
    'tbp_lv_norm_color':          'float32',
    'tbp_lv_perimeterMM':         'float32',
    'tbp_lv_radial_color_std_max':'float32',
    'tbp_lv_stdL':                'float32',
    'tbp_lv_stdLExt':             'float32',
    'tbp_lv_symm_2axis':          'float32',
    'tbp_lv_symm_2axis_angle':    'float32',
    'tbp_lv_x':                   'float32',
    'tbp_lv_y':                   'float32',
    'tbp_lv_z':                   'float32',
})

df_labels = pd.read_csv(FILE_LABELS, dtype={
    'isic_id':   'str',
    'malignant': 'int8',
})

print(f"  Features file : {df_features.shape[0]:,} rows x {df_features.shape[1]} columns")
print(f"  Labels file   : {df_labels.shape[0]:,} rows x {df_labels.shape[1]} columns")
print(f"  Malignant     : {df_labels['malignant'].sum():,}")
print(f"  Benign        : {(df_labels['malignant']==0).sum():,}")


# =============================================================================
# STEP 2 — Merge on isic_id
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2 — Merging on isic_id")
print("=" * 60)

df = pd.merge(df_features, df_labels, on='isic_id', how='inner')

lost = len(df_features) - len(df)
print(f"  Merged shape  : {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"  Rows lost     : {lost:,}")
print(f"  Malignant     : {df['malignant'].sum():,}")
print(f"  Benign        : {(df['malignant']==0).sum():,}")

# Save patient_id before any drops
patient_ids = df[['isic_id', 'patient_id']].copy()


# =============================================================================
# STEP 3 — Handle missing values (on full column set before engineering)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3 — Handling missing values")
print("=" * 60)

all_cat = ['sex', 'anatom_site_general', 'tbp_lv_location',
           'tbp_lv_location_simple', 'image_type', 'tbp_tile_type']
all_num = [c for c in df.columns
           if c not in all_cat + ['isic_id', 'patient_id', 'malignant']
           and df[c].dtype != 'object']

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing) == 0:
    print("  No missing values found.")
else:
    print(f"  Found missing values in {len(missing)} columns:")
    for col, count in missing.items():
        pct   = count / len(df) * 100
        dtype = 'categorical' if col in all_cat else 'numeric'
        print(f"    [{dtype:11s}] {col:40s}: {count:6,} ({pct:.1f}%)")

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

num_cols_present = [c for c in all_num if c in df.columns]
cat_cols_present = [c for c in all_cat if c in df.columns]

df[num_cols_present] = num_imputer.fit_transform(df[num_cols_present])
df[cat_cols_present] = cat_imputer.fit_transform(df[cat_cols_present])

print(f"  Missing after imputation: {df.isnull().sum().sum()}")


# =============================================================================
# STEP 4 — Numeric range sanity check
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4 — Numeric range check and clipping")
print("=" * 60)

RANGES = {
    'age_approx':               (0,   85),
    'clin_size_long_diam_mm':   (0,  150),
    'tbp_lv_nevi_confidence':   (0,  100),
    'tbp_lv_eccentricity':      (0,    1),
    'tbp_lv_symm_2axis':        (0,    1),
    'tbp_lv_norm_border':       (0,   10),
    'tbp_lv_norm_color':        (0,   10),
    'tbp_lv_area_perim_ratio':  (0,  100),
    'tbp_lv_areaMM2':           (0, 5000),
    'tbp_lv_perimeterMM':       (0,  500),
    'tbp_lv_minorAxisMM':       (0,  100),
}

for col, (lo, hi) in RANGES.items():
    if col not in df.columns:
        continue
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    if n_out > 0:
        print(f"  WARNING {col}: {n_out:,} out of range — clipping to [{lo}, {hi}]")
        df[col] = df[col].clip(lo, hi)
    else:
        print(f"  OK      {col}")


# =============================================================================
# STEP 5 — Feature engineering (BEFORE dropping source columns)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5 — Feature engineering")
print("=" * 60)

def add_feature(df, name, formula, sources):
    missing_src = [c for c in sources if c not in df.columns]
    if missing_src:
        print(f"  SKIP  {name:30s} missing source: {missing_src}")
        return df
    if name in df.columns:
        print(f"  EXISTS {name:29s} already present — skipping")
        return df
    df[name] = formula(df)
    print(f"  ADDED {name}")
    return df

df = add_feature(df, 'color_contrast_3d',
    lambda d: np.sqrt(d['tbp_lv_deltaA']**2 +
                      d['tbp_lv_deltaB']**2 +
                      d['tbp_lv_deltaL']**2),
    ['tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL'])

df = add_feature(df, 'elongation',
    lambda d: d['clin_size_long_diam_mm'] / (d['tbp_lv_minorAxisMM'] + 1e-6),
    ['clin_size_long_diam_mm', 'tbp_lv_minorAxisMM'])

df = add_feature(df, 'nevi_color_tension',
    lambda d: d['tbp_lv_nevi_confidence'] - d['tbp_lv_color_std_mean'],
    ['tbp_lv_nevi_confidence', 'tbp_lv_color_std_mean'])

df = add_feature(df, 'log_area',
    lambda d: np.log1p(d['tbp_lv_areaMM2']),
    ['tbp_lv_areaMM2'])

df = add_feature(df, 'stdL_ratio',
    lambda d: d['tbp_lv_stdL'] / (d['tbp_lv_stdLExt'] + 1e-6),
    ['tbp_lv_stdL', 'tbp_lv_stdLExt'])

df = add_feature(df, 'compactness',
    lambda d: 4 * np.pi * d['tbp_lv_areaMM2'] /
              (d['tbp_lv_perimeterMM']**2 + 1e-6),
    ['tbp_lv_areaMM2', 'tbp_lv_perimeterMM'])

df = add_feature(df, 'chroma_contrast',
    lambda d: d['tbp_lv_C'] - d['tbp_lv_Cext'],
    ['tbp_lv_C', 'tbp_lv_Cext'])

df = add_feature(df, 'radial_color_ratio',
    lambda d: d['tbp_lv_radial_color_std_max'] /
              (d['tbp_lv_color_std_mean'] + 1e-6),
    ['tbp_lv_radial_color_std_max', 'tbp_lv_color_std_mean'])


# =============================================================================
# STEP 6 — Drop source and unused columns
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6 — Dropping source and unused columns")
print("=" * 60)

COLS_TO_DROP = [
    'isic_id', 'patient_id',
    'image_type', 'tbp_tile_type',
    'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB',
    'tbp_lv_minorAxisMM', 'tbp_lv_areaMM2', 'tbp_lv_perimeterMM',
    'tbp_lv_stdL', 'tbp_lv_stdLExt',
    'tbp_lv_C', 'tbp_lv_Cext',
    'tbp_lv_A', 'tbp_lv_Aext',
    'tbp_lv_B', 'tbp_lv_Bext',
    'tbp_lv_H', 'tbp_lv_Hext',
    'tbp_lv_L', 'tbp_lv_Lext',
    'tbp_lv_location',
    'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z',
    'tbp_lv_symm_2axis_angle',
]

COLS_TO_DROP = [c for c in COLS_TO_DROP if c in df.columns]
df = df.drop(columns=COLS_TO_DROP)
print(f"  Dropped {len(COLS_TO_DROP)} columns")


# =============================================================================
# STEP 7 — Keep only final analysis columns
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7 — Selecting final analysis columns")
print("=" * 60)

FINAL_COLS = [c for c in ANALYSIS_COLUMNS if c in df.columns]
missing_cols = [c for c in ANALYSIS_COLUMNS if c not in df.columns]

if missing_cols:
    print(f"  WARNING — missing expected columns: {missing_cols}")
else:
    print(f"  All {len(FINAL_COLS)} expected columns present.")

df = df[FINAL_COLS]
CONTINUOUS_COLS = [c for c in CONTINUOUS_COLS if c in df.columns]

print(f"\n  Final columns ({len(FINAL_COLS)}):")
print(f"  {'Column':<35} {'Type'}")
print(f"  {'-'*50}")
for col in FINAL_COLS:
    dtype = 'CATEGORICAL' if col in CATEGORICAL_COLS else \
            'TARGET'      if col == 'malignant'       else \
            'CONTINUOUS'
    print(f"  {col:<35} {dtype}")

print(f"\n  Total rows    : {len(df):,}")
print(f"  Total columns : {len(df.columns)}")


# =============================================================================
# STEP 8 — Save merged_cleaned.csv (full dataset)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8 — Saving merged_cleaned.csv")
print("=" * 60)

df.to_csv(str(_DATASETS / 'merged_cleaned.csv'), index=False)
print(f"  Saved: datasets/merged_cleaned.csv  ({df.shape[0]:,} rows x {df.shape[1]} columns)")


# =============================================================================
# STEP 9 — Split into malignant-only and benign-only datasets
# =============================================================================

print("\n" + "=" * 60)
print("STEP 9 — Splitting into datasetv1 (malignant) and datasetv2 (benign)")
print("=" * 60)

df_malignant = df[df['malignant'] == 1].copy().reset_index(drop=True)
df_benign    = df[df['malignant'] == 0].copy().reset_index(drop=True)

df_malignant.to_csv(str(_DATASETS / 'datasetv1_all_malignant.csv'), index=False)
df_benign.to_csv(str(_DATASETS / 'datasetv2_all_benign.csv'),       index=False)

print(f"  datasetv1_all_malignant.csv — Malignant only : {len(df_malignant):,} rows")
print(f"  datasetv2_all_benign.csv    — Benign only    : {len(df_benign):,} rows")
assert len(df_malignant) + len(df_benign) == len(df)
print(f"  Confirmed: {len(df_malignant):,} + {len(df_benign):,} = {len(df):,} ✓")


# =============================================================================
# STEP 9b — Split benign into TRAIN (80%) and TEST (20%)
# =============================================================================
# KEY RULE:
#   - All 393 malignant records → training only (reference profile)
#   - 80% of benign records    → training
#   - 20% of benign records    → testing (NEVER seen during model training)
#
# This guarantees the test benign records are truly unseen data.

print("\n" + "=" * 60)
print("STEP 9b — Train/test split (benign records only)")
print("=" * 60)

# Also get corresponding patient_ids for the benign rows
# to use in GroupKFold during training
df_benign_with_pid = df_benign.copy()

# Get patient_ids for benign rows from our saved mapping
benign_isic_ids    = patient_ids[
    patient_ids['isic_id'].isin(
        df_features[df_features.index.isin(df_benign.index)]['isic_id']
        if 'isic_id' in df_features.columns else []
    )
]

# Split benign 80/20
df_benign_train, df_benign_test = train_test_split(
    df_benign,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True,
)

# Training set = all 393 malignant + 80% of benign
df_train = pd.concat(
    [df_malignant, df_benign_train],
    ignore_index=True
)

# Test set = 20% of benign ONLY — model never sees these
df_test = df_benign_test.copy().reset_index(drop=True)

# Save
df_train.to_csv(str(_PROCESSED / 'train_final.csv'), index=False)
df_test.to_csv(str(_PROCESSED / 'test_final.csv'),   index=False)

# Verify zero overlap between train and test benign rows
train_benign_idx = set(df_benign_train.index)
test_benign_idx  = set(df_benign_test.index)
overlap          = train_benign_idx & test_benign_idx

print(f"\n  SPLIT SUMMARY")
print(f"  {'─'*45}")
print(f"  train_final.csv")
print(f"    Total rows         : {len(df_train):,}")
print(f"    Malignant (all)    : {df_train['malignant'].sum():,}  ← all 393 kept")
print(f"    Benign (80%)       : {(df_train['malignant']==0).sum():,}")
print(f"")
print(f"  test_final.csv")
print(f"    Total rows         : {len(df_test):,}")
print(f"    Benign only (20%)  : {len(df_test):,}  ← NEVER seen by model")
print(f"    Malignant rows     : 0  ← none, by design")
print(f"")
print(f"  Overlap check        : {'0 rows — CLEAN ✓' if len(overlap)==0 else f'ERROR: {len(overlap)} overlapping rows'}")
print(f"  Total accounted for  : {len(df_train) - df_train['malignant'].sum() + len(df_test):,} benign rows")
print(f"  Original benign      : {len(df_benign):,}")


# =============================================================================
# STEP 10 — Compute feature ranges (benign vs malignant)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 10 — Feature ranges: Benign vs Malignant")
print("=" * 60)

range_rows = []
print(f"\n  {'Feature':<35} {'Ben Mean':>10} {'Mal Mean':>10} {'Ben Median':>12} {'Mal Median':>12}")
print(f"  {'-'*80}")

for col in CONTINUOUS_COLS:
    b_mean   = df_benign[col].mean()
    m_mean   = df_malignant[col].mean()
    b_median = df_benign[col].median()
    m_median = df_malignant[col].median()

    print(f"  {col:<35} {b_mean:>10.3f} {m_mean:>10.3f} {b_median:>12.3f} {m_median:>12.3f}")

    range_rows.append({
        'feature':           col,
        'benign_mean':       round(float(b_mean),            4),
        'benign_median':     round(float(b_median),          4),
        'benign_min':        round(float(df_benign[col].min()), 4),
        'benign_max':        round(float(df_benign[col].max()), 4),
        'benign_std':        round(float(df_benign[col].std()), 4),
        'malignant_mean':    round(float(m_mean),            4),
        'malignant_median':  round(float(m_median),          4),
        'malignant_min':     round(float(df_malignant[col].min()), 4),
        'malignant_max':     round(float(df_malignant[col].max()), 4),
        'malignant_std':     round(float(df_malignant[col].std()), 4),
    })

pd.DataFrame(range_rows).to_csv(str(_DATASETS / 'feature_ranges.csv'), index=False)
print(f"\n  Saved: datasets/feature_ranges.csv")


# =============================================================================
# STEP 11 — Bin continuous features using CUSTOM ranges
# =============================================================================

print("\n" + "=" * 60)
print(f"STEP 11 — Binning features (Range version: {RANGE_VERSION})")
print("=" * 60)

df_binned  = df.copy()
bin_report = []

for col in CONTINUOUS_COLS:
    if col not in df_binned.columns:
        continue
    try:
        if col in CUSTOM_BINS:
            edges  = CUSTOM_BINS[col]
            labels = ['Low', 'Medium', 'High']

            # include_lowest=True ensures the minimum value is included
            binned = pd.cut(
                df_binned[col],
                bins=edges,
                labels=labels,
                include_lowest=True,
            )
            df_binned[col + '_bin'] = binned

            print(f"  {col:<35} "
                  f"Low <{edges[1]}  |  "
                  f"Med {edges[1]}–{edges[2]}  |  "
                  f"High >{edges[2]}")

        else:
            # fallback: auto equal-width if not in CUSTOM_BINS
            binned, edges_auto = pd.cut(
                df_binned[col], bins=3,
                labels=['Low', 'Medium', 'High'],
                retbins=True
            )
            df_binned[col + '_bin'] = binned
            print(f"  {col:<35} AUTO bins (not in CUSTOM_BINS)")

        bin_report.append({
            'feature':        col,
            'range_version':  RANGE_VERSION,
            'low_max':        edges[1] if col in CUSTOM_BINS else None,
            'medium_max':     edges[2] if col in CUSTOM_BINS else None,
        })

    except Exception as e:
        print(f"  ERROR {col}: {e}")

# Include version label in the output filename
output_name = str(_DATASETS / f'merged_cleaned_binned_{RANGE_VERSION}.csv')
df_binned.to_csv(output_name, index=False)
print(f"\n  Saved: datasets/merged_cleaned_binned_{RANGE_VERSION}.csv")

# =============================================================================
# STEP 12 — Risk score for benign records (using full benign set)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 12 — Computing malignancy risk score for benign records")
print("=" * 60)

risk_score      = pd.Series(0.0, index=df_benign.index)
features_scored = []

for col in CONTINUOUS_COLS:
    mal_mean = df_malignant[col].mean()
    mal_std  = df_malignant[col].std()
    lo = mal_mean - mal_std
    hi = mal_mean + mal_std

    in_range = ((df_benign[col] >= lo) & (df_benign[col] <= hi)).astype(float)
    risk_score += in_range
    features_scored.append(col)

risk_score_norm  = risk_score / len(features_scored)

df_benign_scored = df_benign.copy()
df_benign_scored['risk_score'] = risk_score_norm.values

print(f"  Features scored : {len(features_scored)}")
print(f"  Score range     : {risk_score_norm.min():.3f} – {risk_score_norm.max():.3f}")
print(f"  Mean score      : {risk_score_norm.mean():.3f}")


# =============================================================================
# STEP 13 — Save datasetv3.csv
# =============================================================================

print("\n" + "=" * 60)
print("STEP 13 — Saving datasetv3.csv")
print("=" * 60)

df_v3 = df.copy()
df_v3['risk_score'] = np.nan
df_v3.loc[df_benign_scored.index, 'risk_score'] = df_benign_scored['risk_score'].values

df_v3.to_csv(str(_DATASETS / 'datasetv3.csv'), index=False)
print(f"  Saved: datasets/datasetv3.csv  ({df_v3.shape[0]:,} rows x {df_v3.shape[1]} columns)")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("ALL DONE — Output Files")
print("=" * 60)
print(f"""
  FOR ANALYSIS & GRAPHING:
    merged_cleaned.csv        — {df.shape[0]:,} rows, {df.shape[1]} cols (full cleaned)
    merged_cleaned_binned.csv — same + Low/Medium/High bin columns
    datasetv1.csv             — {len(df_malignant):,} rows (malignant only)
    datasetv2.csv             — {len(df_benign):,} rows (benign only, full)
    datasetv3.csv             — {df_v3.shape[0]:,} rows (full + risk scores)
    feature_ranges.csv        — feature profile comparison

  FOR MODEL TRAINING & TESTING:
    train_final.csv           — {len(df_train):,} rows
                                 ({df_train['malignant'].sum()} malignant + {(df_train['malignant']==0).sum():,} benign)
    test_final.csv            — {len(df_test):,} rows
                                 (benign only — NEVER seen during training)

  NEXT STEPS:
    1. Run isic_graphs.py    → uses merged_cleaned.csv
    2. Run isic_analysis.py  → uses merged_cleaned.csv
    3. Run isic_model.py     → trains on train_final.csv
                               tests on  test_final.csv
""")
print("=" * 60)
