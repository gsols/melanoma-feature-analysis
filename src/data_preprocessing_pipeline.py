"""
Regenerate dataset versions from raw_metadata.csv and labels.csv.

Outputs (all written to datasets/):
  datasetv3_raw_metadata_labels_merged.csv  — raw merge on isic_id (all original columns)
  datasetv4_merged_cleaned.csv              — cleaned + engineered features, final columns only
  datasetv5_cleaned_all_malignant.csv       — malignant rows only
  datasetv6_cleaned_all_benign.csv          — benign rows only
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

_HERE    = Path(__file__).resolve().parent
DATASETS = _HERE / "datasets"

RAW_METADATA = DATASETS / "datasetv1_raw_metadata.csv"
LABELS       = DATASETS / "datasetv2_labels.csv"

FINAL_COLS = [
    "age_approx",
    "sex",
    "anatom_site_general",
    "tbp_lv_location_simple",
    "clin_size_long_diam_mm",
    "tbp_lv_area_perim_ratio",
    "tbp_lv_norm_border",
    "tbp_lv_symm_2axis",
    "tbp_lv_eccentricity",
    "tbp_lv_color_std_mean",
    "tbp_lv_norm_color",
    "tbp_lv_deltaLBnorm",
    "tbp_lv_radial_color_std_max",
    "tbp_lv_nevi_confidence",
    "color_contrast_3d",
    "elongation",
    "nevi_color_tension",
    "log_area",
    "stdL_ratio",
    "compactness",
    "chroma_contrast",
    "radial_color_ratio",
    "malignant",
]

CATEGORICAL_COLS = ["sex", "anatom_site_general", "tbp_lv_location_simple"]

CLIP_RANGES = {
    "age_approx":              (0,   85),
    "clin_size_long_diam_mm":  (0,  150),
    "tbp_lv_nevi_confidence":  (0,  100),
    "tbp_lv_eccentricity":     (0,    1),
    "tbp_lv_symm_2axis":       (0,    1),
    "tbp_lv_norm_border":      (0,   10),
    "tbp_lv_norm_color":       (0,   10),
    "tbp_lv_area_perim_ratio": (0,  100),
    "tbp_lv_areaMM2":          (0, 5000),
    "tbp_lv_perimeterMM":      (0,  500),
    "tbp_lv_minorAxisMM":      (0,  100),
}


def _sep(title):
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


# =============================================================================
# STEP 1 — Merge raw metadata with labels on isic_id (keep all columns)
# =============================================================================

_sep("STEP 1 — Merging datasetv1_raw_metadata with datasetv2_labels")

df_meta   = pd.read_csv(RAW_METADATA)
df_labels = pd.read_csv(LABELS, dtype={"malignant": "float32"})

print(f"  Raw metadata : {df_meta.shape[0]:,} rows x {df_meta.shape[1]} cols")
print(f"  Labels       : {df_labels.shape[0]:,} rows x {df_labels.shape[1]} cols")

df_merged = pd.merge(df_meta, df_labels, on="isic_id", how="inner")
print(f"  Merged       : {df_merged.shape[0]:,} rows x {df_merged.shape[1]} cols")
print(f"  Lost rows    : {len(df_meta) - len(df_merged):,}")
print(f"  Malignant    : {(df_merged['malignant'] == 1).sum():,}")
print(f"  Benign       : {(df_merged['malignant'] == 0).sum():,}")

df_merged.to_csv(DATASETS / "datasetv3_raw_metadata_labels_merged.csv", index=False)
print(f"  Saved: datasetv3_raw_metadata_labels_merged.csv")


# =============================================================================
# STEP 2 — Impute missing values
# =============================================================================

_sep("STEP 2 — Imputing missing values")

cat_src = ["sex", "anatom_site_general", "tbp_lv_location",
           "tbp_lv_location_simple", "image_type", "tbp_tile_type"]
num_src = [
    c for c in df_merged.columns
    if c not in cat_src + ["isic_id", "patient_id", "malignant"]
    and df_merged[c].dtype != object
]

missing = df_merged.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if missing.empty:
    print("  No missing values found.")
else:
    print(f"  Columns with missing values: {len(missing)}")
    for col, n in missing.items():
        print(f"    {col}: {n:,} ({n / len(df_merged) * 100:.1f}%)")

df_merged[[c for c in num_src if c in df_merged.columns]] = (
    SimpleImputer(strategy="median")
    .fit_transform(df_merged[[c for c in num_src if c in df_merged.columns]])
)
df_merged[[c for c in cat_src if c in df_merged.columns]] = (
    SimpleImputer(strategy="most_frequent")
    .fit_transform(df_merged[[c for c in cat_src if c in df_merged.columns]])
)
print(f"  Remaining missing after imputation: {df_merged.isnull().sum().sum()}")


# =============================================================================
# STEP 3 — Clip numeric outliers
# =============================================================================

_sep("STEP 3 — Clipping numeric outliers")

for col, (lo, hi) in CLIP_RANGES.items():
    if col not in df_merged.columns:
        continue
    n_out = ((df_merged[col] < lo) | (df_merged[col] > hi)).sum()
    if n_out:
        print(f"  CLIP  {col}: {n_out:,} values → [{lo}, {hi}]")
        df_merged[col] = df_merged[col].clip(lo, hi)


# =============================================================================
# STEP 4 — Feature engineering
# =============================================================================

_sep("STEP 4 — Feature engineering")

d = df_merged

df_merged["color_contrast_3d"] = np.sqrt(
    d["tbp_lv_deltaA"] ** 2 + d["tbp_lv_deltaB"] ** 2 + d["tbp_lv_deltaL"] ** 2
)
df_merged["elongation"] = d["clin_size_long_diam_mm"] / (d["tbp_lv_minorAxisMM"] + 1e-6)
df_merged["nevi_color_tension"] = d["tbp_lv_nevi_confidence"] - d["tbp_lv_color_std_mean"]
df_merged["log_area"] = np.log1p(d["tbp_lv_areaMM2"])
df_merged["stdL_ratio"] = d["tbp_lv_stdL"] / (d["tbp_lv_stdLExt"] + 1e-6)
df_merged["compactness"] = (
    4 * np.pi * d["tbp_lv_areaMM2"] / (d["tbp_lv_perimeterMM"] ** 2 + 1e-6)
)
df_merged["chroma_contrast"] = d["tbp_lv_C"] - d["tbp_lv_Cext"]
df_merged["radial_color_ratio"] = d["tbp_lv_radial_color_std_max"] / (
    d["tbp_lv_color_std_mean"] + 1e-6
)

for feat in ["color_contrast_3d", "elongation", "nevi_color_tension", "log_area",
             "stdL_ratio", "compactness", "chroma_contrast", "radial_color_ratio"]:
    print(f"  ADDED {feat}")


# =============================================================================
# STEP 5 — Select final columns → datasetv4
# =============================================================================

_sep("STEP 5 — Saving datasetv4_merged_cleaned.csv")

missing_final = [c for c in FINAL_COLS if c not in df_merged.columns]
if missing_final:
    print(f"  WARNING — missing columns: {missing_final}")

df_v4 = df_merged[[c for c in FINAL_COLS if c in df_merged.columns]].copy()
df_v4.to_csv(DATASETS / "datasetv4_merged_cleaned_engineered.csv", index=False)
print(f"  Saved: datasetv4_merged_cleaned_engineered.csv  ({df_v4.shape[0]:,} rows x {df_v4.shape[1]} cols)")


# =============================================================================
# STEP 6 — Malignant only → datasetv5
# =============================================================================

_sep("STEP 6 — Saving datasetv5_cleaned_all_malignant.csv")

df_v5 = df_v4[df_v4["malignant"] == 1].copy()
df_v5.to_csv(DATASETS / "datasetv5_cleaned_all_malignant.csv", index=False)
print(f"  Saved: datasetv5_cleaned_all_malignant.csv  ({len(df_v5):,} rows)")


# =============================================================================
# STEP 7 — Benign only → datasetv6
# =============================================================================

_sep("STEP 7 — Saving datasetv6_cleaned_all_benign.csv")

df_v6 = df_v4[df_v4["malignant"] == 0].copy()
df_v6.to_csv(DATASETS / "datasetv6_cleaned_all_benign.csv", index=False)
print(f"  Saved: datasetv6_cleaned_all_benign.csv  ({len(df_v6):,} rows)")


# =============================================================================
# SUMMARY
# =============================================================================

_sep("DONE")
print(f"""
  datasetv3_raw_metadata_labels_merged.csv : {df_merged.shape[0]:,} rows x {df_merged.shape[1]} cols (all original columns)
  datasetv4_merged_cleaned.csv             : {df_v4.shape[0]:,} rows x {df_v4.shape[1]} cols
  datasetv5_cleaned_all_malignant.csv      : {len(df_v5):,} rows
  datasetv6_cleaned_all_benign.csv         : {len(df_v6):,} rows
""")
