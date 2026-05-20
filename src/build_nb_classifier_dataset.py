"""
Build Naive Bayes Classifier Dataset

Steps:
  1. Combine v1 train and test benign similarity CSVs into one dataset.
  2. Replace continuous feature values with categorical labels using the
     same FEATURE_RANGES defined in isic_model_v1_naive_bayes_comparison.py.
  3. Replace near_malignant (0/1) with 'no'/'yes'.
  4. Save the combined categorical CSV to nb_classifier_model_v1/.
  5. For every feature, generate a frequency table CSV showing how many
     records in each category are near_malignant (yes/no), plus
     conditional probabilities P(category | yes) and P(category | no).
"""

import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROCESSED = (_HERE / "../models/model_v1/processed_data").resolve()
_OUT = (_HERE / "../nb_classifier_model_v1").resolve()
_OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Feature ranges (mirrored from isic_model_v1_naive_bayes_comparison.py)
# ---------------------------------------------------------------------------
FEATURE_RANGES = {
    'age_approx':                  [(0, 20, 'Child'), (21, 40, 'Young Adult'), (41, 60, 'Middle Age'), (61, 100, 'Senior')],
    'clin_size_long_diam_mm':      [(0, 4, 'Very Small'), (4, 6, 'Small'), (6, 10, 'Medium'), (10, 1000, 'Large')],
    'tbp_lv_area_perim_ratio':     [(0, 0.5, 'Low'), (0.5, 1.0, 'Medium'), (1.0, 2.0, 'High'), (2.0, 10, 'Very High')],
    'tbp_lv_norm_border':          [(0, 1, 'Low'), (1, 2, 'Moderate'), (2, 5, 'High'), (5, 100, 'Very High')],
    'tbp_lv_symm_2axis':           [(0, 0.3, 'Asymmetric'), (0.3, 0.6, 'Moderate'), (0.6, 0.9, 'Symmetric'), (0.9, 2, 'Very Symmetric')],
    'tbp_lv_eccentricity':         [(0, 0.3, 'Low'), (0.3, 0.6, 'Moderate'), (0.6, 0.85, 'High'), (0.85, 2, 'Very High')],
    'tbp_lv_color_std_mean':       [(0, 5, 'Low'), (5, 15, 'Moderate'), (15, 30, 'High'), (30, 200, 'Very High')],
    'tbp_lv_norm_color':           [(0, 1, 'Low'), (1, 2, 'Moderate'), (2, 5, 'High'), (5, 100, 'Very High')],
    'tbp_lv_deltaLBnorm':          [(0, 5, 'Very Low'), (5, 10, 'Low'), (10, 20, 'Moderate'), (20, 200, 'High')],
    'tbp_lv_radial_color_std_max': [(0, 5, 'Low'), (5, 15, 'Moderate'), (15, 30, 'High'), (30, 200, 'Very High')],
    'tbp_lv_nevi_confidence':      [(0, 0.33, 'Low'), (0.33, 0.67, 'Moderate'), (0.67, 1.0, 'High')],
    'color_contrast_3d':           [(0, 10, 'Low'), (10, 30, 'Moderate'), (30, 60, 'High'), (60, 200, 'Very High')],
    'elongation':                  [(0, 0.5, 'Compact'), (0.5, 0.75, 'Moderate'), (0.75, 1.0, 'Elongated')],
    'nevi_color_tension':          [(0, 5, 'Low'), (5, 15, 'Moderate'), (15, 30, 'High'), (30, 200, 'Very High')],
    'log_area':                    [(0, 4, 'Very Small'), (4, 6, 'Small'), (6, 8, 'Medium'), (8, 15, 'Large')],
    'stdL_ratio':                  [(0, 0.05, 'Low'), (0.05, 0.15, 'Moderate'), (0.15, 0.35, 'High'), (0.35, 2, 'Very High')],
    'compactness':                 [(0, 0.5, 'Low'), (0.5, 1.0, 'Moderate'), (1.0, 2.0, 'High'), (2.0, 10, 'Very High')],
    'chroma_contrast':             [(0, 10, 'Low'), (10, 30, 'Moderate'), (30, 60, 'High'), (60, 200, 'Very High')],
    'radial_color_ratio':          [(0, 0.5, 'Low'), (0.5, 1.0, 'Moderate'), (1.0, 2.0, 'High'), (2.0, 10, 'Very High')],
}

# Descriptive title suffix used in each feature's output filename
FEATURE_TITLES = {
    'age_approx':                  'age_group',
    'clin_size_long_diam_mm':      'lesion_diameter_size',
    'tbp_lv_area_perim_ratio':     'area_perimeter_ratio',
    'tbp_lv_norm_border':          'border_irregularity',
    'tbp_lv_symm_2axis':           'shape_symmetry',
    'tbp_lv_eccentricity':         'shape_eccentricity',
    'tbp_lv_color_std_mean':       'color_std_variation',
    'tbp_lv_norm_color':           'normalized_color',
    'tbp_lv_deltaLBnorm':          'lightness_delta',
    'tbp_lv_radial_color_std_max': 'radial_color_variation',
    'tbp_lv_nevi_confidence':      'nevi_confidence_level',
    'color_contrast_3d':           '3d_color_contrast',
    'elongation':                  'shape_elongation',
    'nevi_color_tension':          'nevi_color_tension',
    'log_area':                    'log_lesion_area',
    'stdL_ratio':                  'lightness_std_ratio',
    'compactness':                 'shape_compactness',
    'chroma_contrast':             'chroma_color_contrast',
    'radial_color_ratio':          'radial_color_ratio',
}


def assign_label(value, ranges):
    if pd.isna(value):
        return None
    for low, high, label in ranges:
        if low <= value < high:
            return label
    return ranges[-1][2]


def build_frequency_table(df, feature, near_mal_col='near_malignant'):
    """
    Returns a DataFrame with the format:
        (blank), yes, no, P(yes), P(no)
        <cat1>,  ..., ..., ...,   ...
        total,   ..., ..., 1,     1
    """
    categories = [label for _, _, label in FEATURE_RANGES[feature]]

    yes_total = (df[near_mal_col] == 'yes').sum()
    no_total  = (df[near_mal_col] == 'no').sum()

    rows = []
    for cat in categories:
        mask = df[feature] == cat
        yes_count = ((df[near_mal_col] == 'yes') & mask).sum()
        no_count  = ((df[near_mal_col] == 'no')  & mask).sum()
        p_yes = yes_count / yes_total if yes_total > 0 else 0.0
        p_no  = no_count  / no_total  if no_total  > 0 else 0.0
        rows.append({
            '': cat,
            'yes':    yes_count,
            'no':     no_count,
            'P(yes)': round(p_yes, 6),
            'P(no)':  round(p_no,  6),
        })

    rows.append({
        '':      'total',
        'yes':    yes_total,
        'no':     no_total,
        'P(yes)': 1,
        'P(no)':  1,
    })

    return pd.DataFrame(rows, columns=['', 'yes', 'no', 'P(yes)', 'P(no)'])


def main():
    # ------------------------------------------------------------------
    # Step 1: Combine train + test
    # ------------------------------------------------------------------
    print("Loading train and test CSVs...")
    train = pd.read_csv(_PROCESSED / "v1_train_benign_similarity.csv")
    test  = pd.read_csv(_PROCESSED / "v1_test_benign_similarity.csv")
    df = pd.concat([train, test], ignore_index=True)
    print(f"  Train: {len(train):,} | Test: {len(test):,} | Combined: {len(df):,}")

    # ------------------------------------------------------------------
    # Step 2: Replace continuous values with categorical labels
    # ------------------------------------------------------------------
    print("Applying feature range categorization...")
    for feature, ranges in FEATURE_RANGES.items():
        df[feature] = df[feature].apply(lambda v, r=ranges: assign_label(v, r))

    # ------------------------------------------------------------------
    # Step 3: Replace near_malignant 0/1 with no/yes
    # ------------------------------------------------------------------
    df['near_malignant'] = df['near_malignant'].map({1: 'yes', 0: 'no'})

    # ------------------------------------------------------------------
    # Step 4: Save combined categorical CSV
    # ------------------------------------------------------------------
    out_csv = _OUT / "v1_benign_near_malignant_similarity_full.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Saved combined dataset → {out_csv.name}")

    # ------------------------------------------------------------------
    # Step 5: Per-feature frequency tables
    # ------------------------------------------------------------------
    print("Building per-feature frequency tables...")
    for feature in FEATURE_RANGES:
        title = FEATURE_TITLES[feature]
        table = build_frequency_table(df, feature)
        out_name = f"{feature}_{title}_near_malignant_counts.csv"
        table.to_csv(_OUT / out_name, index=False)
        print(f"  Saved → {out_name}")

    print(f"\nDone. All outputs saved to: {_OUT}")


if __name__ == "__main__":
    main()
