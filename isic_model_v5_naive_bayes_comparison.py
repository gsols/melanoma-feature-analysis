"""
ISIC 2024 — V5: Range-Based Malignant Similarity Model (Naive Bayes)

Trains TWO Naive Bayes variants for comparison:
  1. GaussianNB     — on raw continuous features
  2. CategoricalNB  — on range-labeled categorical features (matches similarity logic)

CONCEPT:
  Never train directly on malignant records. Use them only as a reference profile.
  For each feature, find the dominant malignant range. Score every benign record
  by how many features match those dominant ranges. Train on benign-only with
  the engineered `near_malignant` target.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)

# ============================================================================
# CONFIGURATION
# ============================================================================

FEATURE_RANGES = {
    'age_approx': [(0, 20, 'Child'), (21, 40, 'Young Adult'), (41, 60, 'Middle Age'), (61, 100, 'Senior')],
    'clin_size_long_diam_mm': [(0, 4, 'Very Small'), (4, 6, 'Small'), (6, 10, 'Medium'), (10, 1000, 'Large')],
    'tbp_lv_area_perim_ratio': [(0, 0.5, 'Low'), (0.5, 1.0, 'Medium'), (1.0, 2.0, 'High'), (2.0, 10, 'Very High')],
    'tbp_lv_norm_border': [(0, 1, 'Low'), (1, 2, 'Moderate'), (2, 5, 'High'), (5, 100, 'Very High')],
    'tbp_lv_symm_2axis': [(0, 0.3, 'Asymmetric'), (0.3, 0.6, 'Moderate'), (0.6, 0.9, 'Symmetric'), (0.9, 2, 'Very Symmetric')],
    'tbp_lv_eccentricity': [(0, 0.3, 'Low'), (0.3, 0.6, 'Moderate'), (0.6, 0.85, 'High'), (0.85, 2, 'Very High')],
    'tbp_lv_color_std_mean': [(0, 5, 'Low'), (5, 15, 'Moderate'), (15, 30, 'High'), (30, 200, 'Very High')],
    'tbp_lv_norm_color': [(0, 1, 'Low'), (1, 2, 'Moderate'), (2, 5, 'High'), (5, 100, 'Very High')],
    'tbp_lv_deltaLBnorm': [(0, 5, 'Very Low'), (5, 10, 'Low'), (10, 20, 'Moderate'), (20, 200, 'High')],
    'tbp_lv_radial_color_std_max': [(0, 5, 'Low'), (5, 15, 'Moderate'), (15, 30, 'High'), (30, 200, 'Very High')],
    'tbp_lv_nevi_confidence': [(0, 0.33, 'Low'), (0.33, 0.67, 'Moderate'), (0.67, 1.0, 'High')],
    'color_contrast_3d': [(0, 10, 'Low'), (10, 30, 'Moderate'), (30, 60, 'High'), (60, 200, 'Very High')],
    'elongation': [(0, 0.5, 'Compact'), (0.5, 0.75, 'Moderate'), (0.75, 1.0, 'Elongated')],
    'nevi_color_tension': [(0, 5, 'Low'), (5, 15, 'Moderate'), (15, 30, 'High'), (30, 200, 'Very High')],
    'log_area': [(0, 4, 'Very Small'), (4, 6, 'Small'), (6, 8, 'Medium'), (8, 15, 'Large')],
    'stdL_ratio': [(0, 0.05, 'Low'), (0.05, 0.15, 'Moderate'), (0.15, 0.35, 'High'), (0.35, 2, 'Very High')],
    'compactness': [(0, 0.5, 'Low'), (0.5, 1.0, 'Moderate'), (1.0, 2.0, 'High'), (2.0, 10, 'Very High')],
    'chroma_contrast': [(0, 10, 'Low'), (10, 30, 'Moderate'), (30, 60, 'High'), (60, 200, 'Very High')],
    'radial_color_ratio': [(0, 0.5, 'Low'), (0.5, 1.0, 'Moderate'), (1.0, 2.0, 'High'), (2.0, 10, 'Very High')],
}

SIMILARITY_THRESHOLD = 0.6
RANDOM_STATE = 42
TEST_SIZE = 0.2

OUTPUT_GRAPHS = "v5_graphs"
OUTPUT_RESULTS = "v5_outputs_naive_bayes_comparison"
OUTPUT_REPORTS = f"{OUTPUT_RESULTS}/reports"
OUTPUT_RESULT_GRAPHS = f"{OUTPUT_RESULTS}/graphs"
OUTPUT_DATASETS = "v5_datasets"


def assign_range_label(value, ranges):
    if pd.isna(value):
        return None
    for low, high, label in ranges:
        if low <= value < high:
            return label
    return ranges[-1][2]


def compute_similarity_score(row, dominant_ranges):
    matches, total = 0, 0
    for feature, dominant_range in dominant_ranges.items():
        range_col = f'{feature}_range'
        if pd.notna(row[range_col]):
            total += 1
            if row[range_col] == dominant_range:
                matches += 1
    return matches / total if total > 0 else 0.0


def evaluate(name, y_true, y_pred, y_proba):
    return {
        'model': name,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'pr_auc': average_precision_score(y_true, y_proba),
    }


def main():
    print("=" * 80)
    print("V5: Range-Based Similarity — Naive Bayes Comparison (Gaussian vs Categorical)")
    print("=" * 80)

    df = pd.read_csv("merged_cleaned.csv")
    print(f"\n✓ Loaded {len(df):,} records (Malignant: {df['malignant'].sum():,}, "
          f"Benign: {(1-df['malignant']).sum():,})")

    # Step 3: assign range labels
    print("\n[Step 3] Assigning range labels...")
    for feature, ranges in FEATURE_RANGES.items():
        df[f'{feature}_range'] = df[feature].apply(
            lambda x, r=ranges: assign_range_label(x, r)
        )

    df_malignant = df[df['malignant'] == 1]
    df_benign = df[df['malignant'] == 0].copy()

    # Step 4-5: graph all features
    print(f"\n[Step 4-5] Graphing all {len(FEATURE_RANGES)} features...")
    os.makedirs(OUTPUT_GRAPHS, exist_ok=True)
    for feature in FEATURE_RANGES:
        range_col = f'{feature}_range'

        mal_counts = df_malignant[range_col].value_counts().sort_index()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.pie(mal_counts, labels=mal_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Malignant: {feature} (n={len(df_malignant)})')
        mal_counts.plot(kind='bar', ax=ax2, color='red', alpha=0.7)
        ax2.set_title(f'Malignant {feature}'), ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_GRAPHS}/malignant_{feature}.png', dpi=100, bbox_inches='tight')
        plt.close()

        ben_counts = df_benign[range_col].value_counts().sort_index()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.pie(ben_counts, labels=ben_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Benign: {feature} (n={len(df_benign):,})')
        ben_counts.plot(kind='bar', ax=ax2, color='blue', alpha=0.7)
        ax2.set_title(f'Benign {feature}'), ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_GRAPHS}/benign_{feature}.png', dpi=100, bbox_inches='tight')
        plt.close()
    print(f"✓ {len(FEATURE_RANGES)*2} feature graphs saved")

    # Step 6: dominant malignant ranges
    print("\n[Step 6] Identifying dominant malignant ranges...")
    dominant_ranges = {}
    for feature in FEATURE_RANGES:
        range_col = f'{feature}_range'
        mal_counts = df_malignant[range_col].value_counts()
        if len(mal_counts) > 0:
            dominant_ranges[feature] = mal_counts.idxmax()
    for f, d in dominant_ranges.items():
        print(f"  {f:32s} → {d}")

    # Step 7: similarity scores
    print("\n[Step 7] Computing similarity scores...")
    df_benign['similarity_score'] = df_benign.apply(
        lambda r: compute_similarity_score(r, dominant_ranges), axis=1
    )
    print(f"  Mean: {df_benign['similarity_score'].mean():.4f}, "
          f"Range: [{df_benign['similarity_score'].min():.4f}, "
          f"{df_benign['similarity_score'].max():.4f}]")

    # Step 8: near_malignant target
    print(f"\n[Step 8] Creating near_malignant target (threshold={SIMILARITY_THRESHOLD})...")
    df_benign['near_malignant'] = (
        df_benign['similarity_score'] >= SIMILARITY_THRESHOLD
    ).astype(int)
    near_mal_count = df_benign['near_malignant'].sum()
    print(f"  Near-malignant: {near_mal_count:,} "
          f"({100*near_mal_count/len(df_benign):.2f}%)")

    # Step 9: prepare features
    print("\n[Step 9] Preparing features and splitting...")
    os.makedirs(OUTPUT_DATASETS, exist_ok=True)
    model_features = list(FEATURE_RANGES.keys())
    range_cols = [f'{f}_range' for f in model_features]

    # Two input variants
    X_continuous = df_benign[model_features].copy()
    X_categorical_raw = df_benign[range_cols].copy().fillna('UNKNOWN')

    # Encode categorical to ordinal integers for CategoricalNB
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_categorical = pd.DataFrame(
        encoder.fit_transform(X_categorical_raw),
        columns=range_cols, index=X_categorical_raw.index
    ).astype(int)
    # CategoricalNB requires non-negative integer category indices
    X_categorical = X_categorical.clip(lower=0)

    y = df_benign['near_malignant'].copy()

    Xc_tr, Xc_te, Xr_tr, Xr_te, y_tr, y_te = train_test_split(
        X_continuous, X_categorical, y,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(Xc_tr):,}, Test: {len(Xc_te):,}")

    # Save datasets
    pd.concat([Xc_tr, y_tr], axis=1).to_csv(
        f'{OUTPUT_DATASETS}/v5_train_benign_similarity.csv', index=False)
    pd.concat([Xc_te, y_te], axis=1).to_csv(
        f'{OUTPUT_DATASETS}/v5_test_benign_similarity.csv', index=False)

    # Step 10: train both models
    print("\n[Step 10] Training Naive Bayes models...")

    gnb = GaussianNB()
    gnb.fit(Xc_tr, y_tr)
    gnb_pred = gnb.predict(Xc_te)
    gnb_proba = gnb.predict_proba(Xc_te)[:, 1]
    print("  ✓ GaussianNB trained")

    cnb = CategoricalNB()
    cnb.fit(Xr_tr, y_tr)
    cnb_pred = cnb.predict(Xr_te)
    cnb_proba = cnb.predict_proba(Xr_te)[:, 1]
    print("  ✓ CategoricalNB trained")

    # Step 11: evaluate
    print("\n[Step 11] Test evaluation...")
    gnb_metrics = evaluate('GaussianNB', y_te, gnb_pred, gnb_proba)
    cnb_metrics = evaluate('CategoricalNB', y_te, cnb_pred, cnb_proba)

    print(f"\n  {'Metric':12s}  {'GaussianNB':>12s}  {'CategoricalNB':>14s}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*14}")
    for m in ['f1', 'accuracy', 'precision', 'recall', 'roc_auc', 'pr_auc']:
        print(f"  {m:12s}  {gnb_metrics[m]:>12.4f}  {cnb_metrics[m]:>14.4f}")

    gnb_cm = confusion_matrix(y_te, gnb_pred)
    cnb_cm = confusion_matrix(y_te, cnb_pred)

    # Step 12: save outputs
    print("\n[Step 12] Saving outputs...")
    os.makedirs(OUTPUT_RESULTS, exist_ok=True)
    os.makedirs(OUTPUT_REPORTS, exist_ok=True)
    os.makedirs(OUTPUT_RESULT_GRAPHS, exist_ok=True)

    # Save both models
    joblib.dump({
        'model': gnb, 'features': model_features,
        'dominant_ranges': dominant_ranges,
        'feature_ranges': FEATURE_RANGES,
        'similarity_threshold': SIMILARITY_THRESHOLD,
        'model_type': 'GaussianNB',
    }, f'{OUTPUT_RESULTS}/v5_gaussian_nb.joblib')

    joblib.dump({
        'model': cnb, 'features': range_cols,
        'encoder': encoder,
        'dominant_ranges': dominant_ranges,
        'feature_ranges': FEATURE_RANGES,
        'similarity_threshold': SIMILARITY_THRESHOLD,
        'model_type': 'CategoricalNB',
    }, f'{OUTPUT_RESULTS}/v5_categorical_nb.joblib')

    # Predictions
    pd.DataFrame({
        'actual': y_te.values,
        'gnb_pred': gnb_pred, 'gnb_proba': gnb_proba,
        'cnb_pred': cnb_pred, 'cnb_proba': cnb_proba,
    }).sort_values('cnb_proba', ascending=False).to_csv(
        f'{OUTPUT_REPORTS}/v5_naive_bayes_predictions.csv', index=False)

    # Metrics JSON
    with open(f'{OUTPUT_REPORTS}/v5_global_metrics.json', 'w') as f:
        json.dump({
            'gaussian_nb': gnb_metrics,
            'categorical_nb': cnb_metrics,
            'similarity_threshold': SIMILARITY_THRESHOLD,
            'train_size': len(Xc_tr), 'test_size': len(Xc_te),
            'benign_near_malignant_count': int(near_mal_count),
            'benign_near_malignant_pct': float(100*near_mal_count/len(df_benign)),
        }, f, indent=2, default=float)

    # Readable report
    report = f"""================================================================================
ISIC 2024 — V5 Naive Bayes Comparison: Range-Based Malignant Similarity
================================================================================

CONCEPT
  Trained on benign records only. Target engineered from similarity to the
  malignant feature profile (dominant ranges). Compares two Naive Bayes
  variants:
    GaussianNB     — raw continuous features
    CategoricalNB  — range labels (matches similarity logic)

DATASET
  Total benign      : {len(df_benign):,}
  Train (80%)       : {len(Xc_tr):,}
  Test  (20%)       : {len(Xc_te):,}
  Near-malignant    : {near_mal_count:,} ({100*near_mal_count/len(df_benign):.2f}%)
  Threshold         : {SIMILARITY_THRESHOLD}

TEST SET METRICS
  {'Metric':12s}  {'GaussianNB':>12s}  {'CategoricalNB':>14s}
  {'-'*12}  {'-'*12}  {'-'*14}
"""
    for m in ['f1', 'accuracy', 'precision', 'recall', 'roc_auc', 'pr_auc']:
        report += f"  {m:12s}  {gnb_metrics[m]:>12.4f}  {cnb_metrics[m]:>14.4f}\n"

    gtn, gfp, gfn, gtp = gnb_cm.ravel()
    ctn, cfp, cfn, ctp = cnb_cm.ravel()
    report += f"""
CONFUSION MATRICES (Test Set)
  GaussianNB    : TP={gtp:,}, FP={gfp:,}, FN={gfn:,}, TN={gtn:,}
  CategoricalNB : TP={ctp:,}, FP={cfp:,}, FN={cfn:,}, TN={ctn:,}

DOMINANT MALIGNANT RANGES (similarity reference)
"""
    for f, d in dominant_ranges.items():
        report += f"  {f:32s} → {d}\n"

    report += "\n" + "=" * 80 + "\n"
    with open(f'{OUTPUT_REPORTS}/v5_readable_report.txt', 'w') as f:
        f.write(report)

    # Step 13: visualizations
    print("\n[Step 13] Creating visualizations...")

    # Side-by-side confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(gnb_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
    axes[0].set_title('GaussianNB - Confusion Matrix')
    axes[0].set_xticklabels(['Not Near-Mal', 'Near-Mal'])
    axes[0].set_yticklabels(['Not Near-Mal', 'Near-Mal'])
    sns.heatmap(cnb_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False)
    axes[1].set_title('CategoricalNB - Confusion Matrix')
    axes[1].set_xticklabels(['Not Near-Mal', 'Near-Mal'])
    axes[1].set_yticklabels(['Not Near-Mal', 'Near-Mal'])
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_RESULT_GRAPHS}/confusion_matrices.png', dpi=100, bbox_inches='tight')
    plt.close()

    # ROC comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr_g, tpr_g, _ = roc_curve(y_te, gnb_proba)
    fpr_c, tpr_c, _ = roc_curve(y_te, cnb_proba)
    ax.plot(fpr_g, tpr_g, label=f"GaussianNB (AUC={gnb_metrics['roc_auc']:.4f})", linewidth=2)
    ax.plot(fpr_c, tpr_c, label=f"CategoricalNB (AUC={cnb_metrics['roc_auc']:.4f})", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate'), ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison'), ax.legend(), ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_RESULT_GRAPHS}/roc_comparison.png', dpi=100, bbox_inches='tight')
    plt.close()

    # PR comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    pg, rg, _ = precision_recall_curve(y_te, gnb_proba)
    pc, rc, _ = precision_recall_curve(y_te, cnb_proba)
    ax.plot(rg, pg, label=f"GaussianNB (AUC={gnb_metrics['pr_auc']:.4f})", linewidth=2)
    ax.plot(rc, pc, label=f"CategoricalNB (AUC={cnb_metrics['pr_auc']:.4f})", linewidth=2)
    ax.set_xlabel('Recall'), ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Comparison'), ax.legend(), ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_RESULT_GRAPHS}/pr_comparison.png', dpi=100, bbox_inches='tight')
    plt.close()

    # Metrics bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_to_plot = ['f1', 'accuracy', 'precision', 'recall', 'roc_auc', 'pr_auc']
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    g_vals = [gnb_metrics[m] for m in metrics_to_plot]
    c_vals = [cnb_metrics[m] for m in metrics_to_plot]
    ax.bar(x - width/2, g_vals, width, label='GaussianNB', color='steelblue')
    ax.bar(x + width/2, c_vals, width, label='CategoricalNB', color='seagreen')
    ax.set_xticks(x), ax.set_xticklabels(metrics_to_plot)
    ax.set_ylabel('Score'), ax.set_title('Metric Comparison: GaussianNB vs CategoricalNB')
    ax.legend(), ax.grid(alpha=0.3, axis='y'), ax.set_ylim([0, 1.05])
    for i, (g, c) in enumerate(zip(g_vals, c_vals)):
        ax.text(i - width/2, g + 0.01, f'{g:.3f}', ha='center', fontsize=8)
        ax.text(i + width/2, c + 0.01, f'{c:.3f}', ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_RESULT_GRAPHS}/metrics_comparison.png', dpi=100, bbox_inches='tight')
    plt.close()

    # Probability distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(gnb_proba[y_te == 0], bins=50, alpha=0.6, label='Not Near-Mal', color='blue')
    axes[0].hist(gnb_proba[y_te == 1], bins=50, alpha=0.6, label='Near-Mal', color='red')
    axes[0].set_title('GaussianNB Probabilities'), axes[0].legend()
    axes[0].set_xlabel('P(near_malignant)'), axes[0].set_ylabel('Count')
    axes[1].hist(cnb_proba[y_te == 0], bins=50, alpha=0.6, label='Not Near-Mal', color='blue')
    axes[1].hist(cnb_proba[y_te == 1], bins=50, alpha=0.6, label='Near-Mal', color='red')
    axes[1].set_title('CategoricalNB Probabilities'), axes[1].legend()
    axes[1].set_xlabel('P(near_malignant)'), axes[1].set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_RESULT_GRAPHS}/probability_distributions.png', dpi=100, bbox_inches='tight')
    plt.close()

    # Similarity score distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df_benign[df_benign['near_malignant'] == 0]['similarity_score'],
            bins=50, alpha=0.6, label='Not Near-Mal', color='blue')
    ax.hist(df_benign[df_benign['near_malignant'] == 1]['similarity_score'],
            bins=50, alpha=0.6, label='Near-Mal', color='red')
    ax.axvline(SIMILARITY_THRESHOLD, color='black', linestyle='--', linewidth=2,
               label=f'Threshold = {SIMILARITY_THRESHOLD}')
    ax.set_xlabel('Similarity Score'), ax.set_ylabel('Count')
    ax.set_title('Benign Similarity Score Distribution'), ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_RESULT_GRAPHS}/similarity_distribution.png', dpi=100, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 80)
    print("V5 NAIVE BAYES PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\n  Models:   {OUTPUT_RESULTS}/v5_gaussian_nb.joblib")
    print(f"            {OUTPUT_RESULTS}/v5_categorical_nb.joblib")
    print(f"  Report:   {OUTPUT_REPORTS}/v5_readable_report.txt")
    print(f"  Metrics:  {OUTPUT_REPORTS}/v5_global_metrics.json")
    print(f"  Graphs:   {OUTPUT_RESULT_GRAPHS}/ (6 comparison graphs)")
    print(f"  Features: {OUTPUT_GRAPHS}/ ({len(FEATURE_RANGES)*2} graphs)")
    print(f"\n  Winner by F1: "
          f"{'CategoricalNB' if cnb_metrics['f1'] > gnb_metrics['f1'] else 'GaussianNB'} "
          f"({max(gnb_metrics['f1'], cnb_metrics['f1']):.4f})")


if __name__ == "__main__":
    main()
