"""
ISIC 2024 — V5: Range-Based Malignant Similarity Model

CONCEPT:
  Never train directly on malignant records. Instead, use them only as a reference
  profile. For each feature, find which range holds the most malignant records
  (the "dominant malignant range"). Then score every benign record by how many
  of its feature values fall within those dominant malignant ranges.

  The model is trained on benign records only, with the engineered target
  `near_malignant` derived from this similarity score. This avoids class
  imbalance entirely and reframes the question:

      "How closely does this benign lesion resemble a confirmed malignant?"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)

# ============================================================================
# CUSTOMIZABLE FEATURE RANGES (edit values here to retune the model)
# ============================================================================

FEATURE_RANGES = {
    'age_approx': [
        (0, 20, 'Child (0-20)'),
        (21, 40, 'Young Adult (21-40)'),
        (41, 60, 'Middle Age (41-60)'),
        (61, 100, 'Senior (61+)'),
    ],
    'clin_size_long_diam_mm': [
        (0, 4, 'Very Small (0-4mm)'),
        (4, 6, 'Small (4-6mm)'),
        (6, 10, 'Medium (6-10mm)'),
        (10, 1000, 'Large (>10mm)'),
    ],
    'tbp_lv_area_perim_ratio': [
        (0, 0.5, 'Low (0-0.5)'),
        (0.5, 1.0, 'Medium (0.5-1.0)'),
        (1.0, 2.0, 'High (1.0-2.0)'),
        (2.0, 10, 'Very High (>2.0)'),
    ],
    'tbp_lv_norm_border': [
        (0, 1, 'Low (0-1)'),
        (1, 2, 'Moderate (1-2)'),
        (2, 5, 'High (2-5)'),
        (5, 100, 'Very High (>5)'),
    ],
    'tbp_lv_symm_2axis': [
        (0, 0.3, 'Asymmetric (0-0.3)'),
        (0.3, 0.6, 'Moderate (0.3-0.6)'),
        (0.6, 0.9, 'Symmetric (0.6-0.9)'),
        (0.9, 2, 'Very Symmetric (>0.9)'),
    ],
    'tbp_lv_eccentricity': [
        (0, 0.3, 'Low (0-0.3)'),
        (0.3, 0.6, 'Moderate (0.3-0.6)'),
        (0.6, 0.85, 'High (0.6-0.85)'),
        (0.85, 2, 'Very High (>0.85)'),
    ],
    'tbp_lv_color_std_mean': [
        (0, 5, 'Low (0-5)'),
        (5, 15, 'Moderate (5-15)'),
        (15, 30, 'High (15-30)'),
        (30, 200, 'Very High (>30)'),
    ],
    'tbp_lv_norm_color': [
        (0, 1, 'Low (0-1)'),
        (1, 2, 'Moderate (1-2)'),
        (2, 5, 'High (2-5)'),
        (5, 100, 'Very High (>5)'),
    ],
    'tbp_lv_deltaLBnorm': [
        (0, 5, 'Very Low (0-5)'),
        (5, 10, 'Low (5-10)'),
        (10, 20, 'Moderate (10-20)'),
        (20, 200, 'High (>20)'),
    ],
    'tbp_lv_radial_color_std_max': [
        (0, 5, 'Low (0-5)'),
        (5, 15, 'Moderate (5-15)'),
        (15, 30, 'High (15-30)'),
        (30, 200, 'Very High (>30)'),
    ],
    'tbp_lv_nevi_confidence': [
        (0, 0.33, 'Low (0-33%)'),
        (0.33, 0.67, 'Moderate (33-67%)'),
        (0.67, 1.0, 'High (67-100%)'),
    ],
    'color_contrast_3d': [
        (0, 10, 'Low (0-10)'),
        (10, 30, 'Moderate (10-30)'),
        (30, 60, 'High (30-60)'),
        (60, 200, 'Very High (>60)'),
    ],
    'elongation': [
        (0, 0.5, 'Compact (0-0.5)'),
        (0.5, 0.75, 'Moderate (0.5-0.75)'),
        (0.75, 1.0, 'Elongated (0.75-1.0)'),
    ],
    'nevi_color_tension': [
        (0, 5, 'Low (0-5)'),
        (5, 15, 'Moderate (5-15)'),
        (15, 30, 'High (15-30)'),
        (30, 200, 'Very High (>30)'),
    ],
    'log_area': [
        (0, 4, 'Very Small (0-4)'),
        (4, 6, 'Small (4-6)'),
        (6, 8, 'Medium (6-8)'),
        (8, 15, 'Large (>8)'),
    ],
    'stdL_ratio': [
        (0, 0.05, 'Low (0-0.05)'),
        (0.05, 0.15, 'Moderate (0.05-0.15)'),
        (0.15, 0.35, 'High (0.15-0.35)'),
        (0.35, 2, 'Very High (>0.35)'),
    ],
    'compactness': [
        (0, 0.5, 'Low (0-0.5)'),
        (0.5, 1.0, 'Moderate (0.5-1.0)'),
        (1.0, 2.0, 'High (1.0-2.0)'),
        (2.0, 10, 'Very High (>2.0)'),
    ],
    'chroma_contrast': [
        (0, 10, 'Low (0-10)'),
        (10, 30, 'Moderate (10-30)'),
        (30, 60, 'High (30-60)'),
        (60, 200, 'Very High (>60)'),
    ],
    'radial_color_ratio': [
        (0, 0.5, 'Low (0-0.5)'),
        (0.5, 1.0, 'Moderate (0.5-1.0)'),
        (1.0, 2.0, 'High (1.0-2.0)'),
        (2.0, 10, 'Very High (>2.0)'),
    ],
}

SIMILARITY_THRESHOLD = 0.6
RANDOM_STATE = 42
TEST_SIZE = 0.2

OUTPUT_GRAPHS = "v5_graphs"
OUTPUT_RESULTS = "v5_outputs_range_based_similarity"
OUTPUT_REPORTS = "v5_outputs_range_based_similarity/reports"
OUTPUT_RESULT_GRAPHS = "v5_outputs_range_based_similarity/graphs"
OUTPUT_DATASETS = "v5_datasets"
MODEL_NAME = "v5_range_based_similarity.joblib"


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


def main():
    print("=" * 80)
    print("V5: Range-Based Malignant Similarity Model")
    print("=" * 80)

    df = pd.read_csv("merged_cleaned.csv")
    print(f"\n✓ Loaded {len(df):,} records")
    print(f"  Malignant: {df['malignant'].sum():,}")
    print(f"  Benign: {(1 - df['malignant']).sum():,}")

    # Step 3: assign range labels
    print("\n[Step 3] Assigning range labels...")
    for feature, ranges in FEATURE_RANGES.items():
        df[f'{feature}_range'] = df[feature].apply(
            lambda x, r=ranges: assign_range_label(x, r)
        )

    df_malignant = df[df['malignant'] == 1]
    df_benign = df[df['malignant'] == 0].copy()
    print(f"  Malignant: {len(df_malignant):,}, Benign: {len(df_benign):,}")

    # Step 4-5: graph all features
    print(f"\n[Step 4-5] Graphing all {len(FEATURE_RANGES)} features...")
    os.makedirs(OUTPUT_GRAPHS, exist_ok=True)

    for feature in FEATURE_RANGES:
        range_col = f'{feature}_range'

        mal_counts = df_malignant[range_col].value_counts().sort_index()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.pie(mal_counts, labels=mal_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Malignant: {feature}\n(n={len(df_malignant)})')
        mal_counts.plot(kind='bar', ax=ax2, color='red', alpha=0.7)
        ax2.set_title(f'Malignant {feature} Count')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_GRAPHS}/malignant_{feature}.png',
                    dpi=100, bbox_inches='tight')
        plt.close()

        ben_counts = df_benign[range_col].value_counts().sort_index()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.pie(ben_counts, labels=ben_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Benign: {feature}\n(n={len(df_benign):,})')
        ben_counts.plot(kind='bar', ax=ax2, color='blue', alpha=0.7)
        ax2.set_title(f'Benign {feature} Count')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_GRAPHS}/benign_{feature}.png',
                    dpi=100, bbox_inches='tight')
        plt.close()

    print(f"✓ All {len(FEATURE_RANGES) * 2} graphs saved")

    # Step 6: identify dominant malignant ranges
    print("\n[Step 6] Identifying dominant malignant ranges...")
    dominant_ranges = {}
    for feature in FEATURE_RANGES:
        range_col = f'{feature}_range'
        mal_counts = df_malignant[range_col].value_counts()
        if len(mal_counts) > 0:
            dominant_ranges[feature] = mal_counts.idxmax()
    for feature, dom in dominant_ranges.items():
        print(f"  {feature:32s} → {dom}")

    # Step 7: compute similarity scores
    print("\n[Step 7] Computing similarity scores for benign records...")
    df_benign['similarity_score'] = df_benign.apply(
        lambda row: compute_similarity_score(row, dominant_ranges), axis=1
    )
    print(f"  Mean: {df_benign['similarity_score'].mean():.4f}")
    print(f"  Std:  {df_benign['similarity_score'].std():.4f}")
    print(f"  Range: [{df_benign['similarity_score'].min():.4f}, "
          f"{df_benign['similarity_score'].max():.4f}]")

    # Step 8: engineered target
    print("\n[Step 8] Creating near_malignant target...")
    df_benign['near_malignant'] = (
        df_benign['similarity_score'] >= SIMILARITY_THRESHOLD
    ).astype(int)
    near_mal_count = df_benign['near_malignant'].sum()
    print(f"  Threshold: {SIMILARITY_THRESHOLD}")
    print(f"  Near-malignant (1): {near_mal_count:,} "
          f"({100*near_mal_count/len(df_benign):.2f}%)")
    print(f"  Not near-malignant (0): {len(df_benign) - near_mal_count:,}")

    # Step 9: split
    print("\n[Step 9] Train/test split (80/20)...")
    os.makedirs(OUTPUT_DATASETS, exist_ok=True)
    model_features = list(FEATURE_RANGES.keys())
    X = df_benign[model_features].copy()
    y = df_benign['near_malignant'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train):,} (near_mal: {y_train.sum():,})")
    print(f"  Test:  {len(X_test):,} (near_mal: {y_test.sum():,})")

    pd.concat([X_train, y_train], axis=1).to_csv(
        f'{OUTPUT_DATASETS}/v5_train_benign_similarity.csv', index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(
        f'{OUTPUT_DATASETS}/v5_test_benign_similarity.csv', index=False)

    # Step 10: train
    print("\n[Step 10] Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=20,
        min_samples_leaf=10, random_state=RANDOM_STATE,
        n_jobs=-1, class_weight='balanced'
    )
    model.fit(X_train, y_train)
    print("✓ Model trained")

    # Step 11: cross-validation
    print("\n[Step 11] Cross-validation (5-fold)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_metrics = {'f1': [], 'accuracy': [], 'precision': [], 'recall': []}

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_cv_t, X_cv_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_cv_t, y_cv_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
        fold_model = RandomForestClassifier(
            n_estimators=1000, max_depth=20, min_samples_split=10,
            min_samples_leaf=5, random_state=RANDOM_STATE,
            n_jobs=-1, class_weight='balanced'
        )
        fold_model.fit(X_cv_t, y_cv_t)
        y_cv_p = fold_model.predict(X_cv_v)
        cv_metrics['f1'].append(f1_score(y_cv_v, y_cv_p, zero_division=0))
        cv_metrics['accuracy'].append(accuracy_score(y_cv_v, y_cv_p))
        cv_metrics['precision'].append(precision_score(y_cv_v, y_cv_p, zero_division=0))
        cv_metrics['recall'].append(recall_score(y_cv_v, y_cv_p, zero_division=0))

    print("  CV Results (mean ± std):")
    for m, vals in cv_metrics.items():
        print(f"    {m:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Step 12: test evaluation
    print("\n[Step 12] Test evaluation...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    test_metrics = {
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'pr_auc': average_precision_score(y_test, y_pred_proba),
    }
    print("  TEST METRICS:")
    for m, v in test_metrics.items():
        print(f"    {m:12s}: {v:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix: TP={tp:,}, FP={fp:,}, FN={fn:,}, TN={tn:,}")

    # Step 13: save model and outputs
    print("\n[Step 13] Saving model + results...")
    os.makedirs(OUTPUT_RESULTS, exist_ok=True)
    os.makedirs(OUTPUT_REPORTS, exist_ok=True)
    os.makedirs(OUTPUT_RESULT_GRAPHS, exist_ok=True)

    # Save trained model
    model_payload = {
        'model': model,
        'features': model_features,
        'dominant_ranges': dominant_ranges,
        'feature_ranges': FEATURE_RANGES,
        'similarity_threshold': SIMILARITY_THRESHOLD,
    }
    joblib.dump(model_payload, f'{OUTPUT_RESULTS}/{MODEL_NAME}')
    print(f"✓ Model saved to {OUTPUT_RESULTS}/{MODEL_NAME}")

    feature_importance = pd.DataFrame({
        'feature': model_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    pd.DataFrame({
        'actual': y_test.values,
        'predicted': y_pred,
        'probability': y_pred_proba,
    }).sort_values('probability', ascending=False).to_csv(
        f'{OUTPUT_REPORTS}/v5_range_based_similarity_predictions.csv', index=False)

    feature_importance.to_csv(
        f'{OUTPUT_REPORTS}/v5_range_based_similarity_feature_importance.csv', index=False)

    # CV fold metrics
    pd.DataFrame(cv_metrics).to_csv(
        f'{OUTPUT_REPORTS}/v5_range_based_similarity_fold_metrics.csv', index=False)

    metrics_out = {
        **{f'test_{k}': float(v) for k, v in test_metrics.items()},
        **{f'cv_{k}_mean': float(np.mean(v)) for k, v in cv_metrics.items()},
        **{f'cv_{k}_std': float(np.std(v)) for k, v in cv_metrics.items()},
        'benign_near_malignant_count': int(near_mal_count),
        'benign_near_malignant_pct': float(100*near_mal_count/len(df_benign)),
        'similarity_threshold': SIMILARITY_THRESHOLD,
        'total_features': len(model_features),
    }
    with open(f'{OUTPUT_REPORTS}/v5_global_metrics.json', 'w') as f:
        json.dump(metrics_out, f, indent=2)

    # Readable report
    report = f"""================================================================================
ISIC 2024 — V5 Random Forest: Range-Based Malignant Similarity
================================================================================

CONCEPT
  Trained on benign records only. Target is engineered from similarity to
  the malignant feature profile. Answers:
  "How closely does this benign lesion resemble a confirmed malignant?"

DATASET
  Total benign records      : {len(df_benign):,}
  Training (80%)            : {len(X_train):,}
    Near-malignant (1)      : {y_train.sum():,}
    Not near-malignant (0)  : {(1-y_train).sum():,}
  Test (20%)                : {len(X_test):,}
    Near-malignant (1)      : {y_test.sum():,}
    Not near-malignant (0)  : {(1-y_test).sum():,}
  Similarity threshold      : {SIMILARITY_THRESHOLD}

MODEL
  Algorithm                 : Random Forest
  n_estimators              : 1000
  max_depth                 : 20
  class_weight              : balanced
  Total features            : {len(model_features)}

CROSS-VALIDATION (5-fold)
  f1         : {np.mean(cv_metrics['f1']):.4f} ± {np.std(cv_metrics['f1']):.4f}
  accuracy   : {np.mean(cv_metrics['accuracy']):.4f} ± {np.std(cv_metrics['accuracy']):.4f}
  precision  : {np.mean(cv_metrics['precision']):.4f} ± {np.std(cv_metrics['precision']):.4f}
  recall     : {np.mean(cv_metrics['recall']):.4f} ± {np.std(cv_metrics['recall']):.4f}

TEST SET
  F1         : {test_metrics['f1']:.4f}
  Accuracy   : {test_metrics['accuracy']:.4f}
  Precision  : {test_metrics['precision']:.4f}
  Recall     : {test_metrics['recall']:.4f}
  ROC-AUC    : {test_metrics['roc_auc']:.4f}
  PR-AUC     : {test_metrics['pr_auc']:.4f}
  Confusion  : TP={tp:,}, FP={fp:,}, FN={fn:,}, TN={tn:,}

DOMINANT MALIGNANT RANGES (used for similarity scoring)
"""
    for feature, dom in dominant_ranges.items():
        report += f"  {feature:32s} → {dom}\n"

    report += "\nTOP 20 FEATURES BY IMPORTANCE\n"
    for rank, (_, row) in enumerate(feature_importance.head(20).iterrows(), 1):
        report += f"  {rank:2d}  {row['feature']:32s} {row['importance']:.4f}\n"

    report += "\n" + "=" * 80 + "\n"
    with open(f'{OUTPUT_REPORTS}/v5_report.txt', 'w') as f:
        f.write(report)

    # Step 14: visualizations
    print("\n[Step 14] Result visualizations...")
    # graphs dir already created above

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title('Confusion Matrix - Test Set')
    ax.set_xticklabels(['Not Near-Mal', 'Near-Mal'])
    ax.set_yticklabels(['Not Near-Mal', 'Near-Mal'])
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_RESULT_GRAPHS}/confusion_matrix.png', dpi=100)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {test_metrics['roc_auc']:.4f}", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('FPR'), ax.set_ylabel('TPR')
    ax.set_title('ROC Curve'), ax.legend(), ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_RESULT_GRAPHS}/roc_curve.png', dpi=100)
    plt.close()

    pr_p, pr_r, _ = precision_recall_curve(y_test, y_pred_proba)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(pr_r, pr_p, label=f"PR-AUC = {test_metrics['pr_auc']:.4f}", linewidth=2)
    ax.set_xlabel('Recall'), ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve'), ax.legend(), ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_RESULT_GRAPHS}/pr_curve.png', dpi=100)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    top = feature_importance.head(20)
    ax.barh(range(len(top)), top['importance'].values)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Importance'), ax.set_title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_RESULT_GRAPHS}/feature_importance.png', dpi=100)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df_benign[df_benign['near_malignant'] == 0]['similarity_score'],
            bins=50, alpha=0.6, label='Not Near-Malignant', color='blue')
    ax.hist(df_benign[df_benign['near_malignant'] == 1]['similarity_score'],
            bins=50, alpha=0.6, label='Near-Malignant', color='red')
    ax.axvline(SIMILARITY_THRESHOLD, color='black', linestyle='--', linewidth=2,
               label=f'Threshold = {SIMILARITY_THRESHOLD}')
    ax.set_xlabel('Similarity Score'), ax.set_ylabel('Count')
    ax.set_title('Benign Similarity Score Distribution'), ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_RESULT_GRAPHS}/similarity_distribution.png', dpi=100)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.6,
            label='Actual Not Near-Mal', color='blue')
    ax.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.6,
            label='Actual Near-Mal', color='red')
    ax.set_xlabel('Model Probability'), ax.set_ylabel('Count')
    ax.set_title('Predicted Probability Distribution'), ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_RESULT_GRAPHS}/probability_distribution.png', dpi=100)
    plt.close()

    print("\n" + "=" * 80)
    print("V5 PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\n  Model:    {OUTPUT_RESULTS}/{MODEL_NAME}")
    print(f"  Report:   {OUTPUT_REPORTS}/v5_report.txt")
    print(f"  Graphs:   {OUTPUT_GRAPHS}/ ({len(FEATURE_RANGES)*2} feature graphs)")
    print(f"  Results:  {OUTPUT_RESULT_GRAPHS}/ (6 model graphs)")
    print(f"  Data:     {OUTPUT_DATASETS}/")


if __name__ == "__main__":
    main()
