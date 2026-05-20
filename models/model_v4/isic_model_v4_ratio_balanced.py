# =============================================================================
# ISIC 2024 — LightGBM V4: Ratio-Balanced Data Prep + Model (merged)
# =============================================================================
# Data preparation (SMOTE + undersampling) and model training are merged into
# a single script. Each ratio in RATIOS gets its own resampled training set
# and a separate output folder under v4_outputs/.
#
# The shared test set (20% benign, extracted before any resampling) is built
# once and reused across all ratio runs to allow fair comparison.
#
# Ratio balancing approach:
#   - Malignant: SMOTE synthetic oversampling
#   - Benign:    random undersampling to achieve target ratio
#   - Result:    ~300k-row datasets with controlled class imbalance
#
# Output folder structure (one per ratio):
#   v4_outputs/
#     v4_outputs_ratio{RATIO}_1/
#       reports/
#         v4_readable_report.txt
#         v4_global_metrics.json
#         v4_ratio_{RATIO}_fold_metrics.csv
#         v4_ratio_{RATIO}_feature_importance.csv
#         v4_ratio_{RATIO}_benign_risk_predictions.csv
#         v4_ratio_{RATIO}_recall_target_metrics.csv
#       graphs/
#         v4_oof_confusion_matrix.png
#         v4_oof_precision_recall_curve.png
#         v4_feature_importance.png
#
# Usage:
#   python isic_model_v4_ratio_balanced.py
#
# Install once:
#   pip install pandas numpy scikit-learn lightgbm matplotlib seaborn imbalanced-learn
# =============================================================================

import pandas as pd
import numpy as np
import os
import json
import datetime
import warnings
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.getcwd(), '.matplotlib-cache'))
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
)
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImblearnPipeline


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DATA = str((_HERE / '../../datasets/datasetv4_merged_cleaned_engineered.csv').resolve())
PROCESSED_DATA_DIR = str(_HERE / 'processed_data')

# Ratios to run (benign:malignant) — each ratio generates its own output folder
RATIOS = [5, 10, 15, 20, 25, 30, 500, 600, 700, 1000]

# Data prep settings
TEST_SIZE         = 0.20
TARGET_TOTAL_ROWS = 300_000
RANDOM_STATE      = 42

# Model settings
DPI           = 150
N_FOLDS       = 5
TARGET_RECALL = 0.50

CLASS_WEIGHT_MULTIPLIER = 0.4

CATEGORICAL_COLS = ['sex', 'anatom_site_general', 'tbp_lv_location_simple']

COLOR_BENIGN    = '#2E86AB'
COLOR_MALIGNANT = '#E84855'
COLOR_NEUTRAL   = '#F4A261'

# Registry file shared across all ratio runs
_V4_OUTPUTS = (_HERE / '../../outputs/v4_outputs').resolve()
_V4_OUTPUTS.mkdir(parents=True, exist_ok=True)
MODEL_REGISTRY_FILE = str(_V4_OUTPUTS / 'v4_ratio_model_registry.csv')


# =============================================================================
# SETUP
# =============================================================================

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', font='Arial')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'savefig.facecolor':'white',
    'font.family':      'sans-serif',
    'axes.titlesize':   13,
    'axes.labelsize':   11,
})


# =============================================================================
# HELPERS
# =============================================================================

def threshold_for_target_recall(y_true, probs, target_recall):
    prec_vals, rec_vals, thresholds = precision_recall_curve(y_true, probs)
    valid_idx = np.where(rec_vals[:-1] >= target_recall)[0]
    if len(valid_idx) == 0:
        return thresholds[np.argmax(rec_vals[:-1])]
    best_valid_idx = valid_idx[np.argmax(prec_vals[valid_idx])]
    return thresholds[best_valid_idx]


def save_graph(fig, filename, graph_dir):
    path = os.path.join(graph_dir, filename)
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filename}")


# =============================================================================
# STEP 0 — Load raw data and create shared test set (done once)
# =============================================================================

print("=" * 70)
print("ISIC 2024 — V4 LightGBM Ratio-Balanced (merged data prep + model)")
print("=" * 70)
print(f"\nRatios to run: {RATIOS}")

print("\nSTEP 0 — Loading data and building shared test set")
print("-" * 70)

df_full = pd.read_csv(INPUT_DATA)
print(f"  Full data: {df_full.shape[0]:,} rows x {df_full.shape[1]} columns")
print(f"    Malignant : {df_full['malignant'].sum():,}")
print(f"    Benign    : {(df_full['malignant'] == 0).sum():,}")
print(f"    Ratio     : {(df_full['malignant'] == 0).sum() / df_full['malignant'].sum():.1f}:1")

# All malignant → train; 20% benign → test
df_benign    = df_full[df_full['malignant'] == 0].reset_index(drop=True)
df_malignant = df_full[df_full['malignant'] == 1].reset_index(drop=True)

df_benign_train, df_test = train_test_split(
    df_benign, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
df_train_base = pd.concat([df_malignant, df_benign_train], ignore_index=True)

print(f"\n  Train base : {len(df_train_base):,} rows "
      f"(malignant={df_train_base['malignant'].sum():,}, "
      f"benign={(df_train_base['malignant'] == 0).sum():,})")
print(f"  Test (shared): {len(df_test):,} benign-only rows (20% of benign, no leakage)")

# Save shared test set (original form, before any encoding)
test_csv = os.path.join(PROCESSED_DATA_DIR, 'test_ratio_shared.csv')
df_test.to_csv(test_csv, index=False)
print(f"  ✓ Shared test set saved: {test_csv}")

# Prepare base feature matrices for SMOTE (requires numeric-only input)
X_base = df_train_base.drop(columns=['malignant']).copy()
y_base = df_train_base['malignant']

X_test_raw = df_test.drop(columns=['malignant']).copy()

ohe_cat_cols = X_base.select_dtypes(include=['object']).columns.tolist()
if ohe_cat_cols:
    X_base     = pd.get_dummies(X_base,     columns=ohe_cat_cols, drop_first=True)
    X_test_raw = pd.get_dummies(X_test_raw, columns=ohe_cat_cols, drop_first=True)
    for col in set(X_base.columns) - set(X_test_raw.columns):
        X_test_raw[col] = 0
    X_test_raw = X_test_raw[X_base.columns]

print(f"  Base feature columns after encoding: {X_base.shape[1]}")


# =============================================================================
# MAIN LOOP — iterate over each ratio
# =============================================================================

for RATIO in RATIOS:

    print("\n" + "=" * 70)
    print(f"RATIO {RATIO}:1  (benign:malignant)")
    print("=" * 70)

    # Per-ratio output dirs
    OUTPUT_FOLDER = str(_V4_OUTPUTS / f'v4_outputs_ratio{RATIO}_1')
    REPORT_DIR    = os.path.join(OUTPUT_FOLDER, 'reports')
    GRAPH_DIR     = os.path.join(OUTPUT_FOLDER, 'graphs')
    OUTPUT_PREFIX = f'v4_ratio_{RATIO}'
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(GRAPH_DIR,  exist_ok=True)

    # -------------------------------------------------------------------------
    # DATA PREP — SMOTE + undersampling for this ratio
    # -------------------------------------------------------------------------

    print(f"\nDATA PREP — Resampling to {RATIO}:1")
    print("-" * 70)

    n_malignant_target = int(TARGET_TOTAL_ROWS / (RATIO + 1))
    n_benign_target    = TARGET_TOTAL_ROWS - n_malignant_target

    resample_pipeline = ImblearnPipeline([
        ('smote', SMOTE(
            sampling_strategy='minority',
            random_state=RANDOM_STATE,
            k_neighbors=5,
        )),
        ('undersample', RandomUnderSampler(
            sampling_strategy={1: n_malignant_target, 0: n_benign_target},
            random_state=RANDOM_STATE,
        )),
    ])

    X_resampled, y_resampled = resample_pipeline.fit_resample(X_base, y_base)

    df_train = pd.DataFrame(X_resampled, columns=X_base.columns)
    df_train['malignant'] = y_resampled.values
    df_train = df_train.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    n_malignant_train = int((y_resampled == 1).sum())
    n_benign_train    = int((y_resampled == 0).sum())
    actual_ratio      = n_benign_train / max(1, n_malignant_train)

    train_csv = os.path.join(PROCESSED_DATA_DIR, f'train_ratio_{RATIO}_1.csv')
    df_train.to_csv(train_csv, index=False)

    print(f"  Train rows   : {len(df_train):,}")
    print(f"  Malignant    : {n_malignant_train:,}  (SMOTE synthetic oversampled)")
    print(f"  Benign       : {n_benign_train:,}  (undersampled)")
    print(f"  Actual ratio : {actual_ratio:.2f}:1")
    print(f"  ✓ Saved: {train_csv}")

    n_test = len(df_test)

    # -------------------------------------------------------------------------
    # STEP 2 — Prepare feature matrices
    # -------------------------------------------------------------------------

    print(f"\nSTEP 2 — Preparing feature matrices")
    print("-" * 70)

    DROP_COLS = ['malignant', 'risk_score']

    X_train = df_train.drop(columns=[c for c in DROP_COLS if c in df_train.columns])
    y_train = df_train['malignant'].values

    X_test = X_test_raw.copy()

    FEATURE_NAMES = list(X_train.columns)
    X_test = X_test.reindex(columns=FEATURE_NAMES, fill_value=0)

    print(f"  Training features : {X_train.shape[0]:,} rows x {X_train.shape[1]} cols")
    print(f"  Test features     : {X_test.shape[0]:,} rows x {X_test.shape[1]} cols")
    print(f"  Feature names match: {'YES ✓' if list(X_train.columns) == list(X_test.columns) else 'NO — ERROR'}")

    # -------------------------------------------------------------------------
    # STEP 3 — Feature engineering: interaction features
    # -------------------------------------------------------------------------

    print(f"\nSTEP 3 — Creating interaction features")
    print("-" * 70)

    interaction_features = {}

    if 'color_contrast_3d' in X_train.columns and 'clin_size_long_diam_mm' in X_train.columns:
        X_train['color_x_size'] = X_train['color_contrast_3d'] * X_train['clin_size_long_diam_mm']
        X_test['color_x_size']  = X_test['color_contrast_3d']  * X_test['clin_size_long_diam_mm']
        interaction_features['color_x_size'] = 'color_contrast_3d × clin_size'

    if 'chroma_contrast' in X_train.columns and 'tbp_lv_symm_2axis' in X_train.columns:
        X_train['chroma_x_symm'] = X_train['chroma_contrast'] * X_train['tbp_lv_symm_2axis']
        X_test['chroma_x_symm']  = X_test['chroma_contrast']  * X_test['tbp_lv_symm_2axis']
        interaction_features['chroma_x_symm'] = 'chroma_contrast × tbp_lv_symm_2axis'

    if 'color_contrast_3d' in X_train.columns and 'chroma_contrast' in X_train.columns:
        X_train['color_contrast_ratio'] = X_train['color_contrast_3d'] / (X_train['chroma_contrast'] + 1e-6)
        X_test['color_contrast_ratio']  = X_test['color_contrast_3d']  / (X_test['chroma_contrast']  + 1e-6)
        interaction_features['color_contrast_ratio'] = 'color_contrast_3d / chroma_contrast'

    if 'clin_size_long_diam_mm' in X_train.columns and 'tbp_lv_eccentricity' in X_train.columns:
        X_train['size_x_ecc'] = X_train['clin_size_long_diam_mm'] * X_train['tbp_lv_eccentricity']
        X_test['size_x_ecc']  = X_test['clin_size_long_diam_mm']  * X_test['tbp_lv_eccentricity']
        interaction_features['size_x_ecc'] = 'clin_size × tbp_lv_eccentricity'

    if 'log_area' in X_train.columns and 'tbp_lv_norm_border' in X_train.columns:
        X_train['area_x_border'] = X_train['log_area'] * X_train['tbp_lv_norm_border']
        X_test['area_x_border']  = X_test['log_area']  * X_test['tbp_lv_norm_border']
        interaction_features['area_x_border'] = 'log_area × tbp_lv_norm_border'

    FEATURE_NAMES = list(X_train.columns)
    print(f"  Created {len(interaction_features)} interaction features")
    for feat_name, formula in interaction_features.items():
        print(f"    • {feat_name:25} = {formula}")
    print(f"  Total features: {len(FEATURE_NAMES)}")

    # -------------------------------------------------------------------------
    # STEP 4 — LightGBM configuration
    # -------------------------------------------------------------------------

    scale_pos_weight           = n_benign_train / n_malignant_train
    effective_scale_pos_weight = scale_pos_weight * CLASS_WEIGHT_MULTIPLIER

    LGBM_PARAMS = {
        'n_estimators':      1000,
        'learning_rate':     0.03,
        'num_leaves':        15,
        'max_depth':         4,
        'min_child_samples': 75,
        'subsample':         0.8,
        'subsample_freq':    1,
        'colsample_bytree':  0.8,
        'reg_alpha':         0.5,
        'reg_lambda':        2.0,
        'random_state':      RANDOM_STATE,
        'n_jobs':            -1,
        'verbose':           -1,
    }
    if CLASS_WEIGHT_MULTIPLIER > 0:
        LGBM_PARAMS['scale_pos_weight'] = effective_scale_pos_weight

    print(f"\nSTEP 4 — Model configuration")
    print("-" * 70)
    print(f"  Algorithm          : LightGBM")
    print(f"  n_estimators       : {LGBM_PARAMS['n_estimators']}")
    print(f"  min_child_samples  : {LGBM_PARAMS['min_child_samples']}")
    if CLASS_WEIGHT_MULTIPLIER > 0:
        print(f"  scale_pos_weight   : {effective_scale_pos_weight:.1f}  "
              f"({CLASS_WEIGHT_MULTIPLIER:.2f}x base imbalance weight — ENABLED)")
    else:
        print("  scale_pos_weight   : disabled")
    print(f"  Validation         : StratifiedKFold (k={N_FOLDS})")
    print(f"  Threshold strategy : Optimal via precision-recall curve")
    print(f"  Screening target   : Recall >= {TARGET_RECALL:.0%}")
    print(f"  Early stopping     : 50 rounds")

    # -------------------------------------------------------------------------
    # STEP 5 — Cross-validated training
    # -------------------------------------------------------------------------

    print(f"\nSTEP 5 — Cross-Validation on Training Set")
    print("-" * 70)

    X_arr  = X_train.values
    cv     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(cv.split(X_arr, y_train))

    fold_results      = []
    cv_probs          = np.zeros(len(y_train))
    feature_imp_folds = np.zeros((N_FOLDS, len(FEATURE_NAMES)))
    best_models       = []
    best_thresholds   = []

    print(f"  {'Fold':<6} {'Train':>8} {'Val':>8} {'Mal/Val':>8} "
          f"{'Threshold':>10} {'F1':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'PR-AUC':>8}")
    print(f"  {'-'*85}")

    for fold, (train_idx, val_idx) in enumerate(splits):
        X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        n_mal_val   = y_val.sum()

        model = LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                early_stopping(stopping_rounds=50, verbose=False),
                log_evaluation(period=-1),
            ],
        )

        probs = model.predict_proba(X_val)[:, 1]
        cv_probs[val_idx] = probs

        if y_val.sum() > 0:
            prec_vals, rec_vals, thresholds = precision_recall_curve(y_val, probs)
            f1_per_thr = 2 * prec_vals[:-1] * rec_vals[:-1] / (
                prec_vals[:-1] + rec_vals[:-1] + 1e-9
            )
            best_thr = thresholds[np.argmax(f1_per_thr)]
        else:
            best_thr = 0.5

        best_thresholds.append(best_thr)
        preds = (probs >= best_thr).astype(int)

        f1     = f1_score(y_val, preds, zero_division=0)
        acc    = accuracy_score(y_val, preds)
        prec   = precision_score(y_val, preds, zero_division=0)
        rec    = recall_score(y_val, preds, zero_division=0)
        try:
            pr_auc = average_precision_score(y_val, probs)
        except Exception:
            pr_auc = float('nan')

        print(f"  {fold+1:<6} {len(X_tr):>8,} {len(X_val):>8,} "
              f"{n_mal_val:>8} {best_thr:>10.4f} "
              f"{f1:>8.4f} {acc:>8.4f} {prec:>8.4f} {rec:>8.4f} {pr_auc:>8.4f}")

        fold_results.append({
            'fold': fold+1, 'n_train': len(X_tr), 'n_val': len(X_val),
            'n_mal_val': int(n_mal_val), 'threshold': round(best_thr, 4),
            'f1': round(f1, 4), 'accuracy': round(acc, 4),
            'precision': round(prec, 4), 'recall': round(rec, 4),
            'pr_auc': round(pr_auc, 4), 'best_iter': model.best_iteration_,
        })

        feature_imp_folds[fold] = model.feature_importances_
        best_models.append(model)

    df_folds = pd.DataFrame(fold_results)
    means = df_folds[['f1', 'accuracy', 'precision', 'recall', 'pr_auc']].mean()
    stds  = df_folds[['f1', 'accuracy', 'precision', 'recall', 'pr_auc']].std()

    print(f"\n  {'MEAN':<6} {'':>8} {'':>8} {'':>8} {'':>10} "
          f"{means['f1']:>8.4f} {means['accuracy']:>8.4f} "
          f"{means['precision']:>8.4f} {means['recall']:>8.4f} {means['pr_auc']:>8.4f}")
    print(f"  {'STD':<6} {'':>8} {'':>8} {'':>8} {'':>10} "
          f"{stds['f1']:>8.4f} {stds['accuracy']:>8.4f} "
          f"{stds['precision']:>8.4f} {stds['recall']:>8.4f} {stds['pr_auc']:>8.4f}")

    df_folds.to_csv(f'{REPORT_DIR}/{OUTPUT_PREFIX}_fold_metrics.csv', index=False)
    print(f"\n  Saved: {REPORT_DIR}/{OUTPUT_PREFIX}_fold_metrics.csv")

    # Overall CV metrics
    prec_overall, rec_overall, thresholds_overall = precision_recall_curve(y_train, cv_probs)
    f1_overall_by_threshold = 2 * prec_overall[:-1] * rec_overall[:-1] / (
        prec_overall[:-1] + rec_overall[:-1] + 1e-9
    )
    overall_threshold = thresholds_overall[np.argmax(f1_overall_by_threshold)]
    cv_preds_final    = (cv_probs >= overall_threshold).astype(int)

    recall_threshold = threshold_for_target_recall(y_train, cv_probs, TARGET_RECALL)
    cv_preds_recall  = (cv_probs >= recall_threshold).astype(int)

    cv_f1   = f1_score(y_train, cv_preds_final, zero_division=0)
    cv_acc  = accuracy_score(y_train, cv_preds_final)
    cv_prec = precision_score(y_train, cv_preds_final, zero_division=0)
    cv_rec  = recall_score(y_train, cv_preds_final, zero_division=0)

    recall_f1   = f1_score(y_train, cv_preds_recall, zero_division=0)
    recall_acc  = accuracy_score(y_train, cv_preds_recall)
    recall_prec = precision_score(y_train, cv_preds_recall, zero_division=0)
    recall_rec  = recall_score(y_train, cv_preds_recall, zero_division=0)

    try:
        cv_pr = average_precision_score(y_train, cv_probs)
    except Exception:
        cv_pr = float('nan')

    cm_cv = confusion_matrix(y_train, cv_preds_final)
    tn, fp, fn, tp = cm_cv.ravel()

    cm_recall = confusion_matrix(y_train, cv_preds_recall)
    recall_tn, recall_fp, recall_fn, recall_tp = cm_recall.ravel()

    target_rows = []
    for target in [0.30, 0.50, 0.70, 0.90]:
        target_thr   = threshold_for_target_recall(y_train, cv_probs, target)
        target_preds = (cv_probs >= target_thr).astype(int)
        target_cm    = confusion_matrix(y_train, target_preds)
        target_tn, target_fp, target_fn, target_tp = target_cm.ravel()
        target_rows.append({
            'target_recall':        target,
            'threshold':            target_thr,
            'f1':                   f1_score(y_train, target_preds, zero_division=0),
            'accuracy':             accuracy_score(y_train, target_preds),
            'precision':            precision_score(y_train, target_preds, zero_division=0),
            'recall':               recall_score(y_train, target_preds, zero_division=0),
            'malignant_caught':     target_tp,
            'benign_flagged':       target_fp,
            'benign_flagged_pct':   target_fp / n_benign_train * 100,
        })

    df_recall_targets = pd.DataFrame(target_rows)
    df_recall_targets.to_csv(f'{REPORT_DIR}/{OUTPUT_PREFIX}_recall_target_metrics.csv', index=False)

    print(f"\n  CV Overall (global F1-optimal threshold={overall_threshold:.4f}):")
    print(f"    F1={cv_f1:.4f}  Acc={cv_acc:.4f}  Prec={cv_prec:.4f}  Rec={cv_rec:.4f}")
    print(f"    Malignant caught: {tp}/{n_malignant_train} ({tp/n_malignant_train*100:.1f}%)")

    print(f"\n  CV Screening Mode (target recall={TARGET_RECALL:.0%}, threshold={recall_threshold:.4f}):")
    print(f"    F1={recall_f1:.4f}  Acc={recall_acc:.4f}  "
          f"Prec={recall_prec:.4f}  Rec={recall_rec:.4f}")
    print(f"    Malignant caught: {recall_tp}/{n_malignant_train} "
          f"({recall_tp/n_malignant_train*100:.1f}%)")
    print(f"    Benign flagged: {recall_fp}/{n_benign_train} "
          f"({recall_fp/n_benign_train*100:.1f}%)")

    # -------------------------------------------------------------------------
    # STEP 6 — Feature importance
    # -------------------------------------------------------------------------

    print(f"\nSTEP 6 — Feature Importance")
    print("-" * 70)

    mean_imp = feature_imp_folds.mean(axis=0)
    std_imp  = feature_imp_folds.std(axis=0)

    df_importance = pd.DataFrame({
        'feature':    FEATURE_NAMES,
        'importance': mean_imp,
        'std':        std_imp,
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    df_importance['rank'] = range(1, len(df_importance) + 1)

    df_importance.to_csv(f'{REPORT_DIR}/{OUTPUT_PREFIX}_feature_importance.csv', index=False)

    print(f"\n  {'Rank':<6} {'Feature':<35} {'Importance':>12} {'Std':>8}")
    print(f"  {'-'*65}")
    for _, row in df_importance.iterrows():
        bar = '█' * int(row['importance'] / df_importance['importance'].max() * 20)
        print(f"  {int(row['rank']):<6} {row['feature']:<35} "
              f"{row['importance']:>12.1f} ±{row['std']:>6.1f}  {bar}")

    print(f"\n  Saved: {REPORT_DIR}/{OUTPUT_PREFIX}_feature_importance.csv")

    # -------------------------------------------------------------------------
    # STEP 7 — Predict on test set (unseen benign records)
    # -------------------------------------------------------------------------

    print(f"\nSTEP 7 — Predicting Malignancy Risk on Test Set")
    print("-" * 70)
    print(f"  Using: best model from fold {df_folds['f1'].idxmax()+1} "
          f"(highest CV F1 = {df_folds['f1'].max():.4f})")

    best_fold_idx = df_folds['f1'].idxmax()
    best_model    = best_models[best_fold_idx]

    test_probs = best_model.predict_proba(X_test.values)[:, 1]

    df_test_out = df_test.copy()
    df_test_out['lgbm_malignancy_prob'] = test_probs
    df_test_out['screening_flag'] = df_test_out['lgbm_malignancy_prob'] >= recall_threshold

    risk_cut_1 = max(float(recall_threshold), 1e-6)
    risk_cut_2 = max(float(overall_threshold), risk_cut_1 * 1.01)
    risk_cut_3 = min(max(risk_cut_2 * 2, risk_cut_2 + 1e-6), 1.0)
    risk_bins   = [0.00, risk_cut_1, risk_cut_2, risk_cut_3, 1.01]
    risk_labels = ['Low', 'Elevated', 'High', 'Very High']

    df_test_out['risk_tier'] = pd.cut(
        df_test_out['lgbm_malignancy_prob'],
        bins=risk_bins,
        labels=risk_labels,
        right=False,
    )

    df_test_out = df_test_out.sort_values('lgbm_malignancy_prob', ascending=False).reset_index(drop=True)
    df_test_out.index += 1
    df_test_out.index.name = 'risk_rank'

    print(f"  Probability stats:")
    print(f"    Mean   : {test_probs.mean():.4f}")
    print(f"    Median : {np.median(test_probs):.4f}")
    print(f"    Max    : {test_probs.max():.4f}")
    print(f"    Min    : {test_probs.min():.4f}")

    print(f"\n  Risk tier distribution:")
    tier_counts = df_test_out['risk_tier'].value_counts().sort_index()
    for tier, count in tier_counts.items():
        pct = count / len(df_test_out) * 100
        bar = '█' * int(pct / 2)
        print(f"    {tier:<12} : {count:>8,} ({pct:>5.1f}%)  {bar}")

    screening_count = int(df_test_out['screening_flag'].sum())
    screening_pct   = screening_count / len(df_test_out) * 100
    print(f"\n  Screening flagged: {screening_count:,}/{len(df_test_out):,} ({screening_pct:.1f}%)")

    df_test_out.to_csv(f'{REPORT_DIR}/{OUTPUT_PREFIX}_benign_risk_predictions.csv')
    print(f"\n  Saved: {REPORT_DIR}/{OUTPUT_PREFIX}_benign_risk_predictions.csv")

    # -------------------------------------------------------------------------
    # STEP 8 — Generate graphs
    # -------------------------------------------------------------------------

    print(f"\nSTEP 8 — Generating graphs")
    print("-" * 70)

    # Graph 1: CV confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm_cv,
                           display_labels=['Benign', 'Malignant']).plot(
        ax=ax, colorbar=False, cmap='Blues', values_format='d'
    )
    ax.set_title(f'Confusion Matrix — Out-of-Fold ({RATIO}:1 Ratio)\n(LightGBM)', fontweight='bold')
    plt.tight_layout()
    save_graph(fig, 'v4_oof_confusion_matrix.png', GRAPH_DIR)

    # Graph 2: PR curve
    prec_cv, rec_cv, _ = precision_recall_curve(y_train, cv_probs)
    baseline = n_malignant_train / len(y_train)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec_cv, prec_cv, color=COLOR_MALIGNANT, linewidth=2,
            label=f'LightGBM OOF  (PR-AUC={cv_pr:.4f})')
    ax.axhline(baseline, color='gray', linestyle='--', linewidth=1.5,
               label=f'Random baseline ({baseline:.4f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve — Out-of-Fold ({RATIO}:1 Ratio)', fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    save_graph(fig, 'v4_oof_precision_recall_curve.png', GRAPH_DIR)

    # Graph 3: Feature importance
    top_n  = min(20, len(df_importance))
    df_top = df_importance.head(top_n)
    colors_imp = [COLOR_MALIGNANT if i < 5 else
                  COLOR_NEUTRAL   if i < 10 else
                  COLOR_BENIGN for i in range(top_n)]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(df_top['feature'][::-1], df_top['importance'][::-1],
            xerr=df_top['std'][::-1],
            color=colors_imp[::-1], edgecolor='white', height=0.65,
            error_kw=dict(capsize=4, linewidth=1.2, color='black'))
    ax.set_xlabel('Feature Importance (averaged across 5 folds)')
    ax.set_title(f'LightGBM Feature Importance — {RATIO}:1 Ratio (Top {top_n})', fontweight='bold')
    ax.legend(handles=[
        mpatches.Patch(color=COLOR_MALIGNANT, label='Top 5'),
        mpatches.Patch(color=COLOR_NEUTRAL,   label='Rank 6–10'),
        mpatches.Patch(color=COLOR_BENIGN,    label='Rank 11+'),
    ], fontsize=9)
    plt.tight_layout()
    save_graph(fig, 'v4_feature_importance.png', GRAPH_DIR)

    # -------------------------------------------------------------------------
    # STEP 9 — Save readable report and global metrics JSON
    # -------------------------------------------------------------------------

    print(f"\nSTEP 9 — Saving reports")
    print("-" * 70)

    report_lines = [
        "=" * 80,
        "ISIC 2024 — V4 LightGBM Model with Benign:Malignant Ratio-Balanced Data",
        "=" * 80,
        "",
        "DATASET",
        f"  Total train rows   : {len(df_train):,}",
        f"  Malignant          : {n_malignant_train:,}  (SMOTE synthetic oversampled)",
        f"  Benign             : {n_benign_train:,}  (randomly undersampled)",
        f"  Benign:Malignant   : {RATIO}:1",
        f"  Test set (benign)  : {n_test:,}",
        "",
        "MODEL",
        f"  Algorithm          : LightGBM",
        f"  CV design          : StratifiedKFold (k={N_FOLDS})",
        f"  n_estimators       : {LGBM_PARAMS['n_estimators']}",
        f"  min_child_samples  : {LGBM_PARAMS['min_child_samples']}",
        f"  scale_pos_weight   : {effective_scale_pos_weight:.2f}  (imbalance weight)",
        f"  Interaction feats  : {len(interaction_features)}",
        f"  Total features     : {len(FEATURE_NAMES)}",
        "",
        "THRESHOLD POLICY",
        f"  Optimal threshold  : {overall_threshold:.6f}  (F1-optimal on CV)",
        f"  Screening threshold: {recall_threshold:.6f}  (target recall {TARGET_RECALL:.0%})",
        "",
        "OUT-OF-FOLD METRICS (Cross-Validation on Training Set)",
        f"  PR-AUC             : {cv_pr:.6f}",
        f"  ROC-AUC            : {roc_auc_score(y_train, cv_probs):.6f}",
        f"  F1                 : {cv_f1:.6f}",
        f"  Precision          : {cv_prec:.6f}",
        f"  Recall             : {cv_rec:.6f}",
        f"  Confusion matrix   : TP={tp}, FP={fp}, FN={fn}, TN={tn}",
        "",
        "MEAN ± STD ACROSS FOLDS",
        f"  {'metric':<12} {'mean':>10} {'std':>10}",
        f"  {'-'*32}",
        f"  {'f1':<12} {means['f1']:>10.6f} {stds['f1']:>10.6f}",
        f"  {'accuracy':<12} {means['accuracy']:>10.6f} {stds['accuracy']:>10.6f}",
        f"  {'precision':<12} {means['precision']:>10.6f} {stds['precision']:>10.6f}",
        f"  {'recall':<12} {means['recall']:>10.6f} {stds['recall']:>10.6f}",
        f"  {'pr_auc':<12} {means['pr_auc']:>10.6f} {stds['pr_auc']:>10.6f}",
        "",
        "TEST SET PREDICTIONS (Unseen Benign Records)",
        f"  Records scored     : {n_test:,}",
        f"  Prob mean          : {test_probs.mean():.6f}",
        f"  Prob median        : {np.median(test_probs):.6f}",
        f"  Prob max           : {test_probs.max():.6f}",
        f"  Screening flagged  : {screening_count:,}  ({screening_pct:.1f}%)",
        "",
        "TOP 20 FEATURES",
        f"  {'rank':<6} {'feature':<35} {'importance':>12}",
        f"  {'-'*55}",
    ]

    for _, row in df_importance.head(20).iterrows():
        report_lines.append(
            f"  {int(row['rank']):<6} {row['feature']:<35} {row['importance']:>12.1f}"
        )

    report_lines += [
        "",
        "OUTPUT FILES",
        f"  {OUTPUT_FOLDER}/reports/v4_readable_report.txt",
        f"  {OUTPUT_FOLDER}/reports/v4_global_metrics.json",
        f"  {OUTPUT_FOLDER}/reports/{OUTPUT_PREFIX}_fold_metrics.csv",
        f"  {OUTPUT_FOLDER}/reports/{OUTPUT_PREFIX}_feature_importance.csv",
        f"  {OUTPUT_FOLDER}/reports/{OUTPUT_PREFIX}_benign_risk_predictions.csv",
        f"  {OUTPUT_FOLDER}/graphs/v4_oof_confusion_matrix.png",
        f"  {OUTPUT_FOLDER}/graphs/v4_oof_precision_recall_curve.png",
        f"  {OUTPUT_FOLDER}/graphs/v4_feature_importance.png",
        "",
        "=" * 80,
    ]

    report_text = "\n".join(report_lines)
    print(report_text)

    with open(f'{REPORT_DIR}/v4_readable_report.txt', 'w') as f:
        f.write(report_text)
    print(f"\n  Saved: {REPORT_DIR}/v4_readable_report.txt")

    global_metrics = {
        "n":            int(len(df_train)),
        "n_malignant":  int(n_malignant_train),
        "n_benign":     int(n_benign_train),
        "ratio":        int(RATIO),
        "threshold":    float(overall_threshold),
        "pr_auc":       float(cv_pr),
        "roc_auc":      float(roc_auc_score(y_train, cv_probs)),
        "f1":           float(cv_f1),
        "precision":    float(cv_prec),
        "recall":       float(cv_rec),
        "tp":           int(tp),
        "fp":           int(fp),
        "fn":           int(fn),
        "tn":           int(tn),
    }

    with open(f'{REPORT_DIR}/v4_global_metrics.json', 'w') as f:
        json.dump(global_metrics, f, indent=2)
    print(f"  Saved: {REPORT_DIR}/v4_global_metrics.json")

    # -------------------------------------------------------------------------
    # STEP 10 — Record to shared model registry
    # -------------------------------------------------------------------------

    print(f"\nSTEP 10 — Recording model to registry")
    print("-" * 70)

    model_record = {
        'timestamp':                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ratio':                        RATIO,
        'min_child_samples':            LGBM_PARAMS['min_child_samples'],
        'scale_pos_weight':             effective_scale_pos_weight,
        'n_estimators':                 LGBM_PARAMS['n_estimators'],
        'num_features_total':           len(FEATURE_NAMES),
        'num_interaction_features':     len(interaction_features),
        'cv_f1':                        round(cv_f1, 4),
        'cv_accuracy':                  round(cv_acc, 4),
        'cv_precision':                 round(cv_prec, 4),
        'cv_recall':                    round(cv_rec, 4),
        'cv_pr_auc':                    round(cv_pr, 4),
        'cv_malignant_caught':          int(tp),
        'cv_malignant_caught_pct':      round(tp / n_malignant_train * 100, 1),
        'screening_f1':                 round(recall_f1, 4),
        'screening_precision':          round(recall_prec, 4),
        'screening_recall':             round(recall_rec, 4),
        'screening_benign_flagged':     int(recall_fp),
        'screening_benign_flagged_pct': round(recall_fp / n_benign_train * 100, 1),
        'test_prob_mean':               round(test_probs.mean(), 4),
        'test_prob_median':             round(np.median(test_probs), 4),
        'test_prob_max':                round(test_probs.max(), 4),
        'test_screening_flagged':       screening_count,
        'test_screening_flagged_pct':   round(screening_pct, 1),
    }

    if not os.path.exists(MODEL_REGISTRY_FILE):
        df_registry = pd.DataFrame([model_record])
    else:
        df_registry = pd.read_csv(MODEL_REGISTRY_FILE)
        df_registry = pd.concat([df_registry, pd.DataFrame([model_record])], ignore_index=True)

    df_registry.to_csv(MODEL_REGISTRY_FILE, index=False)
    print(f"  Registry saved: {MODEL_REGISTRY_FILE}  ({len(df_registry)} model(s) recorded)")

    # -------------------------------------------------------------------------
    # RATIO DONE
    # -------------------------------------------------------------------------

    print(f"\n{'='*70}")
    print(f"RATIO {RATIO}:1 COMPLETE")
    print(f"  Output: {OUTPUT_FOLDER}")
    print(f"  CV F1={cv_f1:.4f}  Precision={cv_prec:.4f}  Recall={cv_rec:.4f}  PR-AUC={cv_pr:.4f}")
    print(f"{'='*70}")


# =============================================================================
# ALL RATIOS DONE
# =============================================================================

print("\n" + "=" * 70)
print("ALL DONE")
print("=" * 70)
print(f"\nCompleted {len(RATIOS)} ratio run(s): {RATIOS}")
print(f"\nOutput structure:")
for ratio in RATIOS:
    print(f"  {_V4_OUTPUTS}/v4_outputs_ratio{ratio}_1/")
print(f"\nShared model registry: {MODEL_REGISTRY_FILE}")
print(f"Shared test set      : {os.path.join(PROCESSED_DATA_DIR, 'test_ratio_shared.csv')}")
