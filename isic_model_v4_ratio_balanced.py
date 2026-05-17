# =============================================================================
# ISIC 2024 — LightGBM Model with Stratified Split + Ratio-Balanced Data (v4 CORRECTED)
# =============================================================================
# PROPER WORKFLOW (prevents data leakage):
#   1. Stratified split on original data (train/test first)
#   2. SMOTE + undersampling applied ONLY to training data
#   3. Test set contains both benign AND malignant (honest evaluation)
#
# Data preparation by isic_data_prep_ratio_v2_stratified.py:
#   - Original 393 malignant, 320.5k benign
#   - Split: 299 malignant + 243.6k benign (train) | 94 malignant + 76.9k benign (test)
#   - For each ratio, resample training data with SMOTE + undersampling
#   - Keep test set completely unchanged (real data only)
#
# Ratio balancing approach:
#   - Malignant: 299 original → SMOTE synthetic oversampling
#   - Benign:    random undersampling to achieve target ratio
#   - Result: training sets with controlled class imbalance for comparison
#
# Evaluation is now HONEST: model tested on real, unseen malignant cases
# (not just benign records as in previous versions)
#
# Output folder structure:
#   v4_outputs_ratio_balanced_[ratio]/
#     reports/
#       v4_readable_report.txt                      — human-readable summary
#       v4_global_metrics.json                      — key metrics in JSON format
#       v4_ratio_[ratio]_fold_metrics.csv           — per-fold CV metrics
#       v4_ratio_[ratio]_feature_importance.csv     — feature importance ranked
#       v4_ratio_[ratio]_benign_risk_predictions.csv — predictions on test set
#       v4_ratio_[ratio]_recall_target_metrics.csv  — recall target tradeoffs
#     graphs/
#       v4_oof_confusion_matrix.png                 — out-of-fold confusion matrix
#       v4_oof_precision_recall_curve.png           — out-of-fold PR curve
#       v4_feature_importance.png                   — feature importance ranking
#
# Usage:
#   python isic_model_v4_ratio_balanced.py                  # default ratio 5:1
#   RATIO=10 python isic_model_v4_ratio_balanced.py         # try ratio 10:1
#   RATIO=20 python isic_model_v4_ratio_balanced.py         # try ratio 20:1
#
# Install once:
#   !pip install pandas numpy scikit-learn lightgbm matplotlib seaborn
# =============================================================================

import pandas as pd
import numpy as np
import os
import json
import datetime
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.getcwd(), '.matplotlib-cache'))
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.ensemble import GradientBoostingClassifier


# =============================================================================
# CONFIGURATION
# =============================================================================

# Ratio to use (5, 10, 15, 20, 25, or 30)
RATIO = int(os.environ.get('RATIO', '5'))

# NEW: Use stratified data with proper train/test split
TRAIN_FILE = f'ratio_data_v2/train_ratio_{RATIO}_1.csv'  # Resampled training data only
TEST_FILE  = 'ratio_data_v2/test_stratified.csv'          # Shared test set (both classes)

DPI          = 150
N_FOLDS      = 5
RANDOM_STATE = 42
TARGET_RECALL = 0.50

COLOR_BENIGN    = '#2E86AB'
COLOR_MALIGNANT = '#E84855'
COLOR_NEUTRAL   = '#F4A261'

CATEGORICAL_COLS = ['sex', 'anatom_site_general', 'tbp_lv_location_simple']

# Registry file at root level (shared across all ratio runs)
os.makedirs('reports', exist_ok=True)
MODEL_REGISTRY_FILE = 'reports/v4_ratio_model_registry.csv'

# Output folder structure
OUTPUT_FOLDER = f'v4_outputs_ratio_balanced_{RATIO}_1'
REPORT_DIR    = os.path.join(OUTPUT_FOLDER, 'reports')
GRAPH_DIR     = os.path.join(OUTPUT_FOLDER, 'graphs')

# Output prefix to distinguish from other v4 variants
OUTPUT_PREFIX = f'v4_ratio_{RATIO}'


# =============================================================================
# SETUP
# =============================================================================

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR,  exist_ok=True)

sns.set_theme(style='whitegrid', font='Arial')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'savefig.facecolor':'white',
    'font.family':      'sans-serif',
    'axes.titlesize':   13,
    'axes.labelsize':   11,
})

print("=" * 60)
print(f"ISIC 2024 — LightGBM Model (Ratio-Balanced v4)")
print(f"Using {RATIO}:1 benign:malignant ratio data")
print("=" * 60)


# =============================================================================
# STEP 1 — Load training and test files
# =============================================================================

print(f"\nLoading files...")

df_train = pd.read_csv(TRAIN_FILE)
df_test  = pd.read_csv(TEST_FILE)

n_benign_train    = (df_train['malignant'] == 0).sum()
n_malignant_train = (df_train['malignant'] == 1).sum()
n_benign_test     = (df_test['malignant'] == 0).sum()
n_malignant_test  = (df_test['malignant'] == 1).sum()
n_test            = len(df_test)

print(f"\n  TRAINING SET  ({TRAIN_FILE})")
print(f"    Total rows    : {len(df_train):,}")
print(f"    Malignant (1) : {n_malignant_train:,}  ← SMOTE synthetic oversampled")
print(f"    Benign    (0) : {n_benign_train:,}  ← randomly undersampled")
print(f"    Imbalance     : {n_benign_train / n_malignant_train:.1f}:1  (target ratio)")

print(f"\n  TEST SET (SHARED)  ({TEST_FILE})")
print(f"    Total rows    : {n_test:,}  ← stratified hold-out (original data, never resampled)")
print(f"    Malignant (1) : {n_malignant_test:,}  ← real, unseen malignant cases")
print(f"    Benign    (0) : {n_benign_test:,}  ← real, unseen benign cases")
print(f"    Test ratio    : {n_benign_test / n_malignant_test:.1f}:1  (similar to original)")


# =============================================================================
# STEP 2 — Encode categorical columns
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2 — Encoding categorical columns")
print("=" * 60)

encoders = {}

for col in CATEGORICAL_COLS:
    if col not in df_train.columns:
        continue
    le = LabelEncoder()

    df_train[col] = le.fit_transform(df_train[col].astype(str))
    encoders[col] = le

    if col in df_test.columns:
        test_vals = df_test[col].astype(str)
        known     = set(le.classes_)
        unknown   = set(test_vals.unique()) - known
        if unknown:
            print(f"  WARNING {col}: {len(unknown)} unseen categories in test → mapped to 'unknown'")
            test_vals = test_vals.apply(lambda x: x if x in known else le.classes_[0])
        df_test[col] = le.transform(test_vals)

    print(f"  Encoded: {col} → {len(le.classes_)} categories")


# =============================================================================
# STEP 3 — Prepare feature matrices
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3 — Preparing feature matrices")
print("=" * 60)

DROP_COLS = ['malignant', 'risk_score']

X_train = df_train.drop(columns=[c for c in DROP_COLS if c in df_train.columns])
y_train = df_train['malignant'].values

X_test  = df_test.drop(columns=[c for c in DROP_COLS if c in df_test.columns])

FEATURE_NAMES = list(X_train.columns)

# Align test columns to match training (same order, same columns)
X_test = X_test.reindex(columns=FEATURE_NAMES, fill_value=0)

print(f"  Training features : {X_train.shape[0]:,} rows x {X_train.shape[1]} cols")
print(f"  Test features     : {X_test.shape[0]:,}  rows x {X_test.shape[1]} cols")
print(f"  Feature names match: {'YES ✓' if list(X_train.columns) == list(X_test.columns) else 'NO — ERROR'}")
print(f"\n  Features ({len(FEATURE_NAMES)}):")
for i, col in enumerate(FEATURE_NAMES, 1):
    print(f"    {i:>2}. {col}")


# =============================================================================
# STEP 3.5 — Feature Engineering: Create Interaction Features
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3.5 — Creating interaction features")
print("=" * 60)

interaction_features = {}

# Interaction 1: Color contrast × Clinical size
if 'color_contrast_3d' in X_train.columns and 'clin_size_long_diam_mm' in X_train.columns:
    X_train['color_x_size'] = X_train['color_contrast_3d'] * X_train['clin_size_long_diam_mm']
    X_test['color_x_size'] = X_test['color_contrast_3d'] * X_test['clin_size_long_diam_mm']
    interaction_features['color_x_size'] = 'color_contrast_3d × clin_size'

# Interaction 2: Chroma contrast × Symmetry
if 'chroma_contrast' in X_train.columns and 'tbp_lv_symm_2axis' in X_train.columns:
    X_train['chroma_x_symm'] = X_train['chroma_contrast'] * X_train['tbp_lv_symm_2axis']
    X_test['chroma_x_symm'] = X_test['chroma_contrast'] * X_test['tbp_lv_symm_2axis']
    interaction_features['chroma_x_symm'] = 'chroma_contrast × tbp_lv_symm_2axis'

# Interaction 3: Color ratio
if 'color_contrast_3d' in X_train.columns and 'chroma_contrast' in X_train.columns:
    X_train['color_contrast_ratio'] = X_train['color_contrast_3d'] / (X_train['chroma_contrast'] + 1e-6)
    X_test['color_contrast_ratio'] = X_test['color_contrast_3d'] / (X_test['chroma_contrast'] + 1e-6)
    interaction_features['color_contrast_ratio'] = 'color_contrast_3d / chroma_contrast'

# Interaction 4: Size × Eccentricity
if 'clin_size_long_diam_mm' in X_train.columns and 'tbp_lv_eccentricity' in X_train.columns:
    X_train['size_x_ecc'] = X_train['clin_size_long_diam_mm'] * X_train['tbp_lv_eccentricity']
    X_test['size_x_ecc'] = X_test['clin_size_long_diam_mm'] * X_test['tbp_lv_eccentricity']
    interaction_features['size_x_ecc'] = 'clin_size × tbp_lv_eccentricity'

# Interaction 5: Area × Border norm
if 'log_area' in X_train.columns and 'tbp_lv_norm_border' in X_train.columns:
    X_train['area_x_border'] = X_train['log_area'] * X_train['tbp_lv_norm_border']
    X_test['area_x_border'] = X_test['log_area'] * X_test['tbp_lv_norm_border']
    interaction_features['area_x_border'] = 'log_area × tbp_lv_norm_border'

print(f"  Created {len(interaction_features)} interaction features:")
for feat_name, formula in interaction_features.items():
    print(f"    • {feat_name:25} = {formula}")

FEATURE_NAMES = list(X_train.columns)
print(f"\n  Total features after engineering: {len(FEATURE_NAMES)}")


# =============================================================================
# STEP 4 — LightGBM configuration
# =============================================================================

scale_pos_weight = n_benign_train / n_malignant_train
CLASS_WEIGHT_MULTIPLIER = 0.4
effective_scale_pos_weight = scale_pos_weight * CLASS_WEIGHT_MULTIPLIER

GB_PARAMS = {
    'n_estimators':      200,
    'learning_rate':     0.05,
    'max_depth':         5,
    'min_samples_split': 50,
    'min_samples_leaf':  20,
    'subsample':         0.8,
    'max_features':      'sqrt',
    'random_state':      RANDOM_STATE,
    'validation_fraction': 0.1,
    'n_iter_no_change':  20,
    'verbose':           0,
}

print("\n" + "=" * 60)
print("STEP 4 — Model configuration")
print("=" * 60)
print(f"  Algorithm          : Gradient Boosting (sklearn)")
print(f"  n_estimators       : {GB_PARAMS['n_estimators']}")
print(f"  min_samples_leaf   : {GB_PARAMS['min_samples_leaf']}")
print(f"  Class weight       : {CLASS_WEIGHT_MULTIPLIER:.2f}x base imbalance")
print(f"  Validation         : StratifiedKFold (k={N_FOLDS})")
print(f"  Threshold strategy : Optimal via precision-recall curve")
print(f"  Screening target   : Recall >= {TARGET_RECALL:.0%}")
print(f"  Feature engineering: {len(interaction_features)} interaction features added")
print(f"  Early stopping     : 20 rounds")


# =============================================================================
# STEP 5 — Cross-validated training on ratio-balanced data
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5 — Cross-Validation on Training Set")
print("=" * 60)
print(f"  NOTE: Cross-validation uses only {TRAIN_FILE}")
print(f"        {TEST_FILE} is NOT touched until Step 7\n")

X_arr = X_train.values

cv     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
splits = list(cv.split(X_arr, y_train))

fold_results       = []
cv_probs           = np.zeros(len(y_train))
cv_preds           = np.zeros(len(y_train))
feature_imp_folds  = np.zeros((N_FOLDS, len(FEATURE_NAMES)))
best_models        = []
best_thresholds    = []

print(f"  {'Fold':<6} {'Train':>8} {'Val':>8} {'Mal/Val':>8} "
      f"{'Threshold':>10} {'F1':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'PR-AUC':>8}")
print(f"  {'-'*85}")

for fold, (train_idx, val_idx) in enumerate(splits):
    X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    n_mal_val = y_val.sum()

    # Apply class weight balancing
    class_weight = {0: 1.0, 1: effective_scale_pos_weight}

    model = GradientBoostingClassifier(**GB_PARAMS)
    model.fit(X_tr, y_tr, sample_weight=np.where(y_tr==1, class_weight[1], class_weight[0]))

    probs = model.predict_proba(X_val)[:, 1]
    cv_probs[val_idx] = probs

    # Optimal threshold via precision-recall curve
    if y_val.sum() > 0:
        prec_vals, rec_vals, thresholds = precision_recall_curve(y_val, probs)
        f1_per_thr = 2 * prec_vals[:-1] * rec_vals[:-1] / (
            prec_vals[:-1] + rec_vals[:-1] + 1e-9
        )
        best_idx = np.argmax(f1_per_thr)
        best_thr = thresholds[best_idx]
    else:
        best_thr = 0.5

    best_thresholds.append(best_thr)
    preds = (probs >= best_thr).astype(int)
    cv_preds[val_idx] = preds

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
        'f1': round(f1,4), 'accuracy': round(acc,4),
        'precision': round(prec,4), 'recall': round(rec,4),
        'pr_auc': round(pr_auc,4), 'best_iter': model.n_estimators,
    })

    feature_imp_folds[fold] = model.feature_importances_
    best_models.append(model)

df_folds = pd.DataFrame(fold_results)
means    = df_folds[['f1','accuracy','precision','recall','pr_auc']].mean()
stds     = df_folds[['f1','accuracy','precision','recall','pr_auc']].std()

print(f"\n  {'MEAN':<6} {'':>8} {'':>8} {'':>8} {'':>10} "
      f"{means['f1']:>8.4f} {means['accuracy']:>8.4f} "
      f"{means['precision']:>8.4f} {means['recall']:>8.4f} {means['pr_auc']:>8.4f}")
print(f"  {'STD':<6} {'':>8} {'':>8} {'':>8} {'':>10} "
      f"{stds['f1']:>8.4f} {stds['accuracy']:>8.4f} "
      f"{stds['precision']:>8.4f} {stds['recall']:>8.4f} {stds['pr_auc']:>8.4f}")

df_folds.to_csv(f'{REPORT_DIR}/{OUTPUT_PREFIX}_fold_metrics.csv', index=False)
print(f"\n  Saved: {REPORT_DIR}/{OUTPUT_PREFIX}_fold_metrics.csv")

def threshold_for_target_recall(y_true, probs, target_recall):
    prec_vals, rec_vals, thresholds = precision_recall_curve(y_true, probs)
    valid_idx = np.where(rec_vals[:-1] >= target_recall)[0]

    if len(valid_idx) == 0:
        return thresholds[np.argmax(rec_vals[:-1])]

    best_valid_idx = valid_idx[np.argmax(prec_vals[valid_idx])]
    return thresholds[best_valid_idx]

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

cm_cv    = confusion_matrix(y_train, cv_preds_final)
tn, fp, fn, tp = cm_cv.ravel()

cm_recall = confusion_matrix(y_train, cv_preds_recall)
recall_tn, recall_fp, recall_fn, recall_tp = cm_recall.ravel()

target_rows = []
for target in [0.30, 0.50, 0.70, 0.90]:
    target_thr = threshold_for_target_recall(y_train, cv_probs, target)
    target_preds = (cv_probs >= target_thr).astype(int)
    target_cm = confusion_matrix(y_train, target_preds)
    target_tn, target_fp, target_fn, target_tp = target_cm.ravel()
    target_rows.append({
        'target_recall': target,
        'threshold': target_thr,
        'f1': f1_score(y_train, target_preds, zero_division=0),
        'accuracy': accuracy_score(y_train, target_preds),
        'precision': precision_score(y_train, target_preds, zero_division=0),
        'recall': recall_score(y_train, target_preds, zero_division=0),
        'malignant_caught': target_tp,
        'benign_flagged': target_fp,
        'benign_flagged_pct': target_fp / n_benign_train * 100,
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
print(f"    Saved recall tradeoffs: {REPORT_DIR}/{OUTPUT_PREFIX}_recall_target_metrics.csv")


# =============================================================================
# STEP 6 — Feature importance
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6 — Feature Importance")
print("=" * 60)

mean_imp = feature_imp_folds.mean(axis=0)
std_imp  = feature_imp_folds.std(axis=0)

df_importance = pd.DataFrame({
    'feature':    FEATURE_NAMES,
    'importance': mean_imp,
    'std':        std_imp,
}).sort_values('importance', ascending=False).reset_index(drop=True)
df_importance['rank'] = range(1, len(df_importance)+1)

df_importance.to_csv(f'{REPORT_DIR}/{OUTPUT_PREFIX}_feature_importance.csv', index=False)

print(f"\n  {'Rank':<6} {'Feature':<35} {'Importance':>12} {'Std':>8}")
print(f"  {'-'*65}")
for _, row in df_importance.iterrows():
    bar = '█' * int(row['importance'] / df_importance['importance'].max() * 20)
    print(f"  {int(row['rank']):<6} {row['feature']:<35} "
          f"{row['importance']:>12.1f} ±{row['std']:>6.1f}  {bar}")

print(f"\n  Saved: {REPORT_DIR}/{OUTPUT_PREFIX}_feature_importance.csv")


# =============================================================================
# STEP 7 — EVALUATE ON TEST SET (unseen benign + malignant records)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7 — Evaluating on Test Set (Both Classes)")
print("=" * 60)
print(f"  Test set: {n_test:,} records ({n_malignant_test} malignant + {n_benign_test} benign)")
print(f"  All test data is REAL, UNSEEN (never seen during training)")
print(f"  Using: best model from fold {df_folds['f1'].idxmax()+1} "
      f"(highest CV F1 = {df_folds['f1'].max():.4f})\n")

best_fold_idx = df_folds['f1'].idxmax()
best_model    = best_models[best_fold_idx]

test_probs = best_model.predict_proba(X_test.values)[:, 1]
test_preds = (test_probs >= overall_threshold).astype(int)

# Calculate test set metrics
y_test = df_test['malignant'].values

test_f1   = f1_score(y_test, test_preds, zero_division=0)
test_acc  = accuracy_score(y_test, test_preds)
test_prec = precision_score(y_test, test_preds, zero_division=0)
test_rec  = recall_score(y_test, test_preds, zero_division=0)
try:
    test_pr_auc = average_precision_score(y_test, test_probs)
except Exception:
    test_pr_auc = float('nan')

cm_test = confusion_matrix(y_test, test_preds)
test_tn, test_fp, test_fn, test_tp = cm_test.ravel()

print(f"  TEST SET METRICS (using optimal threshold={overall_threshold:.4f}):")
print(f"    F1        : {test_f1:.4f}")
print(f"    Accuracy  : {test_acc:.4f}")
print(f"    Precision : {test_prec:.4f}")
print(f"    Recall    : {test_rec:.4f}  (malignant detection rate)")
print(f"    PR-AUC    : {test_pr_auc:.4f}")
print(f"    Confusion matrix: TP={test_tp}, FP={test_fp}, FN={test_fn}, TN={test_tn}")
print(f"    Malignant caught: {test_tp}/{n_malignant_test} ({test_tp/n_malignant_test*100:.1f}%)")
print(f"    False positives: {test_fp}/{n_benign_test} ({test_fp/n_benign_test*100:.1f}%)")

df_test_out = pd.read_csv(TEST_FILE)
df_test_out['lgbm_malignancy_prob'] = test_probs
df_test_out['lgbm_pred'] = test_preds
df_test_out['screening_flag'] = df_test_out['lgbm_malignancy_prob'] >= recall_threshold

# Risk tier labels
risk_cut_1 = max(float(recall_threshold), 1e-6)
risk_cut_2 = max(float(overall_threshold), risk_cut_1 * 1.01)
risk_cut_3 = min(max(risk_cut_2 * 2, risk_cut_2 + 1e-6), 1.0)
risk_bins = [0.00, risk_cut_1, risk_cut_2, risk_cut_3, 1.01]
risk_labels = ['Low', 'Elevated', 'High', 'Very High']

df_test_out['risk_tier'] = pd.cut(
    df_test_out['lgbm_malignancy_prob'],
    bins=risk_bins,
    labels=risk_labels,
    right=False,
)

df_test_out = df_test_out.sort_values('lgbm_malignancy_prob', ascending=False)
df_test_out = df_test_out.reset_index(drop=True)
df_test_out.index += 1
df_test_out.index.name = 'risk_rank'

print(f"\n  Probability statistics for all test records:")
benign_probs = test_probs[y_test == 0]
malignant_probs = test_probs[y_test == 1]
print(f"    All records:")
print(f"      Mean   : {test_probs.mean():.4f}")
print(f"      Median : {np.median(test_probs):.4f}")
print(f"      Max    : {test_probs.max():.4f}")
print(f"    Benign records:")
print(f"      Mean   : {benign_probs.mean():.4f}")
print(f"      Median : {np.median(benign_probs):.4f}")
print(f"      Max    : {benign_probs.max():.4f}")
print(f"    Malignant records:")
print(f"      Mean   : {malignant_probs.mean():.4f}")
print(f"      Median : {np.median(malignant_probs):.4f}")
print(f"      Max    : {malignant_probs.max():.4f}")

print(f"\n  Risk tier distribution (on full test set):")
tier_counts = df_test_out['risk_tier'].value_counts().sort_index()
for tier, count in tier_counts.items():
    pct = count / len(df_test_out) * 100
    bar = '█' * int(pct / 2)
    print(f"    {tier:<12} : {count:>8,} ({pct:>5.1f}%)  {bar}")

screening_count = int(df_test_out['screening_flag'].sum())
screening_pct = screening_count / len(df_test_out) * 100
screening_mal = int(df_test_out[df_test_out['malignant']==1]['screening_flag'].sum())
print(f"\n  Screening mode flags (threshold={recall_threshold:.4f}):")
print(f"    Total flagged  : {screening_count:,}/{len(df_test_out):,} ({screening_pct:.1f}%)")
print(f"    Malignant flagged: {screening_mal:,}/{n_malignant_test:,} ({screening_mal/n_malignant_test*100:.1f}%)")

print(f"\n  Top 20 highest-risk records (mixed benign + malignant):")
display_cols = ['malignant', 'lgbm_malignancy_prob', 'lgbm_pred', 'screening_flag',
                'age_approx', 'anatom_site_general', 'clin_size_long_diam_mm',
                'tbp_lv_nevi_confidence']
display_cols = [c for c in display_cols if c in df_test_out.columns]
print(df_test_out[display_cols].head(20).to_string())

df_test_out.to_csv(f'{REPORT_DIR}/{OUTPUT_PREFIX}_test_predictions.csv')
print(f"\n  Saved: {REPORT_DIR}/{OUTPUT_PREFIX}_test_predictions.csv")


# =============================================================================
# STEP 8 — Generate core graphs (matching v3 format)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8 — Generating core graphs")
print("=" * 60)

saved_graphs = []

def save_graph(fig, filename):
    path = os.path.join(GRAPH_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    saved_graphs.append(filename)
    print(f"  Saved: {filename}")


# ── Graph 1: CV confusion matrix ────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_cv,
                               display_labels=['Benign', 'Malignant'])
disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='d')
ax.set_title(f'Confusion Matrix — Out-of-Fold ({RATIO}:1 Ratio)\n(LightGBM)', fontweight='bold')
plt.tight_layout()
save_graph(fig, f'v4_oof_confusion_matrix.png')


# ── Graph 2: PR curve ───────────────────────────────────────────────────────

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
save_graph(fig, 'v4_oof_precision_recall_curve.png')


# ── Graph 3: Feature importance ──────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 7))
top_n  = min(20, len(df_importance))
df_top = df_importance.head(top_n)
colors_imp = [COLOR_MALIGNANT if i < 5 else
              COLOR_NEUTRAL   if i < 10 else
              COLOR_BENIGN for i in range(top_n)]

ax.barh(df_top['feature'][::-1], df_top['importance'][::-1],
        xerr=df_top['std'][::-1],
        color=colors_imp[::-1], edgecolor='white', height=0.65,
        error_kw=dict(capsize=4, linewidth=1.2, color='black'))
ax.set_xlabel('Feature Importance (averaged across 5 folds)')
ax.set_title(f'LightGBM Feature Importance — {RATIO}:1 Ratio (Top {top_n})', fontweight='bold')
top5_patch = mpatches.Patch(color=COLOR_MALIGNANT, label='Top 5')
mid_patch  = mpatches.Patch(color=COLOR_NEUTRAL,   label='Rank 6–10')
rest_patch = mpatches.Patch(color=COLOR_BENIGN,     label='Rank 11+')
ax.legend(handles=[top5_patch, mid_patch, rest_patch], fontsize=9)
plt.tight_layout()
save_graph(fig, 'v4_feature_importance.png')


# =============================================================================
# STEP 9 — Save readable report and global metrics JSON
# =============================================================================

print("\n" + "=" * 60)
print("STEP 9 — Saving reports")
print("=" * 60)

# Create readable report
report_lines = [
    "=" * 80,
    "ISIC 2024 — V4 LightGBM Model with Benign:Malignant Ratio-Balanced Data",
    "=" * 80,
    "",
    "DATASET",
    f"  Total rows         : {len(df_train):,}",
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
    f"  Interaction features : {len(interaction_features)} engineering features",
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
    "TEST SET EVALUATION (Unseen Benign + Malignant Records)",
    f"  Test set size      : {n_test:,}  ({n_malignant_test} malignant, {n_benign_test} benign)",
    f"  Test F1            : {test_f1:.6f}",
    f"  Test Accuracy      : {test_acc:.6f}",
    f"  Test Precision     : {test_prec:.6f}",
    f"  Test Recall        : {test_rec:.6f}  (malignant detection rate)",
    f"  Test PR-AUC        : {test_pr_auc:.6f}",
    f"  Confusion matrix   : TP={test_tp}, FP={test_fp}, FN={test_fn}, TN={test_tn}",
    f"  Malignant caught   : {test_tp}/{n_malignant_test}  ({test_tp/n_malignant_test*100:.1f}%)",
    f"  False positives    : {test_fp}/{n_benign_test}  ({test_fp/n_benign_test*100:.1f}%)",
    "",
    "TEST SET PROBABILITY STATISTICS",
    f"  Prob mean (all)    : {test_probs.mean():.6f}",
    f"  Prob mean (benign) : {benign_probs.mean():.6f}",
    f"  Prob mean (malig)  : {malignant_probs.mean():.6f}",
    f"  Prob max           : {test_probs.max():.6f}",
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

# Save global metrics as JSON
global_metrics = {
    "n": int(len(df_train)),
    "n_malignant": int(n_malignant_train),
    "n_benign": int(n_benign_train),
    "ratio": int(RATIO),
    "threshold": float(overall_threshold),
    "pr_auc": float(cv_pr),
    "roc_auc": float(roc_auc_score(y_train, cv_probs)),
    "f1": float(cv_f1),
    "precision": float(cv_prec),
    "recall": float(cv_rec),
    "tp": int(tp),
    "fp": int(fp),
    "fn": int(fn),
    "tn": int(tn),
}

with open(f'{REPORT_DIR}/v4_global_metrics.json', 'w') as f:
    json.dump(global_metrics, f, indent=2)

print(f"  Saved: {REPORT_DIR}/v4_global_metrics.json")


# =============================================================================
# STEP 10 — Record Model to Registry
# =============================================================================

print("\n" + "=" * 60)
print("STEP 10 — Recording model to registry")
print("=" * 60)

model_record = {
    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'ratio': RATIO,
    'min_child_samples': LGBM_PARAMS['min_child_samples'],
    'scale_pos_weight': effective_scale_pos_weight,
    'n_estimators': LGBM_PARAMS['n_estimators'],
    'num_features_total': len(FEATURE_NAMES),
    'num_interaction_features': len(interaction_features),
    'cv_f1': round(cv_f1, 4),
    'cv_accuracy': round(cv_acc, 4),
    'cv_precision': round(cv_prec, 4),
    'cv_recall': round(cv_rec, 4),
    'cv_pr_auc': round(cv_pr, 4),
    'cv_malignant_caught': int(tp),
    'cv_malignant_caught_pct': round(tp/n_malignant_train*100, 1),
    'screening_f1': round(recall_f1, 4),
    'screening_precision': round(recall_prec, 4),
    'screening_recall': round(recall_rec, 4),
    'screening_benign_flagged': int(recall_fp),
    'screening_benign_flagged_pct': round(recall_fp/n_benign_train*100, 1),
    'test_prob_mean': round(test_probs.mean(), 4),
    'test_prob_median': round(np.median(test_probs), 4),
    'test_prob_max': round(test_probs.max(), 4),
    'test_screening_flagged': screening_count,
    'test_screening_flagged_pct': round(screening_pct, 1),
}

if not os.path.exists(MODEL_REGISTRY_FILE):
    df_registry = pd.DataFrame([model_record])
else:
    df_registry = pd.read_csv(MODEL_REGISTRY_FILE)
    df_registry = pd.concat([df_registry, pd.DataFrame([model_record])], ignore_index=True)

df_registry.to_csv(MODEL_REGISTRY_FILE, index=False)

print(f"\n  Model registry saved to: {MODEL_REGISTRY_FILE}")
print(f"  Total models recorded: {len(df_registry)}")
print(f"\n  Latest model summary:")
print(f"    Timestamp            : {model_record['timestamp']}")
print(f"    Ratio                : {model_record['ratio']}:1")
print(f"    CV F1                : {model_record['cv_f1']}")
print(f"    CV Recall            : {model_record['cv_recall']}")
print(f"    CV Precision         : {model_record['cv_precision']}")
print(f"    Screening Recall     : {model_record['screening_recall']}")
print(f"    Screening Precision  : {model_record['screening_precision']}")
print(f"    min_child_samples    : {model_record['min_child_samples']}")
print(f"    scale_pos_weight     : {model_record['scale_pos_weight']:.1f}")
print(f"    Interaction features : {model_record['num_interaction_features']}")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("ALL DONE")
print("=" * 60)
print(f"""
  OUTPUT STRUCTURE:
    {OUTPUT_FOLDER}/
      ├── reports/
      │   ├── v4_readable_report.txt              ✓ Human-readable summary
      │   ├── v4_global_metrics.json              ✓ Key metrics in JSON
      │   ├── v4_ratio_{RATIO}_fold_metrics.csv
      │   ├── v4_ratio_{RATIO}_feature_importance.csv
      │   ├── v4_ratio_{RATIO}_benign_risk_predictions.csv
      │   └── v4_ratio_{RATIO}_recall_target_metrics.csv
      └── graphs/
          ├── v4_oof_confusion_matrix.png         ✓ Out-of-fold CM
          ├── v4_oof_precision_recall_curve.png   ✓ Out-of-fold PR curve
          └── v4_feature_importance.png           ✓ Feature importance

  KEY RESULTS ({RATIO}:1 Ratio):
    CV F1         : {cv_f1:.4f}
    CV Precision  : {cv_prec:.4f}
    CV Recall     : {cv_rec:.4f}
    CV PR-AUC     : {cv_pr:.4f}

  TEST SET:
    {n_test:,} benign records scored and ranked
    Screening flagged: {screening_count:,} ({screening_pct:.1f}%)

  MODEL REGISTRY:
    {MODEL_REGISTRY_FILE} (shared across all ratios)

  TO RUN OTHER RATIOS:
    RATIO=10 python isic_model_v4_ratio_balanced.py
    RATIO=20 python isic_model_v4_ratio_balanced.py
""")
print("=" * 60)
