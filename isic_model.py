# =============================================================================
# ISIC 2024 — LightGBM Model Training & Testing
# =============================================================================
# Trains on train_final.csv (all 393 malignant + 80% benign)
# Tests  on test_final.csv  (20% benign — NEVER seen during training)
#
# The model outputs a malignancy risk probability for every test benign record.
# This probability answers: "How closely does this benign lesion resemble
# a confirmed malignant melanoma?"
#
# Output files:
#   reports/lgbm_fold_metrics.csv        — per-fold CV metrics
#   reports/lgbm_overall_metrics.csv     — overall CV metrics
#   reports/feature_importance.csv       — feature importance ranked
#   reports/benign_risk_predictions.csv  — FINAL ANSWER: test benign records
#                                          ranked by malignancy risk
#   reports/metrics_report.txt           — full readable report
#   graphs/lgbm_*.png                    — all model graphs
#
# Install once:
#   !pip install pandas numpy scikit-learn lightgbm matplotlib seaborn
# =============================================================================

import pandas as pd
import numpy as np
import os
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.getcwd(), '.matplotlib-cache'))
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
)
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

  Interaction Features   : 5

# =============================================================================
# CONFIGURATION
# =============================================================================

TRAIN_FILE = 'train_final_oversampled_random_10x.csv'   # 393 malignant + 80% benign
TEST_FILE  = 'test_final.csv'    # 20% benign — never seen by model

REPORT_DIR   = 'reports'
GRAPH_DIR    = 'graphs'
DPI          = 150
N_FOLDS      = 5
RANDOM_STATE = 42
TARGET_RECALL = 0.50

COLOR_BENIGN    = '#2E86AB'
COLOR_MALIGNANT = '#E84855'
COLOR_NEUTRAL   = '#F4A261'

CATEGORICAL_COLS = ['sex', 'anatom_site_general', 'tbp_lv_location_simple']
MODEL_REGISTRY_FILE = 'reports/model_registry.csv'


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
print("ISIC 2024 — LightGBM Model Training & Testing")
print("=" * 60)


# =============================================================================
# STEP 1 — Load training and test files
# =============================================================================

print(f"\nLoading files...")

df_train = pd.read_csv(TRAIN_FILE)
df_test  = pd.read_csv(TEST_FILE)

n_benign_train    = (df_train['malignant'] == 0).sum()
n_malignant_train = (df_train['malignant'] == 1).sum()
n_test            = len(df_test)

print(f"\n  TRAINING SET  ({TRAIN_FILE})")
print(f"    Total rows    : {len(df_train):,}")
print(f"    Malignant (1) : {n_malignant_train:,}  ← all real malignant records")
print(f"    Benign    (0) : {n_benign_train:,}  ← 80% of benign records")
print(f"    Imbalance     : {n_benign_train / n_malignant_train:.0f}:1")

print(f"\n  TEST SET  ({TEST_FILE})")
print(f"    Total rows    : {n_test:,}  ← 20% of benign, never seen by model")
print(f"    Malignant (1) : 0  ← none by design")
print(f"    Benign    (0) : {n_test:,}")
print(f"\n  These two sets have ZERO overlap — confirmed by isic_data_prep.py")


# =============================================================================
# STEP 2 — Encode categorical columns
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2 — Encoding categorical columns")
print("=" * 60)

# Fit encoders on TRAINING data only — never on test data
encoders = {}

for col in CATEGORICAL_COLS:
    if col not in df_train.columns:
        continue
    le = LabelEncoder()

    # Fit on train
    df_train[col] = le.fit_transform(df_train[col].astype(str))
    encoders[col] = le

    # Transform test using the SAME encoder fitted on train
    if col in df_test.columns:
        # Handle unseen categories in test gracefully
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
# Test has no malignant column — all benign by design

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

# Interaction 1: Color contrast × Clinical size (color intensity + lesion size)
if 'color_contrast_3d' in X_train.columns and 'clin_size_long_diam_mm' in X_train.columns:
    X_train['color_x_size'] = X_train['color_contrast_3d'] * X_train['clin_size_long_diam_mm']
    X_test['color_x_size'] = X_test['color_contrast_3d'] * X_test['clin_size_long_diam_mm']
    interaction_features['color_x_size'] = 'color_contrast_3d × clin_size'

# Interaction 2: Chroma contrast × Symmetry (color variance + shape regularity)
if 'chroma_contrast' in X_train.columns and 'tbp_lv_symm_2axis' in X_train.columns:
    X_train['chroma_x_symm'] = X_train['chroma_contrast'] * X_train['tbp_lv_symm_2axis']
    X_test['chroma_x_symm'] = X_test['chroma_contrast'] * X_test['tbp_lv_symm_2axis']
    interaction_features['chroma_x_symm'] = 'chroma_contrast × tbp_lv_symm_2axis'

# Interaction 3: Color ratio (differential color properties)
if 'color_contrast_3d' in X_train.columns and 'chroma_contrast' in X_train.columns:
    X_train['color_contrast_ratio'] = X_train['color_contrast_3d'] / (X_train['chroma_contrast'] + 1e-6)
    X_test['color_contrast_ratio'] = X_test['color_contrast_3d'] / (X_test['chroma_contrast'] + 1e-6)
    interaction_features['color_contrast_ratio'] = 'color_contrast_3d / chroma_contrast'

# Interaction 4: Size × Eccentricity (size + shape elongation)
if 'clin_size_long_diam_mm' in X_train.columns and 'tbp_lv_eccentricity' in X_train.columns:
    X_train['size_x_ecc'] = X_train['clin_size_long_diam_mm'] * X_train['tbp_lv_eccentricity']
    X_test['size_x_ecc'] = X_test['clin_size_long_diam_mm'] * X_test['tbp_lv_eccentricity']
    interaction_features['size_x_ecc'] = 'clin_size × tbp_lv_eccentricity'

# Interaction 5: Area × Border norm (size + border irregularity)
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

LGBM_PARAMS = {
    'n_estimators':      1500,
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

print("\n" + "=" * 60)
print("STEP 4 — Model configuration")
print("=" * 60)
print(f"  Algorithm          : LightGBM")
print(f"  n_estimators       : {LGBM_PARAMS['n_estimators']}")
print(f"  min_child_samples  : {LGBM_PARAMS['min_child_samples']}  (relaxed from 200 for minority learning)")
if CLASS_WEIGHT_MULTIPLIER > 0:
    print(f"  scale_pos_weight   : {effective_scale_pos_weight:.1f}  "
          f"({CLASS_WEIGHT_MULTIPLIER:.2f}x base imbalance weight — ENABLED)")
else:
    print("  scale_pos_weight   : disabled  (threshold optimization handles imbalance)")
print(f"  Validation         : StratifiedKFold (k={N_FOLDS})")
print(f"  Threshold strategy : Optimal via precision-recall curve")
print(f"  Screening target   : Recall >= {TARGET_RECALL:.0%}")
print(f"  Feature engineering: {len(interaction_features)} interaction features added")
print(f"  Early stopping     : 50 rounds")


# =============================================================================
# STEP 5 — Cross-validated training on train_final.csv
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5 — Cross-Validation on Training Set")
print("=" * 60)
print(f"  NOTE: Cross-validation uses only train_final.csv")
print(f"        test_final.csv is NOT touched until Step 7\n")

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
        'pr_auc': round(pr_auc,4), 'best_iter': model.best_iteration_,
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

df_folds.to_csv(f'{REPORT_DIR}/lgbm_fold_metrics.csv', index=False)
print(f"\n  Saved: {REPORT_DIR}/lgbm_fold_metrics.csv")

def threshold_for_target_recall(y_true, probs, target_recall):
    prec_vals, rec_vals, thresholds = precision_recall_curve(y_true, probs)
    valid_idx = np.where(rec_vals[:-1] >= target_recall)[0]

    if len(valid_idx) == 0:
        return thresholds[np.argmax(rec_vals[:-1])]

    # Among thresholds that reach the recall target, choose the best precision.
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
df_recall_targets.to_csv(f'{REPORT_DIR}/lgbm_recall_target_metrics.csv', index=False)

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
print(f"    Saved recall tradeoffs: {REPORT_DIR}/lgbm_recall_target_metrics.csv")


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

df_importance.to_csv(f'{REPORT_DIR}/feature_importance.csv', index=False)

print(f"\n  {'Rank':<6} {'Feature':<35} {'Importance':>12} {'Std':>8}")
print(f"  {'-'*65}")
for _, row in df_importance.iterrows():
    bar = '█' * int(row['importance'] / df_importance['importance'].max() * 20)
    print(f"  {int(row['rank']):<6} {row['feature']:<35} "
          f"{row['importance']:>12.1f} ±{row['std']:>6.1f}  {bar}")

print(f"\n  Saved: {REPORT_DIR}/feature_importance.csv")


# =============================================================================
# STEP 7 — PREDICT ON TEST SET (unseen benign records)
# =============================================================================
# This is the core output of the project.
# The model scores every test benign record by how closely it resembles
# the malignant profile learned from the 393 real malignant records.

print("\n" + "=" * 60)
print("STEP 7 — Predicting Malignancy Risk on Test Set")
print("=" * 60)
print(f"  Test set: {n_test:,} benign records that were NEVER seen during training")
print(f"  Using: best model from fold {df_folds['f1'].idxmax()+1} "
      f"(highest CV F1 = {df_folds['f1'].max():.4f})\n")

# Use the best model from cross-validation
best_fold_idx = df_folds['f1'].idxmax()
best_model    = best_models[best_fold_idx]

# Get malignancy probability for every test benign record
test_probs = best_model.predict_proba(X_test.values)[:, 1]

# Reload original (unencoded) test data to save readable output
df_test_out = pd.read_csv(TEST_FILE)
df_test_out['lgbm_malignancy_prob'] = test_probs
df_test_out['screening_flag'] = df_test_out['lgbm_malignancy_prob'] >= recall_threshold

# Risk tier labels based on the CV-optimized operating threshold
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

# Sort by probability — highest risk first
df_test_out = df_test_out.sort_values('lgbm_malignancy_prob', ascending=False)
df_test_out = df_test_out.reset_index(drop=True)
df_test_out.index += 1   # rank starts at 1
df_test_out.index.name = 'risk_rank'

# Summary
print(f"  Probability stats for test benign records:")
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
screening_pct = screening_count / len(df_test_out) * 100
print(f"\n  Screening mode flags:")
print(f"    Threshold      : {recall_threshold:.4f}")
print(f"    Test flagged   : {screening_count:,}/{len(df_test_out):,} ({screening_pct:.1f}%)")

print(f"\n  Top 20 highest-risk benign records:")
display_cols = ['lgbm_malignancy_prob', 'risk_tier', 'screening_flag', 'age_approx',
                'anatom_site_general', 'clin_size_long_diam_mm',
                'tbp_lv_nevi_confidence', 'tbp_lv_norm_border']
display_cols = [c for c in display_cols if c in df_test_out.columns]
print(df_test_out[display_cols].head(20).to_string())

df_test_out.to_csv(f'{REPORT_DIR}/benign_risk_predictions.csv')
print(f"\n  Saved: {REPORT_DIR}/benign_risk_predictions.csv")
print(f"  This is the FINAL ANSWER — {len(df_test_out):,} unseen benign records")
print(f"  ranked by how closely they resemble malignant melanoma.")


# =============================================================================
# STEP 8 — Generate all graphs
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8 — Generating graphs")
print("=" * 60)

saved_graphs = []

def save_graph(fig, filename):
    path = os.path.join(GRAPH_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    saved_graphs.append(filename)
    print(f"  Saved: {filename}")


# ── Graph 1: CV fold metrics ──────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('LightGBM — Cross-Validation Performance (Training Set)',
             fontsize=14, fontweight='bold')

for metric, col in zip(['f1','precision','recall','pr_auc'],
                       [COLOR_MALIGNANT, COLOR_BENIGN, COLOR_NEUTRAL, '#A23B72']):
    axes[0].plot(df_folds['fold'], df_folds[metric],
                 marker='o', label=metric.upper().replace('_','-'),
                 color=col, linewidth=2, markersize=7)
axes[0].set_xlabel('Fold')
axes[0].set_ylabel('Score')
axes[0].set_title('Metrics per Fold')
axes[0].set_xticks(df_folds['fold'])
axes[0].set_ylim(0, 1.05)
axes[0].legend(fontsize=9)

metric_labels = ['F1', 'Accuracy', 'Precision', 'Recall', 'PR-AUC']
metric_keys   = ['f1', 'accuracy', 'precision', 'recall', 'pr_auc']
bar_means = [df_folds[m].mean() for m in metric_keys]
bar_stds  = [df_folds[m].std()  for m in metric_keys]
bars = axes[1].bar(metric_labels, bar_means,
                   color=[COLOR_MALIGNANT,'#57CC99',COLOR_BENIGN,COLOR_NEUTRAL,'#A23B72'],
                   edgecolor='white', width=0.55, alpha=0.9)
axes[1].errorbar(range(len(metric_keys)), bar_means, yerr=bar_stds,
                 fmt='none', color='black', capsize=6, linewidth=1.5)
axes[1].set_ylim(0, 1.15)
axes[1].set_title('Mean Metrics ±std')
for bar, val in zip(bars, bar_means):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.04,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
save_graph(fig, 'lgbm_cv_metrics.png')


# ── Graph 2: CV confusion matrix ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_cv,
                               display_labels=['Benign', 'Malignant'])
disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='d')
ax.set_title('Confusion Matrix — CV on Training Set\n(LightGBM)', fontweight='bold')
plt.tight_layout()
save_graph(fig, 'lgbm_cv_confusion_matrix.png')


# ── Graph 3: PR curve (training CV) ──────────────────────────────────────────

prec_cv, rec_cv, _ = precision_recall_curve(y_train, cv_probs)
baseline = n_malignant_train / len(y_train)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(rec_cv, prec_cv, color=COLOR_MALIGNANT, linewidth=2,
        label=f'LightGBM CV  (PR-AUC={cv_pr:.4f})')
ax.axhline(baseline, color='gray', linestyle='--', linewidth=1.5,
           label=f'Random baseline ({baseline:.4f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve — CV on Training Set', fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
plt.tight_layout()
save_graph(fig, 'lgbm_pr_curve.png')


# ── Graph 4: ROC curve ───────────────────────────────────────────────────────

try:
    cv_roc = roc_auc_score(y_train, cv_probs)
    fpr, tpr, _ = roc_curve(y_train, cv_probs)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=COLOR_MALIGNANT, linewidth=2,
            label=f'LightGBM  (ROC-AUC={cv_roc:.4f})')
    ax.plot([0,1],[0,1], color='gray', linestyle='--', linewidth=1.5,
            label='Random baseline')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.set_title('ROC Curve — CV on Training Set', fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    save_graph(fig, 'lgbm_roc_curve.png')
except Exception as e:
    print(f"  SKIP ROC curve: {e}")


# ── Graph 5: Feature importance ──────────────────────────────────────────────

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
ax.set_title(f'LightGBM Feature Importance (Top {top_n})', fontweight='bold')
top5_patch = mpatches.Patch(color=COLOR_MALIGNANT, label='Top 5')
mid_patch  = mpatches.Patch(color=COLOR_NEUTRAL,   label='Rank 6–10')
rest_patch = mpatches.Patch(color=COLOR_BENIGN,     label='Rank 11+')
ax.legend(handles=[top5_patch, mid_patch, rest_patch], fontsize=9)
plt.tight_layout()
save_graph(fig, 'lgbm_feature_importance.png')


# ── Graph 6: Test set probability distribution ───────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Malignancy Risk Probability Distribution\n(Unseen Test Benign Records)',
             fontsize=14, fontweight='bold')

axes[0].hist(test_probs, bins=80, color=COLOR_BENIGN,
             alpha=0.8, edgecolor='white', linewidth=0.3)
for thr, col, lbl in [(risk_cut_1, 'green', f'Low→Elevated ({risk_cut_1:.3f})'),
                       (risk_cut_2, 'orange', f'Elevated→High ({risk_cut_2:.3f})'),
                       (risk_cut_3, 'red', f'High→Very High ({risk_cut_3:.3f})')]:
    axes[0].axvline(thr, color=col, linestyle='--', linewidth=1.5, label=lbl)
axes[0].set_xlabel('P(malignant | features)')
axes[0].set_ylabel('Count')
axes[0].set_title('Probability Histogram')
axes[0].legend(fontsize=8)

sorted_p = np.sort(test_probs)
cdf      = np.arange(1, len(sorted_p)+1) / len(sorted_p)
axes[1].plot(sorted_p, cdf, color=COLOR_BENIGN, linewidth=2)
for thr, col in [(risk_cut_1, 'green'), (risk_cut_2, 'orange'), (risk_cut_3, 'red')]:
    axes[1].axvline(thr, color=col, linestyle='--', linewidth=1.5)
axes[1].set_xlabel('P(malignant | features)')
axes[1].set_ylabel('Proportion of test records')
axes[1].set_title('Cumulative Distribution')

plt.tight_layout()
save_graph(fig, 'lgbm_test_probability_distribution.png')


# ── Graph 7: Risk tier bar chart ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
tier_order  = risk_labels
tier_counts = df_test_out['risk_tier'].value_counts()
tier_vals   = [tier_counts.get(t, 0) for t in tier_order]
tier_colors = ['#57CC99', COLOR_NEUTRAL, COLOR_MALIGNANT, '#8B0000']

bars = ax.bar(tier_order, tier_vals, color=tier_colors,
              edgecolor='white', width=0.6, alpha=0.9)
for bar, val in zip(bars, tier_vals):
    pct = val / len(df_test_out) * 100
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + len(df_test_out)*0.005,
            f'{val:,}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Risk Tier')
ax.set_ylabel('Number of Benign Records')
ax.set_title('Risk Tier Distribution\n(Unseen Test Benign Records)',
             fontweight='bold')
ax.set_ylim(0, max(tier_vals) * 1.2)
plt.tight_layout()
save_graph(fig, 'lgbm_test_risk_tiers.png')


# ── Graph 8: Threshold sensitivity ───────────────────────────────────────────

thresholds_range = np.linspace(0.01, 0.99, 200)
f1s, precs, recs = [], [], []

for thr in thresholds_range:
    preds_thr = (cv_probs >= thr).astype(int)
    f1s.append(f1_score(y_train, preds_thr, zero_division=0))
    precs.append(precision_score(y_train, preds_thr, zero_division=0))
    recs.append(recall_score(y_train, preds_thr, zero_division=0))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresholds_range, f1s,   color=COLOR_MALIGNANT, linewidth=2, label='F1')
ax.plot(thresholds_range, precs, color=COLOR_BENIGN,    linewidth=2,
        label='Precision', linestyle='--')
ax.plot(thresholds_range, recs,  color=COLOR_NEUTRAL,   linewidth=2,
        label='Recall', linestyle=':')
ax.axvline(overall_threshold, color='black', linestyle='--', linewidth=1.5,
           label=f'F1 threshold ({overall_threshold:.3f})')
ax.axvline(recall_threshold, color='red', linestyle='-.', linewidth=1.5,
           label=f'{TARGET_RECALL:.0%} recall threshold ({recall_threshold:.3f})')
ax.set_xlabel('Classification Threshold')
ax.set_ylabel('Score')
ax.set_title('Threshold Sensitivity', fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
plt.tight_layout()
save_graph(fig, 'lgbm_threshold_sensitivity.png')


# =============================================================================
# STEP 9 — Save complete metrics report
# =============================================================================

print("\n" + "=" * 60)
print("STEP 9 — Saving metrics report")
print("=" * 60)

report_lines = [
    "=" * 60,
    "ISIC 2024 — LightGBM Metrics Report",
    "=" * 60,
    "",
    "DATASET SPLIT",
    f"  train_final.csv   : {len(df_train):,} rows",
    f"    Malignant (all) : {n_malignant_train:,}",
    f"    Benign (80%)    : {n_benign_train:,}",
    f"  test_final.csv    : {n_test:,} rows (benign only — never seen by model)",
    f"  Overlap           : 0 rows (confirmed clean split)",
    "",
    "MODEL CONFIGURATION",
    f"  Algorithm         : LightGBM",
    f"  scale_pos_weight  : {'disabled' if CLASS_WEIGHT_MULTIPLIER == 0 else f'{effective_scale_pos_weight:.2f}'}",
    f"  Base imbalance    : {scale_pos_weight:.2f}",
    f"  Validation        : StratifiedKFold (k={N_FOLDS})",
    f"  Threshold         : {overall_threshold:.4f} (global CV F1-optimal threshold)",
    f"  Screening target  : recall >= {TARGET_RECALL:.0%}",
    f"  Screening thresh. : {recall_threshold:.4f}",
    "",
    "CROSS-VALIDATION RESULTS (on training set)",
    f"  {'Fold':<6} {'F1':>8} {'Accuracy':>10} {'Precision':>11} {'Recall':>8} {'PR-AUC':>8}",
    "  " + "-"*55,
]

for _, row in df_folds.iterrows():
    report_lines.append(
        f"  {int(row['fold']):<6} {row['f1']:>8.4f} {row['accuracy']:>10.4f} "
        f"{row['precision']:>11.4f} {row['recall']:>8.4f} {row['pr_auc']:>8.4f}"
    )

report_lines += [
    "  " + "-"*55,
    f"  {'MEAN':<6} {means['f1']:>8.4f} {means['accuracy']:>10.4f} "
    f"{means['precision']:>11.4f} {means['recall']:>8.4f} {means['pr_auc']:>8.4f}",
    f"  {'STD':<6} {stds['f1']:>8.4f} {stds['accuracy']:>10.4f} "
    f"{stds['precision']:>11.4f} {stds['recall']:>8.4f} {stds['pr_auc']:>8.4f}",
    "",
    "OVERALL CV METRICS (on training set)",
    f"  F1        : {cv_f1:.4f}",
    f"  Accuracy  : {cv_acc:.4f}",
    f"  Precision : {cv_prec:.4f}",
    f"  Recall    : {cv_rec:.4f}",
    f"  PR-AUC    : {cv_pr:.4f}",
    f"  Malignant caught: {tp}/{n_malignant_train} ({tp/n_malignant_train*100:.1f}%)",
    "",
    f"SCREENING MODE (target recall >= {TARGET_RECALL:.0%})",
    f"  Threshold        : {recall_threshold:.4f}",
    f"  F1               : {recall_f1:.4f}",
    f"  Accuracy         : {recall_acc:.4f}",
    f"  Precision        : {recall_prec:.4f}",
    f"  Recall           : {recall_rec:.4f}",
    f"  Malignant caught : {recall_tp}/{n_malignant_train} ({recall_tp/n_malignant_train*100:.1f}%)",
    f"  Benign flagged   : {recall_fp}/{n_benign_train} ({recall_fp/n_benign_train*100:.1f}%)",
    "",
    "RECALL TARGET TRADEOFFS",
    f"  {'Target':<8} {'Thresh':>9} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Benign flagged':>16}",
    "  " + "-"*66,
]

for _, row in df_recall_targets.iterrows():
    report_lines.append(
        f"  {row['target_recall']:<8.0%} {row['threshold']:>9.4f} "
        f"{row['f1']:>8.4f} {row['precision']:>8.4f} {row['recall']:>8.4f} "
        f"{int(row['benign_flagged']):>8,} ({row['benign_flagged_pct']:>5.1f}%)"
    )

report_lines += [
    "",
    "TEST SET RESULTS (unseen benign records)",
    f"  Records scored : {n_test:,}",
    f"  Prob mean      : {test_probs.mean():.4f}",
    f"  Prob median    : {np.median(test_probs):.4f}",
    f"  Prob max       : {test_probs.max():.4f}",
    f"  Screening flags: {screening_count:,}/{len(df_test_out):,} ({screening_pct:.1f}%)",
    "",
    "RISK TIER DISTRIBUTION (test set)",
]

for tier in risk_labels:
    count = tier_counts.get(tier, 0)
    pct   = count / len(df_test_out) * 100
    report_lines.append(f"  {tier:<12} : {count:>8,}  ({pct:.1f}%)")

report_lines += [
    "",
    "TOP 10 MOST IMPORTANT FEATURES",
]
for _, row in df_importance.head(10).iterrows():
    report_lines.append(
        f"  {int(row['rank']):<4} {row['feature']:<35} importance={row['importance']:.1f}"
    )

report_lines += ["", "=" * 60]
report_text = "\n".join(report_lines)

print(report_text)

with open(f'{REPORT_DIR}/metrics_report.txt', 'w') as f:
    f.write(report_text)

print(f"\n  Saved: {REPORT_DIR}/metrics_report.txt")


# =============================================================================
# STEP 10 — Record Model to Registry
# =============================================================================

print("\n" + "=" * 60)
print("STEP 10 — Recording model to registry")
print("=" * 60)

import datetime

model_record = {
    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
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
  KEY RESULTS:
    CV F1        : {cv_f1:.4f}
    CV Accuracy  : {cv_acc:.4f}
    CV Precision : {cv_prec:.4f}
    CV Recall    : {cv_rec:.4f}
    CV PR-AUC    : {cv_pr:.4f}
    Malignant caught (CV) : {tp}/{n_malignant_train} ({tp/n_malignant_train*100:.1f}%)

  SCREENING MODE ({TARGET_RECALL:.0%} recall target):
    Threshold       : {recall_threshold:.4f}
    Recall          : {recall_rec:.4f}
    Precision       : {recall_prec:.4f}
    Benign flagged  : {recall_fp:,}/{n_benign_train:,} ({recall_fp/n_benign_train*100:.1f}%)

  TEST SET OUTPUT:
    {n_test:,} unseen benign records scored and ranked
    Screening flagged: {screening_count:,} ({screening_pct:.1f}%)
    Saved to: {REPORT_DIR}/benign_risk_predictions.csv

  GRAPHS SAVED ({len(saved_graphs)} files):""")

for g in saved_graphs:
    print(f"    {GRAPH_DIR}/{g}")

print(f"""
  DOWNLOAD ALL AT ONCE:
    import shutil
    from google.colab import files
    shutil.make_archive('isic_outputs', 'zip', '.')
    files.download('isic_outputs.zip')
""")
print("=" * 60)
