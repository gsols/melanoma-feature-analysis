# =============================================================================
# ISIC 2024 — LightGBM V5: Optuna Full-Data Tuned + Platt Calibration
# =============================================================================
# Key features vs prior models:
#   - Optuna hyperparameter search on the full 400k-row dataset (50 trials, 3-fold CV)
#     so discovered parameters generalise to the true data distribution
#   - 5-fold StratifiedKFold CV with SMOTE oversampling (tuned sampling_strategy)
#   - Platt scaling calibration; C chosen via 5-fold Brier-score CV on OOF scores
#   - 3-tier risk system: Low / Medium / High
#       Low|Medium boundary  — Youden's J on ROC curve
#       Medium|High boundary — minimum recall constraint on PR curve
#   - All 8 engineered features from datasetv4
#   - Reads datasetv4 directly — no merge step
#
# Output folder: outputs/v5_outputs/
#   reports/
#     v5_readable_report.txt
#     v5_global_metrics.json
#     v5_fold_metrics.csv
#     v5_optuna_trials.csv
#     v5_threshold_sensitivity.csv
#     v5_risk_tiers.csv
#     v5_benign_risk_predictions.csv
#   graphs/
#     v5_optuna_param_importance.png
#     v5_optuna_trial_history.png
#     v5_fold_results.png
#     v5_calibration_curve.png
#     v5_roc_pr_curves.png
#     v5_risk_tier_distribution.png
#     v5_oof_confusion_matrix.png
#
# Usage:
#   python isic_model_v5_optuna_lgbm_platt.py
#
# Quick smoke test (fewer trials, fewer folds):
#   N_TRIALS=10 N_FOLDS=3 python isic_model_v5_optuna_lgbm_platt.py
#
# Change the recall target for the High-tier boundary:
#   RECALL_TARGET=0.40 python isic_model_v5_optuna_lgbm_platt.py
#
# Additional dependencies (beyond base requirements.txt):
#   pip install optuna imbalanced-learn
# =============================================================================

import os
import json
import warnings
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.getcwd(), '.matplotlib-cache'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    precision_recall_curve, roc_curve,
    precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from imblearn.over_sampling import SMOTE


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DATA    = str((_HERE / '../../datasets/datasetv4_merged_cleaned_engineered.csv').resolve())
PROCESSED_DIR = str(_HERE / 'processed_data')

N_TRIALS     = int(os.getenv('N_TRIALS', '50'))
N_FOLDS      = int(os.getenv('N_FOLDS', '5'))
RANDOM_STATE = 42
DPI          = 150

# Recall target for the Medium|High tier boundary.
# Adjust via env var: RECALL_TARGET=0.40 python isic_model_v5_optuna_lgbm_platt.py
CHOSEN_RECALL_TARGET = float(os.getenv('RECALL_TARGET', '0.50'))

_V5_OUTPUTS = (_HERE / '../../outputs/v5_outputs').resolve()
REPORT_DIR  = str(_V5_OUTPUTS / 'reports')
GRAPH_DIR   = str(_V5_OUTPUTS / 'graphs')

COLOR_BENIGN    = '#2E86AB'
COLOR_MALIGNANT = '#E84855'
COLOR_NEUTRAL   = '#F4A261'

# All 11 raw TBP numeric features + 8 engineered features from datasetv4.
NUMERIC_FEATURES = [
    'age_approx', 'clin_size_long_diam_mm',
    'tbp_lv_area_perim_ratio', 'tbp_lv_norm_border', 'tbp_lv_symm_2axis',
    'tbp_lv_eccentricity', 'tbp_lv_color_std_mean', 'tbp_lv_norm_color',
    'tbp_lv_deltaLBnorm', 'tbp_lv_radial_color_std_max', 'tbp_lv_nevi_confidence',
    # 8 engineered features (computed by data_preprocessing_pipeline.py)
    'color_contrast_3d', 'elongation', 'nevi_color_tension', 'log_area',
    'stdL_ratio', 'compactness', 'chroma_contrast', 'radial_color_ratio',
]
CAT_FEATURES = ['sex', 'anatom_site_general', 'tbp_lv_location_simple']


# =============================================================================
# SETUP
# =============================================================================

os.makedirs(REPORT_DIR,   exist_ok=True)
os.makedirs(GRAPH_DIR,    exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', font='Arial')
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor':    'white',
    'savefig.facecolor': 'white', 'font.family':      'sans-serif',
    'axes.titlesize':   13,       'axes.labelsize':   11,
})


# =============================================================================
# HELPERS
# =============================================================================

def save_graph(fig, filename):
    path = os.path.join(GRAPH_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {filename}')


# =============================================================================
# STEP 0 — Load datasetv4
# =============================================================================

print('=' * 70)
print('ISIC 2024 — V5 LightGBM: Optuna Full-Data Tuned + Platt Calibration')
print('=' * 70)

print('\nSTEP 0 — Loading datasetv4')
print('-' * 70)

df_full = pd.read_csv(INPUT_DATA)
print(f'  Loaded: {df_full.shape[0]:,} rows x {df_full.shape[1]} columns')
print(f'  Malignant: {df_full["malignant"].sum():,}')
print(f'  Benign   : {(df_full["malignant"] == 0).sum():,}')
print(f'  Imbalance: 1:{int((df_full["malignant"] == 0).sum() / df_full["malignant"].sum())}')

# One-hot encode categorical columns; keep all 19 numeric features as-is.
df_enc = pd.get_dummies(
    df_full[NUMERIC_FEATURES + CAT_FEATURES + ['malignant']],
    columns=CAT_FEATURES,
    drop_first=True,
)
bool_cols = df_enc.select_dtypes(include='bool').columns
df_enc[bool_cols] = df_enc[bool_cols].astype(int)

X_model      = df_enc.drop(columns=['malignant']).astype(float)
y_model      = df_full['malignant'].values.astype(int)
FEATURE_NAMES = list(X_model.columns)
n_encoded_cat = len(FEATURE_NAMES) - len(NUMERIC_FEATURES)

print(f'  Features: {len(NUMERIC_FEATURES)} numeric + {n_encoded_cat} encoded categorical '
      f'= {len(FEATURE_NAMES)} total')


# =============================================================================
# STEP 1 — Optuna hyperparameter tuning (full data, 3-fold CV)
# =============================================================================

print(f'\nSTEP 1 — Optuna hyperparameter tuning ({N_TRIALS} trials, 3-fold StratifiedKFold)')
print('-' * 70)
print('  Searching on the full dataset so parameters generalise to the true distribution.')
print(f'  Expected runtime: ~2–3 h for N_TRIALS=50. Use N_TRIALS=10 for a smoke test.')


def objective(trial):
    sampling_strategy = trial.suggest_float('sampling_strategy', 0.05, 0.30)
    scale_pos_weight  = trial.suggest_float('scale_pos_weight',  2.0, 50.0, log=True)
    learning_rate     = trial.suggest_float('learning_rate',     0.005, 0.10, log=True)
    num_leaves        = trial.suggest_int(  'num_leaves',        31, 127)
    min_child_samples = trial.suggest_int(  'min_child_samples', 10, 100)
    subsample         = trial.suggest_float('subsample',         0.5, 1.0)
    colsample_bytree  = trial.suggest_float('colsample_bytree',  0.5, 1.0)
    reg_alpha         = trial.suggest_float('reg_alpha',         1e-8, 1.0, log=True)
    reg_lambda        = trial.suggest_float('reg_lambda',        1e-8, 1.0, log=True)

    cv3     = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    pr_aucs = []

    for tr_idx, val_idx in cv3.split(X_model, y_model):
        X_tr, X_val = X_model.iloc[tr_idx], X_model.iloc[val_idx]
        y_tr, y_val = y_model[tr_idx], y_model[val_idx]

        if y_val.sum() < 2:
            continue

        try:
            smote = SMOTE(sampling_strategy=sampling_strategy,
                          random_state=RANDOM_STATE, k_neighbors=5)
            X_res, y_res = smote.fit_resample(X_tr, y_tr)
        except ValueError:
            return 0.0

        clf = LGBMClassifier(
            n_estimators=500,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            scale_pos_weight=scale_pos_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            metric='auc',
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(
            X_res, y_res,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(period=-1)],
        )
        proba = clf.predict_proba(X_val)[:, 1]
        pr_aucs.append(average_precision_score(y_val, proba))

    return float(np.mean(pr_aucs)) if pr_aucs else 0.0


def _progress_cb(study, trial):
    if (trial.number + 1) % 10 == 0 or trial.number == 0:
        print(f'  Trial {trial.number + 1:>3}/{N_TRIALS} | best PR-AUC: {study.best_value:.4f}')


study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False, callbacks=[_progress_cb])

best_params = study.best_params
best_value  = study.best_value
print(f'\n  Best PR-AUC (3-fold): {best_value:.4f}')
for k, v in best_params.items():
    print(f'    {k}: {v:.6g}' if isinstance(v, float) else f'    {k}: {v}')

# Save trial log
df_trials = pd.DataFrame([
    {**{'trial': t.number + 1, 'value': t.value}, **t.params}
    for t in study.trials if t.value is not None
])
df_trials.to_csv(f'{REPORT_DIR}/v5_optuna_trials.csv', index=False)
print(f'\n  Saved: v5_optuna_trials.csv ({len(df_trials)} trials recorded)')

# Optuna parameter importance plot
try:
    importances   = optuna.importance.get_param_importances(study)
    params_sorted = list(importances.keys())
    imp_sorted    = list(importances.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(params_sorted[::-1], imp_sorted[::-1], color=COLOR_BENIGN)
    ax.set_xlabel('Importance (fANOVA)')
    ax.set_title('Optuna Hyperparameter Importance — V5', fontweight='bold')
    plt.tight_layout()
    save_graph(fig, 'v5_optuna_param_importance.png')
except Exception:
    print('  Optuna param importance skipped (insufficient completed trials).')

# Optuna trial history plot
trial_values = [t.value for t in study.trials if t.value is not None]
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(range(1, len(trial_values) + 1), trial_values, 'o-',
        color=COLOR_BENIGN, alpha=0.7, ms=4)
ax.axhline(best_value, color=COLOR_MALIGNANT, linestyle='--',
           label=f'Best = {best_value:.4f}')
ax.set_xlabel('Trial')
ax.set_ylabel('PR-AUC (3-fold)')
ax.set_title('Optuna Trial History — V5', fontweight='bold')
ax.legend()
plt.tight_layout()
save_graph(fig, 'v5_optuna_trial_history.png')


# =============================================================================
# STEP 2 — Cross-validation with SMOTE (tuned parameters)
# =============================================================================

print(f'\nSTEP 2 — {N_FOLDS}-fold StratifiedKFold CV with SMOTE (tuned parameters)')
print('-' * 70)

sgkf          = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
oof_proba_raw = np.zeros(len(y_model))
fold_results  = []
best_iterations = []

for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_model, y_model)):
    X_tr, X_val = X_model.iloc[tr_idx], X_model.iloc[val_idx]
    y_tr, y_val = y_model[tr_idx], y_model[val_idx]

    smote = SMOTE(
        sampling_strategy=best_params['sampling_strategy'],
        random_state=RANDOM_STATE,
        k_neighbors=5,
    )
    X_res, y_res = smote.fit_resample(X_tr, y_tr)

    lgbm = LGBMClassifier(
        n_estimators=1000,
        learning_rate=best_params['learning_rate'],
        num_leaves=best_params['num_leaves'],
        min_child_samples=best_params['min_child_samples'],
        scale_pos_weight=best_params['scale_pos_weight'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        reg_alpha=best_params['reg_alpha'],
        reg_lambda=best_params['reg_lambda'],
        metric='auc',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm.fit(
        X_res, y_res,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(100, verbose=False), log_evaluation(period=-1)],
    )

    best_iterations.append(lgbm.best_iteration_)
    raw_proba = lgbm.predict_proba(X_val)[:, 1]
    oof_proba_raw[val_idx] = raw_proba

    pr_auc = average_precision_score(y_val, raw_proba)
    auroc  = roc_auc_score(y_val, raw_proba)
    n_pos  = int(y_val.sum())

    fold_results.append({
        'fold': fold + 1, 'pr_auc': pr_auc, 'auroc': auroc,
        'n_malignant_val': n_pos,
        'best_iteration': lgbm.best_iteration_,
        'n_train_after_smote': len(y_res),
    })
    print(f'  Fold {fold + 1}: PR-AUC={pr_auc:.4f}  AUROC={auroc:.4f}  '
          f'val_malignant={n_pos}  best_iter={lgbm.best_iteration_}  '
          f'train_size={len(y_res):,}')

fold_df = pd.DataFrame(fold_results)
print(f'\n  Mean PR-AUC : {fold_df["pr_auc"].mean():.4f} ± {fold_df["pr_auc"].std():.4f}')
print(f'  Mean AUROC  : {fold_df["auroc"].mean():.4f} ± {fold_df["auroc"].std():.4f}')
print(f'  Mean best_iteration: {np.mean(best_iterations):.0f}')
fold_df.to_csv(f'{REPORT_DIR}/v5_fold_metrics.csv', index=False)
print(f'  Saved: v5_fold_metrics.csv')

# Fold results bar plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, metric, title in [
    (axes[0], 'pr_auc', 'PR-AUC per Fold'),
    (axes[1], 'auroc',  'AUROC per Fold'),
]:
    ax.bar(fold_df['fold'], fold_df[metric], color=COLOR_BENIGN, edgecolor='black')
    ax.axhline(fold_df[metric].mean(), color=COLOR_MALIGNANT, linestyle='--',
               label=f'Mean = {fold_df[metric].mean():.4f}')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Fold')
    ax.set_ylabel(metric.replace('_', '-').upper())
    ax.legend()
    ax.set_ylim(0, max(fold_df[metric].max() * 1.25, 0.5))
plt.tight_layout()
save_graph(fig, 'v5_fold_results.png')


# =============================================================================
# STEP 3 — Platt scaling calibration
# =============================================================================

print('\nSTEP 3 — Platt scaling calibration')
print('-' * 70)
print('  Selecting C via 5-fold Brier-score cross-validation on OOF predictions.')

C_candidates = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
brier_by_C   = {}

print(f'\n  {"C":>10} | {"Mean Brier (5-fold CV)":>22}')
print('  ' + '-' * 35)
for C in C_candidates:
    lr     = LogisticRegression(C=C, max_iter=1000)
    scores = cross_val_score(
        lr,
        oof_proba_raw.reshape(-1, 1),
        y_model,
        cv=5,
        scoring='neg_brier_score',
    )
    brier_by_C[C] = -scores.mean()
    print(f'  {C:>10} | {brier_by_C[C]:>22.8f}')

best_C = min(brier_by_C, key=brier_by_C.get)
print(f'\n  Selected C = {best_C}  (lowest Brier score)')

calibrator    = LogisticRegression(C=best_C, max_iter=1000)
calibrator.fit(oof_proba_raw.reshape(-1, 1), y_model)
oof_proba_cal = calibrator.predict_proba(oof_proba_raw.reshape(-1, 1))[:, 1]

brier_raw  = brier_score_loss(y_model, oof_proba_raw)
brier_cal  = brier_score_loss(y_model, oof_proba_cal)
brier_base = brier_score_loss(y_model, np.full(len(y_model), y_model.mean()))
pr_auc_cal = average_precision_score(y_model, oof_proba_cal)
auroc_cal  = roc_auc_score(y_model, oof_proba_cal)

print(f'\n  Brier (raw)        : {brier_raw:.6f}')
print(f'  Brier (calibrated) : {brier_cal:.6f}  (lower = better)')
print(f'  Brier (base rate)  : {brier_base:.6f}')
print(f'  PR-AUC (cal. OOF)  : {pr_auc_cal:.4f}')
print(f'  AUROC  (cal. OOF)  : {auroc_cal:.4f}')
print(f'  True malignancy rate         : {y_model.mean():.6f}')
print(f'  Mean calibrated probability  : {oof_proba_cal.mean():.6f}')

# Calibration curves (raw vs calibrated)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, proba, label, color in [
    (axes[0], oof_proba_raw, 'Raw (uncalibrated)',            COLOR_BENIGN),
    (axes[1], oof_proba_cal, f'Calibrated (Platt, C={best_C})', COLOR_MALIGNANT),
]:
    frac_pos, mean_pred = calibration_curve(y_model, proba, n_bins=20, strategy='quantile')
    ax.plot(mean_pred, frac_pos, 's-', color=color, label=label)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction actually malignant')
    ax.set_title(f'Calibration Curve — {label}', fontweight='bold')
    ax.legend()
plt.tight_layout()
save_graph(fig, 'v5_calibration_curve.png')


# =============================================================================
# STEP 4 — Risk tier threshold selection
# =============================================================================

print('\nSTEP 4 — Risk tier threshold selection')
print('-' * 70)

fpr_arr, tpr_arr, roc_thresh = roc_curve(y_model, oof_proba_cal)
prec_arr, rec_arr, pr_thresh = precision_recall_curve(y_model, oof_proba_cal)

# Low|Medium boundary: Youden's J maximises sensitivity + specificity jointly.
j_scores       = tpr_arr - fpr_arr
best_j_idx     = int(np.argmax(j_scores))
youdens_thresh = float(roc_thresh[best_j_idx])
sensitivity_j  = tpr_arr[best_j_idx]
specificity_j  = 1 - fpr_arr[best_j_idx]
total_malignant = int(y_model.sum())

print(f"  Low|Medium boundary (Youden's J): {youdens_thresh:.6f}")
print(f'    Sensitivity: {sensitivity_j:.1%}  Specificity: {specificity_j:.1%}')

# Medium|High boundary: sensitivity analysis across recall targets.
recall_targets = [0.30, 0.40, 0.50, 0.60, 0.75]
analysis_rows  = []

print(f'\n  {"Recall target":>14} | {"High threshold":>15} | '
      f'{"High tier size":>14} | {"Sensitivity":>12} | {"PPV":>10}')
print('  ' + '-' * 75)

for target in recall_targets:
    valid = np.where(rec_arr[:-1] >= target)[0]
    if len(valid) == 0:
        print(f'  {target:>13.0%}  | (no valid threshold at this recall level)')
        continue

    high_thresh = float(pr_thresh[valid[-1]])
    if high_thresh <= youdens_thresh:
        high_thresh = youdens_thresh * 1.001

    high_mask  = oof_proba_cal >= high_thresh
    n_high     = int(high_mask.sum())
    n_high_mal = int(y_model[high_mask].sum())
    sens       = n_high_mal / total_malignant
    ppv        = n_high_mal / n_high if n_high > 0 else 0.0

    analysis_rows.append({
        'recall_target':    target,
        'high_threshold':   round(high_thresh, 6),
        'high_tier_size':   n_high,
        'high_sensitivity': round(sens, 4),
        'high_ppv':         round(ppv, 6),
    })
    print(f'  {target:>13.0%}  | {high_thresh:>15.6f} | {n_high:>14,} | '
          f'{sens:>11.1%} | {ppv:>9.3%}')

df_thresh = pd.DataFrame(analysis_rows)
df_thresh.to_csv(f'{REPORT_DIR}/v5_threshold_sensitivity.csv', index=False)
print(f'\n  Saved: v5_threshold_sensitivity.csv')

chosen_row = df_thresh[df_thresh['recall_target'] == CHOSEN_RECALL_TARGET]
if chosen_row.empty:
    available = list(df_thresh['recall_target'])
    raise ValueError(
        f'No sensitivity row for recall target {CHOSEN_RECALL_TARGET}. '
        f'Available: {available}. Set RECALL_TARGET to one of these values.'
    )

threshold_high   = float(chosen_row['high_threshold'].iloc[0])
threshold_medium = youdens_thresh

print(f'\n  Applied thresholds (RECALL_TARGET = {CHOSEN_RECALL_TARGET:.0%}):')
print(f'    High   >= {threshold_high:.6f}  '
      f'(sensitivity={chosen_row["high_sensitivity"].iloc[0]:.1%}, '
      f'PPV={chosen_row["high_ppv"].iloc[0]:.3%})')
print(f"    Medium  {threshold_medium:.6f} – {threshold_high:.6f}  "
      f"(Youden's J: sensitivity={sensitivity_j:.1%}, specificity={specificity_j:.1%})")
print(f'    Low    < {threshold_medium:.6f}')

# ROC + PR curve plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(fpr_arr, tpr_arr, color=COLOR_BENIGN, lw=2, label=f'AUROC = {auroc_cal:.4f}')
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
axes[0].axvline(fpr_arr[best_j_idx], color=COLOR_NEUTRAL, linestyle=':',
                label=f"Youden's J (TPR={sensitivity_j:.2f})")
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve — V5', fontweight='bold')
axes[0].legend(fontsize=9)

axes[1].plot(rec_arr, prec_arr, color=COLOR_BENIGN, lw=2, label=f'PR-AUC = {pr_auc_cal:.4f}')
axes[1].axhline(y_model.mean(), color='gray', linestyle='--',
                label=f'Baseline ({y_model.mean():.4%})')
axes[1].axvline(CHOSEN_RECALL_TARGET, color=COLOR_MALIGNANT, linestyle=':',
                label=f'{CHOSEN_RECALL_TARGET:.0%} recall (Medium|High boundary)')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve — V5', fontweight='bold')
axes[1].legend(fontsize=9)

plt.tight_layout()
save_graph(fig, 'v5_roc_pr_curves.png')


# =============================================================================
# STEP 5 — Assign risk tiers
# =============================================================================

print('\nSTEP 5 — Assigning risk tiers')
print('-' * 70)


def assign_tier(p):
    if p >= threshold_high:
        return 'High'
    elif p >= threshold_medium:
        return 'Medium'
    return 'Low'


df_out = df_full.copy()
df_out['malignancy_proba'] = oof_proba_cal
df_out['risk_tier']        = df_out['malignancy_proba'].map(assign_tier)

TIER_ORDER  = ['High', 'Medium', 'Low']
TIER_COLORS = [COLOR_MALIGNANT, COLOR_NEUTRAL, COLOR_BENIGN]

print(f'  {"Tier":<8} {"Records":>10} {"Malignant":>10} {"Sensitivity":>13} {"PPV":>10}')
print('  ' + '-' * 57)
tier_rows = []
for tier in TIER_ORDER:
    mask   = df_out['risk_tier'] == tier
    n_tot  = int(mask.sum())
    n_mal  = int(df_out.loc[mask, 'malignant'].sum())
    sens   = n_mal / total_malignant
    ppv    = n_mal / n_tot if n_tot > 0 else 0.0
    print(f'  {tier:<8} {n_tot:>10,} {n_mal:>10} {sens:>12.1%} {ppv:>10.4%}')
    tier_rows.append({
        'tier': tier, 'n_records': n_tot, 'n_malignant': n_mal,
        'sensitivity': round(sens, 4), 'ppv': round(ppv, 6),
    })

df_tiers = pd.DataFrame(tier_rows)
df_tiers.to_csv(f'{REPORT_DIR}/v5_risk_tiers.csv', index=False)
print(f'\n  Saved: v5_risk_tiers.csv')

# Per-record predictions sorted by risk (highest first)
df_predictions = (
    df_out[['malignant', 'malignancy_proba', 'risk_tier']]
    .sort_values('malignancy_proba', ascending=False)
    .reset_index(drop=True)
)
df_predictions.index += 1
df_predictions.index.name = 'risk_rank'
df_predictions.to_csv(f'{REPORT_DIR}/v5_benign_risk_predictions.csv')
print(f'  Saved: v5_benign_risk_predictions.csv')

# Full scored dataset for downstream analysis
df_out.to_csv(f'{PROCESSED_DIR}/v5_full_scored.csv', index=False)
print(f'  Saved: processed_data/v5_full_scored.csv')


# =============================================================================
# STEP 6 — Confusion matrix at the High-tier threshold
# =============================================================================

print('\nSTEP 6 — Confusion matrix (High-tier threshold)')
print('-' * 70)

oof_preds = (oof_proba_cal >= threshold_high).astype(int)
cm        = confusion_matrix(y_model, oof_preds)
tn, fp, fn, tp = cm.ravel()

cv_f1   = f1_score(y_model, oof_preds, zero_division=0)
cv_prec = precision_score(y_model, oof_preds, zero_division=0)
cv_rec  = recall_score(y_model, oof_preds, zero_division=0)

print(f'  Threshold : {threshold_high:.6f}')
print(f'  TP={tp}  FP={fp}  FN={fn}  TN={tn}')
print(f'  F1={cv_f1:.4f}  Precision={cv_prec:.4f}  Recall={cv_rec:.4f}')

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm,
                       display_labels=['Benign', 'Malignant']).plot(
    ax=ax, colorbar=False, cmap='Blues', values_format='d',
)
ax.set_title(
    f'Confusion Matrix — V5 OOF (High-tier threshold = {threshold_high:.4f})',
    fontweight='bold',
)
plt.tight_layout()
save_graph(fig, 'v5_oof_confusion_matrix.png')


# =============================================================================
# STEP 7 — Risk tier distribution plots
# =============================================================================

print('\nSTEP 7 — Risk tier distribution graphs')
print('-' * 70)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

tier_n = [df_out[df_out['risk_tier'] == t].shape[0] for t in TIER_ORDER]
axes[0].bar(TIER_ORDER, tier_n, color=TIER_COLORS)
axes[0].set_title('Records per Risk Tier', fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(tier_n):
    axes[0].text(i, v + max(tier_n) * 0.01, f'{v:,}',
                 ha='center', fontsize=9, fontweight='bold')

mal_n = [int(df_out[df_out['risk_tier'] == t]['malignant'].sum()) for t in TIER_ORDER]
axes[1].bar(TIER_ORDER, mal_n, color=TIER_COLORS)
axes[1].set_title('Malignant Cases per Risk Tier', fontweight='bold')
axes[1].set_ylabel('Count')
for i, v in enumerate(mal_n):
    axes[1].text(i, v + 2, f'{v}\n({v / total_malignant:.0%})',
                 ha='center', fontsize=9, fontweight='bold')

for tier, color in zip(TIER_ORDER, TIER_COLORS):
    subset = df_out[df_out['risk_tier'] == tier]['malignancy_proba']
    axes[2].hist(subset, bins=50, alpha=0.6, label=tier, color=color, density=True)
axes[2].set_xlabel('Calibrated Probability')
axes[2].set_ylabel('Density')
axes[2].set_title('Probability Distribution by Tier', fontweight='bold')
axes[2].legend()
axes[2].axvline(x=threshold_high,   color=COLOR_MALIGNANT, linestyle='--', lw=1.5)
axes[2].axvline(x=threshold_medium, color=COLOR_NEUTRAL,   linestyle='--', lw=1.5)
plt.tight_layout()
save_graph(fig, 'v5_risk_tier_distribution.png')


# =============================================================================
# STEP 8 — Save readable report and global metrics JSON
# =============================================================================

print('\nSTEP 8 — Saving reports')
print('-' * 70)

global_metrics = {
    'n':                    int(len(y_model)),
    'n_malignant':          int(y_model.sum()),
    'n_benign':             int((y_model == 0).sum()),
    'optuna_n_trials':      N_TRIALS,
    'optuna_best_pr_auc':   float(best_value),
    'cv_n_folds':           N_FOLDS,
    'mean_fold_pr_auc':     float(fold_df['pr_auc'].mean()),
    'std_fold_pr_auc':      float(fold_df['pr_auc'].std()),
    'mean_fold_auroc':      float(fold_df['auroc'].mean()),
    'std_fold_auroc':       float(fold_df['auroc'].std()),
    'brier_raw':            float(brier_raw),
    'brier_calibrated':     float(brier_cal),
    'brier_baseline':       float(brier_base),
    'platt_C':              float(best_C),
    'pr_auc_calibrated':    float(pr_auc_cal),
    'auroc_calibrated':     float(auroc_cal),
    'threshold_high':       float(threshold_high),
    'threshold_medium':     float(threshold_medium),
    'chosen_recall_target': float(CHOSEN_RECALL_TARGET),
    'sensitivity_high':     float(chosen_row['high_sensitivity'].iloc[0]),
    'specificity_medium':   float(specificity_j),
    'f1':                   float(cv_f1),
    'precision':            float(cv_prec),
    'recall':               float(cv_rec),
    'tp':                   int(tp),
    'fp':                   int(fp),
    'fn':                   int(fn),
    'tn':                   int(tn),
    **{f'param_{k}': v for k, v in best_params.items()},
}

with open(f'{REPORT_DIR}/v5_global_metrics.json', 'w') as f:
    json.dump(global_metrics, f, indent=2)
print('  Saved: v5_global_metrics.json')

report_lines = [
    '=' * 80,
    'ISIC 2024 — V5 LightGBM: Optuna Full-Data Tuned + Platt Calibration',
    '=' * 80,
    '',
    'DATASET',
    f'  Total rows         : {len(y_model):,}',
    f'  Malignant          : {y_model.sum():,}',
    f'  Benign             : {(y_model == 0).sum():,}',
    f'  Imbalance ratio    : 1:{int((y_model == 0).sum() / y_model.sum())}',
    '',
    'MODEL',
    f'  Algorithm          : LightGBM + SMOTE oversampling',
    f'  CV design          : StratifiedKFold ({N_FOLDS} folds)',
    f'  Optuna trials      : {N_TRIALS}  (3-fold CV on full data per trial)',
    f'  Features (numeric) : {len(NUMERIC_FEATURES)}  (11 raw TBP + 8 engineered)',
    f'  Features (total)   : {len(FEATURE_NAMES)}  (after one-hot encoding of 3 categorical cols)',
    '',
    'OPTUNA BEST PARAMETERS',
] + [
    (f'  {k:<25} : {v:.6g}' if isinstance(v, float) else f'  {k:<25} : {v}')
    for k, v in best_params.items()
] + [
    f'  {"Best 3-fold PR-AUC":<25} : {best_value:.6f}',
    '',
    'CROSS-VALIDATION (raw probabilities)',
    f'  {"Mean PR-AUC":<25} : {fold_df["pr_auc"].mean():.6f} ± {fold_df["pr_auc"].std():.6f}',
    f'  {"Mean AUROC":<25} : {fold_df["auroc"].mean():.6f} ± {fold_df["auroc"].std():.6f}',
    '',
    fold_df.to_string(index=False),
    '',
    'PLATT CALIBRATION',
    f'  {"Platt C":<25} : {best_C}  (5-fold Brier CV)',
    f'  {"Brier (raw)":<25} : {brier_raw:.6f}',
    f'  {"Brier (calibrated)":<25} : {brier_cal:.6f}',
    f'  {"Brier (base rate)":<25} : {brier_base:.6f}',
    f'  {"PR-AUC (calibrated OOF)":<25} : {pr_auc_cal:.6f}',
    f'  {"AUROC  (calibrated OOF)":<25} : {auroc_cal:.6f}',
    '',
    'RISK TIER THRESHOLDS',
    f"  Low|Medium boundary  : {threshold_medium:.6f}  "
    f"(Youden's J, sensitivity={sensitivity_j:.1%}, specificity={specificity_j:.1%})",
    f'  Medium|High boundary : {threshold_high:.6f}  '
    f'(recall target={CHOSEN_RECALL_TARGET:.0%}, '
    f'actual={chosen_row["high_sensitivity"].iloc[0]:.1%}, '
    f'PPV={chosen_row["high_ppv"].iloc[0]:.3%})',
    '',
    'RISK TIER SUMMARY',
    df_tiers.to_string(index=False),
    '',
    'THRESHOLD SENSITIVITY ANALYSIS',
    df_thresh.to_string(index=False),
    '',
    'CONFUSION MATRIX (High-tier threshold)',
    f'  TP={tp}  FP={fp}  FN={fn}  TN={tn}',
    f'  F1        : {cv_f1:.6f}',
    f'  Precision : {cv_prec:.6f}',
    f'  Recall    : {cv_rec:.6f}',
    '',
    'OUTPUT FILES',
    f'  {REPORT_DIR}/v5_readable_report.txt',
    f'  {REPORT_DIR}/v5_global_metrics.json',
    f'  {REPORT_DIR}/v5_fold_metrics.csv',
    f'  {REPORT_DIR}/v5_optuna_trials.csv',
    f'  {REPORT_DIR}/v5_threshold_sensitivity.csv',
    f'  {REPORT_DIR}/v5_risk_tiers.csv',
    f'  {REPORT_DIR}/v5_benign_risk_predictions.csv',
    f'  {GRAPH_DIR}/v5_optuna_param_importance.png',
    f'  {GRAPH_DIR}/v5_optuna_trial_history.png',
    f'  {GRAPH_DIR}/v5_fold_results.png',
    f'  {GRAPH_DIR}/v5_calibration_curve.png',
    f'  {GRAPH_DIR}/v5_roc_pr_curves.png',
    f'  {GRAPH_DIR}/v5_risk_tier_distribution.png',
    f'  {GRAPH_DIR}/v5_oof_confusion_matrix.png',
    '',
    '=' * 80,
]

report_text = '\n'.join(report_lines)
with open(f'{REPORT_DIR}/v5_readable_report.txt', 'w') as f:
    f.write(report_text)
print('  Saved: v5_readable_report.txt')

print('\n' + report_text)

# =============================================================================
# DONE
# =============================================================================

print('\n' + '=' * 70)
print('ALL DONE')
print('=' * 70)
print(f'\nOutputs saved to:')
print(f'  Reports : {REPORT_DIR}')
print(f'  Graphs  : {GRAPH_DIR}')
