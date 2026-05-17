# =============================================================================
# ISIC 2024 — LightGBM Experiment Runner
# =============================================================================
# Tests several LightGBM configurations on train_final.csv using the same
# cross-validation setup as isic_model.py. The goal is to compare ranking quality
# and screening tradeoffs before choosing settings for the final model.
#
# Output:
#   reports/lgbm_experiment_results.csv
# =============================================================================

import os
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.getcwd(), '.matplotlib-cache'))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


TRAIN_FILE = 'train_final.csv'
REPORT_DIR = 'reports'
N_FOLDS = 5
RANDOM_STATE = 42
TARGET_RECALLS = [0.30, 0.50, 0.70, 0.90]

CATEGORICAL_COLS = ['sex', 'anatom_site_general', 'tbp_lv_location_simple']
DROP_COLS = ['malignant', 'risk_score']


def threshold_for_best_f1(y_true, probs):
    prec_vals, rec_vals, thresholds = precision_recall_curve(y_true, probs)
    f1_vals = 2 * prec_vals[:-1] * rec_vals[:-1] / (
        prec_vals[:-1] + rec_vals[:-1] + 1e-9
    )
    return thresholds[np.argmax(f1_vals)]


def threshold_for_target_recall(y_true, probs, target_recall):
    prec_vals, rec_vals, thresholds = precision_recall_curve(y_true, probs)
    valid_idx = np.where(rec_vals[:-1] >= target_recall)[0]

    if len(valid_idx) == 0:
        return thresholds[np.argmax(rec_vals[:-1])]

    best_valid_idx = valid_idx[np.argmax(prec_vals[valid_idx])]
    return thresholds[best_valid_idx]


def metrics_at_threshold(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    return {
        'threshold': threshold,
        'f1': f1_score(y_true, preds, zero_division=0),
        'accuracy': accuracy_score(y_true, preds),
        'precision': precision_score(y_true, preds, zero_division=0),
        'recall': recall_score(y_true, preds, zero_division=0),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
    }


def make_configs(base_weight):
    base_params = {
        'n_estimators': 1200,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1,
    }

    configs = [
        {
            'name': 'current_shallow_no_weight',
            **base_params,
            'learning_rate': 0.03,
            'num_leaves': 7,
            'max_depth': 3,
            'min_child_samples': 200,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
        },
        {
            'name': 'slightly_deeper_no_weight',
            **base_params,
            'learning_rate': 0.03,
            'num_leaves': 15,
            'max_depth': 4,
            'min_child_samples': 200,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
        },
        {
            'name': 'conservative_no_weight',
            **base_params,
            'learning_rate': 0.03,
            'num_leaves': 5,
            'max_depth': 2,
            'min_child_samples': 500,
            'reg_alpha': 1.0,
            'reg_lambda': 5.0,
        },
        {
            'name': 'light_weight_shallow',
            **base_params,
            'learning_rate': 0.03,
            'num_leaves': 7,
            'max_depth': 3,
            'min_child_samples': 200,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
            'scale_pos_weight': base_weight * 0.10,
        },
        {
            'name': 'quarter_weight_shallow',
            **base_params,
            'learning_rate': 0.03,
            'num_leaves': 7,
            'max_depth': 3,
            'min_child_samples': 200,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
            'scale_pos_weight': base_weight * 0.25,
        },
        {
            'name': 'slow_learning_no_weight',
            **base_params,
            'learning_rate': 0.015,
            'num_leaves': 7,
            'max_depth': 3,
            'min_child_samples': 200,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
        },
    ]

    return configs


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=" * 70)
    print("ISIC 2024 — LightGBM Experiment Runner")
    print("=" * 70)

    df_train = pd.read_csv(TRAIN_FILE)

    for col in CATEGORICAL_COLS:
        if col in df_train.columns:
            df_train[col] = LabelEncoder().fit_transform(df_train[col].astype(str))

    X = df_train.drop(columns=[c for c in DROP_COLS if c in df_train.columns])
    y = df_train['malignant'].values

    n_benign = int((y == 0).sum())
    n_malignant = int((y == 1).sum())
    base_weight = n_benign / n_malignant

    print(f"Rows       : {len(df_train):,}")
    print(f"Benign     : {n_benign:,}")
    print(f"Malignant  : {n_malignant:,}")
    print(f"Imbalance  : {base_weight:.1f}:1")
    print(f"CV folds   : {N_FOLDS}")
    print()

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    configs = make_configs(base_weight)
    rows = []

    print(f"{'Experiment':<28} {'PR-AUC':>8} {'F1':>8} {'Prec':>8} "
          f"{'Recall':>8} {'50% flags':>10} {'90% flags':>10}")
    print("-" * 96)

    for config in configs:
        name = config['name']
        params = {k: v for k, v in config.items() if k != 'name'}
        cv_probs = np.zeros(len(y))
        best_iterations = []

        for train_idx, val_idx in cv.split(X, y):
            model = LGBMClassifier(**params)
            model.fit(
                X.iloc[train_idx], y[train_idx],
                eval_set=[(X.iloc[val_idx], y[val_idx])],
                callbacks=[
                    early_stopping(stopping_rounds=50, verbose=False),
                    log_evaluation(period=-1),
                ],
            )
            cv_probs[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
            best_iterations.append(model.best_iteration_)

        pr_auc = average_precision_score(y, cv_probs)
        best_f1_threshold = threshold_for_best_f1(y, cv_probs)
        best_f1_metrics = metrics_at_threshold(y, cv_probs, best_f1_threshold)

        row = {
            'experiment': name,
            'pr_auc': pr_auc,
            'best_iteration_mean': np.mean(best_iterations),
            **{f'f1_mode_{k}': v for k, v in best_f1_metrics.items()},
        }

        for target_recall in TARGET_RECALLS:
            threshold = threshold_for_target_recall(y, cv_probs, target_recall)
            target_metrics = metrics_at_threshold(y, cv_probs, threshold)
            prefix = f'recall_{int(target_recall * 100)}'

            row[f'{prefix}_threshold'] = target_metrics['threshold']
            row[f'{prefix}_f1'] = target_metrics['f1']
            row[f'{prefix}_precision'] = target_metrics['precision']
            row[f'{prefix}_recall'] = target_metrics['recall']
            row[f'{prefix}_malignant_caught'] = target_metrics['tp']
            row[f'{prefix}_benign_flagged'] = target_metrics['fp']
            row[f'{prefix}_benign_flagged_pct'] = target_metrics['fp'] / n_benign * 100

        rows.append(row)

        print(f"{name:<28} {pr_auc:>8.4f} "
              f"{best_f1_metrics['f1']:>8.4f} "
              f"{best_f1_metrics['precision']:>8.4f} "
              f"{best_f1_metrics['recall']:>8.4f} "
              f"{row['recall_50_benign_flagged_pct']:>9.1f}% "
              f"{row['recall_90_benign_flagged_pct']:>9.1f}%")

    df_results = pd.DataFrame(rows)
    df_results = df_results.sort_values(
        ['pr_auc', 'recall_50_benign_flagged_pct'],
        ascending=[False, True],
    )

    output_path = os.path.join(REPORT_DIR, 'lgbm_experiment_results.csv')
    df_results.to_csv(output_path, index=False)

    print()
    print(f"Saved: {output_path}")
    print()
    print("Best experiments by PR-AUC:")
    display_cols = [
        'experiment',
        'pr_auc',
        'f1_mode_f1',
        'f1_mode_precision',
        'f1_mode_recall',
        'recall_50_precision',
        'recall_50_benign_flagged_pct',
        'recall_90_benign_flagged_pct',
    ]
    print(df_results[display_cols].head(5).to_string(index=False))


if __name__ == '__main__':
    main()
