# =============================================================================
# ISIC 2024 — Statistical Analysis Script
# =============================================================================
# Reads merged_cleaned.csv and datasetv3.csv (outputs of isic_data_prep.py)
#
# What this script does:
#   1. Mann-Whitney U test — confirms which features are statistically
#      different between benign and malignant groups
#   2. Effect size (rank-biserial correlation) — how LARGE the difference is
#   3. Naive Bayes (Gaussian) — assigns a malignancy probability to every
#      benign record based on its feature values
#   4. Bayesian probability profile — which features contribute most to
#      the malignancy probability
#   5. Descriptive statistics table — mean, median, std per feature per class
#   6. Saves all results as CSV reports
#
# Install once:
#   !pip install pandas numpy scikit-learn scipy matplotlib seaborn
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import mannwhitneyu, pointbiserialr
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
)
import warnings
import os
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE  = 'merged_cleaned.csv'
RISK_FILE   = 'datasetv3.csv'
REPORT_DIR  = 'reports'
GRAPH_DIR   = 'graphs'
DPI         = 150

COLOR_BENIGN    = '#2E86AB'
COLOR_MALIGNANT = '#E84855'
COLOR_BOTH      = '#F4A261'

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

COL_LABELS = {
    'age_approx':               'Age (approx)',
    'clin_size_long_diam_mm':   'Lesion Diameter (mm)',
    'tbp_lv_area_perim_ratio':  'Area-Perimeter Ratio',
    'tbp_lv_norm_border':       'Border Irregularity',
    'tbp_lv_symm_2axis':        'Symmetry Score',
    'tbp_lv_eccentricity':      'Eccentricity',
    'tbp_lv_color_std_mean':    'Color Std Mean',
    'tbp_lv_norm_color':        'Color Irregularity',
    'tbp_lv_deltaLBnorm':       'Color Contrast (normalized)',
    'tbp_lv_radial_color_std_max': 'Radial Color Std Max',
    'tbp_lv_nevi_confidence':   'Nevi Confidence',
    'color_contrast_3d':        '3D Color Contrast',
    'elongation':               'Elongation',
    'nevi_color_tension':       'Nevi-Color Tension',
    'log_area':                 'Log Area',
    'stdL_ratio':               'Lightness Ratio',
    'compactness':              'Compactness',
    'chroma_contrast':          'Chroma Contrast',
    'radial_color_ratio':       'Radial Color Ratio',
}


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
})

print("=" * 60)
print("ISIC 2024 — Statistical Analysis")
print("=" * 60)


# =============================================================================
# LOAD DATA
# =============================================================================

print(f"\nLoading {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)

CONTINUOUS_COLS = [c for c in CONTINUOUS_COLS if c in df.columns]

df_benign    = df[df['malignant'] == 0].copy()
df_malignant = df[df['malignant'] == 1].copy()

print(f"  Total rows    : {len(df):,}")
print(f"  Benign        : {len(df_benign):,}")
print(f"  Malignant     : {len(df_malignant):,}")
print(f"  Features      : {len(CONTINUOUS_COLS)}")


# =============================================================================
# ANALYSIS 1 — Descriptive Statistics
# =============================================================================

print("\n" + "=" * 60)
print("ANALYSIS 1 — Descriptive Statistics per Feature per Class")
print("=" * 60)

rows = []
for col in CONTINUOUS_COLS:
    for cls, subset in [('Benign', df_benign), ('Malignant', df_malignant)]:
        s = subset[col].dropna()
        rows.append({
            'feature':  col,
            'class':    cls,
            'mean':     round(s.mean(),   4),
            'median':   round(s.median(), 4),
            'std':      round(s.std(),    4),
            'min':      round(s.min(),    4),
            'max':      round(s.max(),    4),
            'q25':      round(s.quantile(0.25), 4),
            'q75':      round(s.quantile(0.75), 4),
        })

df_stats = pd.DataFrame(rows)
df_stats.to_csv(f'{REPORT_DIR}/descriptive_statistics.csv', index=False)

# Print summary
print(f"\n  {'Feature':<35} {'Ben Mean':>10} {'Mal Mean':>10} {'Difference':>12}")
print(f"  {'-'*70}")
for col in CONTINUOUS_COLS:
    b = df_stats[(df_stats['feature']==col) & (df_stats['class']=='Benign')].iloc[0]
    m = df_stats[(df_stats['feature']==col) & (df_stats['class']=='Malignant')].iloc[0]
    diff = m['mean'] - b['mean']
    direction = '▲' if diff > 0 else '▼'
    print(f"  {col:<35} {b['mean']:>10.3f} {m['mean']:>10.3f} {direction} {abs(diff):>10.3f}")

print(f"\n  Saved: {REPORT_DIR}/descriptive_statistics.csv")


# =============================================================================
# ANALYSIS 2 — Mann-Whitney U Test + Effect Size
# =============================================================================
# Mann-Whitney U: tests whether malignant values tend to be
# higher/lower than benign values for each feature.
# p < 0.05 = statistically significant difference.
#
# Effect size (rank-biserial r):
#   0.1 = small,  0.3 = medium,  0.5 = large

print("\n" + "=" * 60)
print("ANALYSIS 2 — Mann-Whitney U Test + Effect Size")
print("=" * 60)

mw_rows = []

print(f"\n  {'Feature':<35} {'U-stat':>12} {'p-value':>12} {'Effect r':>10} {'Size':>8} {'Sig':>6}")
print(f"  {'-'*85}")

for col in CONTINUOUS_COLS:
    ben_vals = df_benign[col].dropna().values
    mal_vals = df_malignant[col].dropna().values

    try:
        stat, p = mannwhitneyu(mal_vals, ben_vals, alternative='two-sided')

        # Rank-biserial correlation as effect size
        n1, n2  = len(mal_vals), len(ben_vals)
        effect_r = 1 - (2 * stat) / (n1 * n2)
        effect_r = abs(effect_r)

        if   effect_r >= 0.5: effect_size = 'Large'
        elif effect_r >= 0.3: effect_size = 'Medium'
        elif effect_r >= 0.1: effect_size = 'Small'
        else:                  effect_size = 'Negligible'

        significant = 'YES' if p < 0.05 else 'no'

        print(f"  {col:<35} {stat:>12.1f} {p:>12.4e} {effect_r:>10.4f} {effect_size:>8} {significant:>6}")

        mw_rows.append({
            'feature':     col,
            'label':       COL_LABELS.get(col, col),
            'u_statistic': round(stat, 2),
            'p_value':     p,
            'neg_log10_p': round(-np.log10(max(p, 1e-300)), 4),
            'effect_r':    round(effect_r, 4),
            'effect_size': effect_size,
            'significant': p < 0.05,
            'ben_mean':    round(float(df_benign[col].mean()),    4),
            'mal_mean':    round(float(df_malignant[col].mean()), 4),
            'direction':   'Higher in malignant' if df_malignant[col].mean() > df_benign[col].mean()
                           else 'Lower in malignant',
        })

    except Exception as e:
        print(f"  {col:<35} ERROR: {e}")

df_mw = pd.DataFrame(mw_rows).sort_values('p_value')
df_mw.to_csv(f'{REPORT_DIR}/mann_whitney_results.csv', index=False)

sig_count = df_mw['significant'].sum()
print(f"\n  Significant features (p < 0.05): {sig_count} / {len(CONTINUOUS_COLS)}")
print(f"  Saved: {REPORT_DIR}/mann_whitney_results.csv")

# Print top features
print(f"\n  Top 10 most significant features:")
print(f"  {'Rank':<6} {'Feature':<35} {'p-value':>12} {'Effect':>10} {'Direction'}")
print(f"  {'-'*80}")
for i, row in df_mw.head(10).iterrows():
    print(f"  {df_mw.index.get_loc(i)+1:<6} {row['feature']:<35} "
          f"{row['p_value']:>12.4e} {row['effect_size']:>10} {row['direction']}")


# =============================================================================
# ANALYSIS 2b — Mann-Whitney graph (enhanced)
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Mann-Whitney U Test Results\nFeature Significance: Benign vs Malignant',
             fontsize=14, fontweight='bold')

df_mw_plot = df_mw.copy()
df_mw_plot['label'] = df_mw_plot['feature'].map(COL_LABELS)

# Left: -log10(p-value)
colors_p = [COLOR_MALIGNANT if sig else '#CCCCCC'
            for sig in df_mw_plot['significant']]
axes[0].barh(df_mw_plot['label'], df_mw_plot['neg_log10_p'],
             color=colors_p, edgecolor='white', height=0.65)
axes[0].axvline(-np.log10(0.05), color='black', linestyle='--',
                linewidth=1.5, label='p = 0.05')
axes[0].set_xlabel('-log10(p-value)  |  Higher = more significant')
axes[0].set_title('Statistical Significance')
axes[0].legend(fontsize=9)

sig_patch   = mpatches.Patch(color=COLOR_MALIGNANT, label='Significant (p<0.05)')
insig_patch = mpatches.Patch(color='#CCCCCC', label='Not significant')
axes[0].legend(handles=[sig_patch, insig_patch,
               plt.Line2D([0],[0], color='black', linestyle='--', label='p=0.05')],
               fontsize=9)

# Right: effect size
effect_colors = {
    'Large':      '#C1121F',
    'Medium':     '#F4A261',
    'Small':      '#57CC99',
    'Negligible': '#CCCCCC',
}
colors_e = [effect_colors.get(e, '#CCCCCC') for e in df_mw_plot['effect_size']]
axes[1].barh(df_mw_plot['label'], df_mw_plot['effect_r'],
             color=colors_e, edgecolor='white', height=0.65)
axes[1].axvline(0.1, color='gray',   linestyle=':', linewidth=1, label='Small (0.1)')
axes[1].axvline(0.3, color='orange', linestyle=':', linewidth=1, label='Medium (0.3)')
axes[1].axvline(0.5, color='red',    linestyle=':', linewidth=1, label='Large (0.5)')
axes[1].set_xlabel('Effect Size (rank-biserial r)')
axes[1].set_title('Effect Size')
axes[1].legend(fontsize=9)

plt.tight_layout()
path = os.path.join(GRAPH_DIR, 'mann_whitney_results.png')
fig.savefig(path, dpi=DPI, bbox_inches='tight')
plt.close(fig)
print(f"\n  Graph saved: {path}")


# =============================================================================
# ANALYSIS 3 — Naive Bayes (Bayesian classifier)
# =============================================================================
# GaussianNB assumes each feature follows a Gaussian distribution
# per class and uses Bayes' theorem to compute:
#   P(malignant | features) for each record
#
# This gives every benign record a malignancy probability score
# that is statistically grounded — not just a feature count.

print("\n" + "=" * 60)
print("ANALYSIS 3 — Naive Bayes Malignancy Probability")
print("=" * 60)

X = df[CONTINUOUS_COLS].values
y = df['malignant'].values

# Scale features (NB works better with standardized data)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cross-validated Naive Bayes (5 folds, stratified)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

nb_probs_all = np.zeros(len(df))
nb_preds_all = np.zeros(len(df))

fold_metrics = []

print(f"\n  Running 5-fold stratified cross-validation...\n")
print(f"  {'Fold':<6} {'F1':>8} {'Accuracy':>10} {'Precision':>11} {'Recall':>8} {'ROC-AUC':>9}")
print(f"  {'-'*55}")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y[train_idx],        y[val_idx]

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    probs = gnb.predict_proba(X_val)[:, 1]
    nb_probs_all[val_idx] = probs

    # Find best threshold via precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_val, probs)
    f1_per_thr = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx   = np.argmax(f1_per_thr)
    best_thr   = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    preds = (probs >= best_thr).astype(int)
    nb_preds_all[val_idx] = preds

    f1   = f1_score(y_val, preds, zero_division=0)
    acc  = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, zero_division=0)
    rec  = recall_score(y_val, preds, zero_division=0)

    try:
        auc = roc_auc_score(y_val, probs)
    except Exception:
        auc = float('nan')

    print(f"  {fold+1:<6} {f1:>8.4f} {acc:>10.4f} {prec:>11.4f} {rec:>8.4f} {auc:>9.4f}")
    fold_metrics.append({
        'fold': fold+1, 'f1': f1, 'accuracy': acc,
        'precision': prec, 'recall': rec, 'roc_auc': auc,
        'threshold': best_thr,
    })

df_fold = pd.DataFrame(fold_metrics)

print(f"\n  {'MEAN':<6} "
      f"{df_fold['f1'].mean():>8.4f} "
      f"{df_fold['accuracy'].mean():>10.4f} "
      f"{df_fold['precision'].mean():>11.4f} "
      f"{df_fold['recall'].mean():>8.4f} "
      f"{df_fold['roc_auc'].mean():>9.4f}")
print(f"  {'STD':<6} "
      f"{df_fold['f1'].std():>8.4f} "
      f"{df_fold['accuracy'].std():>10.4f} "
      f"{df_fold['precision'].std():>11.4f} "
      f"{df_fold['recall'].std():>8.4f} "
      f"{df_fold['roc_auc'].std():>9.4f}")

df_fold.to_csv(f'{REPORT_DIR}/naive_bayes_fold_metrics.csv', index=False)
print(f"\n  Saved: {REPORT_DIR}/naive_bayes_fold_metrics.csv")


# =============================================================================
# ANALYSIS 3b — Assign NB probability to every benign record
# =============================================================================

print("\n  Assigning malignancy probability to all benign records...")

# Train final NB on the full dataset
gnb_final = GaussianNB()
gnb_final.fit(X_scaled, y)

# Get probability for every row
all_probs = gnb_final.predict_proba(X_scaled)[:, 1]

df_with_probs = df.copy()
df_with_probs['nb_malignancy_prob'] = all_probs

# Benign records only
df_benign_probs = df_with_probs[df_with_probs['malignant'] == 0].copy()
df_benign_probs = df_benign_probs.sort_values('nb_malignancy_prob', ascending=False)

print(f"\n  Malignancy probability stats for BENIGN records:")
print(f"    Mean prob   : {df_benign_probs['nb_malignancy_prob'].mean():.4f}")
print(f"    Median prob : {df_benign_probs['nb_malignancy_prob'].median():.4f}")
print(f"    Max prob    : {df_benign_probs['nb_malignancy_prob'].max():.4f}")
print(f"    Min prob    : {df_benign_probs['nb_malignancy_prob'].min():.4f}")

# How many benign records have >50% malignancy probability
high_risk = (df_benign_probs['nb_malignancy_prob'] >= 0.50).sum()
print(f"\n  Benign records with prob >= 0.50 : {high_risk:,}")
print(f"  Benign records with prob >= 0.30 : {(df_benign_probs['nb_malignancy_prob'] >= 0.30).sum():,}")
print(f"  Benign records with prob >= 0.10 : {(df_benign_probs['nb_malignancy_prob'] >= 0.10).sum():,}")

print(f"\n  Top 15 highest-risk benign records (by NB probability):")
top15_cols = ['nb_malignancy_prob'] + CONTINUOUS_COLS[:5]
top15_cols = [c for c in top15_cols if c in df_benign_probs.columns]
print(df_benign_probs[top15_cols].head(15).to_string(index=False))

df_benign_probs.to_csv(f'{REPORT_DIR}/benign_malignancy_probabilities.csv', index=False)
print(f"\n  Saved: {REPORT_DIR}/benign_malignancy_probabilities.csv")
print(f"  (All benign records sorted by malignancy probability — highest first)")


# =============================================================================
# ANALYSIS 3c — Confusion matrix for Naive Bayes
# =============================================================================

# Use the cross-validated predictions
cm = confusion_matrix(y, nb_preds_all)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Naive Bayes — Cross-Validated Performance',
             fontsize=14, fontweight='bold')

# Confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Benign', 'Malignant'])
disp.plot(ax=axes[0], colorbar=False,
          cmap='Blues', values_format='d')
axes[0].set_title('Confusion Matrix\n(5-fold cross-validated)')

# Metrics bar chart
metrics_mean = {
    'F1':        df_fold['f1'].mean(),
    'Accuracy':  df_fold['accuracy'].mean(),
    'Precision': df_fold['precision'].mean(),
    'Recall':    df_fold['recall'].mean(),
    'ROC-AUC':   df_fold['roc_auc'].mean(),
}
metrics_std = {
    'F1':        df_fold['f1'].std(),
    'Accuracy':  df_fold['accuracy'].std(),
    'Precision': df_fold['precision'].std(),
    'Recall':    df_fold['recall'].std(),
    'ROC-AUC':   df_fold['roc_auc'].std(),
}

bars = axes[1].bar(
    metrics_mean.keys(),
    metrics_mean.values(),
    color=[COLOR_BENIGN, '#57CC99', COLOR_MALIGNANT, COLOR_BOTH, '#A23B72'],
    edgecolor='white',
    width=0.55,
)
axes[1].errorbar(
    range(len(metrics_mean)),
    list(metrics_mean.values()),
    yerr=list(metrics_std.values()),
    fmt='none', color='black', capsize=5, linewidth=1.5,
)
axes[1].set_ylim(0, 1.05)
axes[1].set_title('Mean Metrics (±std across 5 folds)')
axes[1].set_ylabel('Score')
for bar, val in zip(bars, metrics_mean.values()):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
path = os.path.join(GRAPH_DIR, 'naive_bayes_performance.png')
fig.savefig(path, dpi=DPI, bbox_inches='tight')
plt.close(fig)
print(f"\n  Graph saved: {path}")


# =============================================================================
# ANALYSIS 3d — NB probability distribution graph
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Naive Bayes Malignancy Probability\nDistribution for Benign Records',
             fontsize=14, fontweight='bold')

# Histogram of probabilities
sns.histplot(df_benign_probs['nb_malignancy_prob'],
             bins=60, kde=True, ax=axes[0],
             color=COLOR_BENIGN, edgecolor='white', linewidth=0.3)
axes[0].axvline(0.5, color='red',    linestyle='--', linewidth=1.5, label='Threshold 0.50')
axes[0].axvline(0.3, color='orange', linestyle='--', linewidth=1.5, label='Threshold 0.30')
axes[0].set_title('Probability Distribution\n(benign records only)')
axes[0].set_xlabel('P(malignant | features)')
axes[0].set_ylabel('Count')
axes[0].legend(fontsize=9)

# CDF
sorted_p = np.sort(df_benign_probs['nb_malignancy_prob'].values)
cdf      = np.arange(1, len(sorted_p)+1) / len(sorted_p)
axes[1].plot(sorted_p, cdf, color=COLOR_BENIGN, linewidth=2)
axes[1].axvline(0.5, color='red',    linestyle='--', linewidth=1.5, label='0.50')
axes[1].axvline(0.3, color='orange', linestyle='--', linewidth=1.5, label='0.30')
axes[1].set_title('Cumulative Distribution')
axes[1].set_xlabel('P(malignant | features)')
axes[1].set_ylabel('Proportion of benign records')
axes[1].legend(fontsize=9)

plt.tight_layout()
path = os.path.join(GRAPH_DIR, 'nb_probability_distribution.png')
fig.savefig(path, dpi=DPI, bbox_inches='tight')
plt.close(fig)
print(f"  Graph saved: {path}")


# =============================================================================
# ANALYSIS 4 — Feature contribution to malignancy (log-likelihood ratio)
# =============================================================================
# For each feature, compute how much each class distribution differs.
# This shows which features the Naive Bayes model relies on most.

print("\n" + "=" * 60)
print("ANALYSIS 4 — Feature Contribution to Malignancy Probability")
print("=" * 60)

contrib_rows = []

for i, col in enumerate(CONTINUOUS_COLS):
    # NB stores theta (mean) and sigma (std) per class per feature
    ben_mean = gnb_final.theta_[0][i]
    mal_mean = gnb_final.theta_[1][i]
    ben_std  = np.sqrt(gnb_final.var_[0][i])
    mal_std  = np.sqrt(gnb_final.var_[1][i])

    # Mean separation in units of pooled std (like Cohen's d)
    pooled_std = np.sqrt((ben_std**2 + mal_std**2) / 2)
    cohens_d   = abs(mal_mean - ben_mean) / (pooled_std + 1e-9)

    contrib_rows.append({
        'feature':       col,
        'label':         COL_LABELS.get(col, col),
        'nb_ben_mean':   round(ben_mean, 4),
        'nb_mal_mean':   round(mal_mean, 4),
        'nb_ben_std':    round(ben_std,  4),
        'nb_mal_std':    round(mal_std,  4),
        'cohens_d':      round(cohens_d, 4),
        'direction':     'Higher in malignant' if mal_mean > ben_mean
                         else 'Lower in malignant',
    })

df_contrib = pd.DataFrame(contrib_rows).sort_values('cohens_d', ascending=False)
df_contrib.to_csv(f'{REPORT_DIR}/feature_contributions.csv', index=False)

print(f"\n  {'Rank':<6} {'Feature':<35} {'Cohen d':>9} {'Direction'}")
print(f"  {'-'*70}")
for rank, (_, row) in enumerate(df_contrib.iterrows(), 1):
    print(f"  {rank:<6} {row['feature']:<35} {row['cohens_d']:>9.4f}  {row['direction']}")

print(f"\n  Saved: {REPORT_DIR}/feature_contributions.csv")

# Graph: Cohen's d per feature
fig, ax = plt.subplots(figsize=(12, 7))

colors_d = [COLOR_MALIGNANT if d >= 0.5 else
            COLOR_BOTH      if d >= 0.3 else
            COLOR_BENIGN    if d >= 0.1 else '#CCCCCC'
            for d in df_contrib['cohens_d']]

ax.barh(df_contrib['label'], df_contrib['cohens_d'],
        color=colors_d, edgecolor='white', height=0.65)
ax.axvline(0.2, color='steelblue', linestyle=':', linewidth=1, label="Small (0.2)")
ax.axvline(0.5, color='orange',    linestyle=':', linewidth=1, label="Medium (0.5)")
ax.axvline(0.8, color='red',       linestyle=':', linewidth=1, label="Large (0.8)")
ax.set_xlabel("Cohen's d  |  Higher = stronger separation between classes")
ax.set_title("Feature Contribution to Malignancy\n(Cohen's d from Naive Bayes class distributions)",
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)

plt.tight_layout()
path = os.path.join(GRAPH_DIR, 'feature_contributions.png')
fig.savefig(path, dpi=DPI, bbox_inches='tight')
plt.close(fig)
print(f"  Graph saved: {path}")


# =============================================================================
# ANALYSIS 5 — Point-biserial correlation
# =============================================================================
# Measures linear correlation between each continuous feature and
# the binary malignant label. Simpler than Mann-Whitney but complementary.

print("\n" + "=" * 60)
print("ANALYSIS 5 — Point-Biserial Correlation with Malignant Label")
print("=" * 60)

corr_rows = []

print(f"\n  {'Feature':<35} {'Correlation r':>15} {'p-value':>12} {'Strength'}")
print(f"  {'-'*70}")

for col in CONTINUOUS_COLS:
    try:
        r, p = pointbiserialr(df[col].dropna(),
                               df.loc[df[col].notna(), 'malignant'])
        strength = ('Strong' if abs(r) >= 0.3 else
                    'Moderate' if abs(r) >= 0.1 else 'Weak')
        direction = '+' if r > 0 else '-'
        print(f"  {col:<35} {r:>+15.4f} {p:>12.4e}  {direction} {strength}")
        corr_rows.append({'feature': col, 'label': COL_LABELS.get(col,col),
                          'r': round(r,4), 'p_value': p, 'strength': strength})
    except Exception as e:
        print(f"  {col:<35} ERROR: {e}")

df_corr = pd.DataFrame(corr_rows).sort_values('r', key=abs, ascending=False)
df_corr.to_csv(f'{REPORT_DIR}/pointbiserial_correlations.csv', index=False)
print(f"\n  Saved: {REPORT_DIR}/pointbiserial_correlations.csv")


# =============================================================================
# ANALYSIS 6 — Combined ranking of all features
# =============================================================================
# Merges Mann-Whitney significance, effect size, Cohen's d, and
# point-biserial correlation into one final ranking table.

print("\n" + "=" * 60)
print("ANALYSIS 6 — Final Feature Ranking (Combined)")
print("=" * 60)

df_rank = df_mw[['feature', 'label', 'p_value', 'neg_log10_p',
                  'effect_r', 'effect_size', 'significant', 'direction']].copy()

df_rank = df_rank.merge(
    df_contrib[['feature', 'cohens_d']],
    on='feature', how='left'
)
df_rank = df_rank.merge(
    df_corr[['feature', 'r', 'strength']].rename(
        columns={'r': 'pb_corr', 'strength': 'corr_strength'}),
    on='feature', how='left'
)

# Composite score: normalized sum of effect_r, cohens_d, and abs(pb_corr)
df_rank['composite_score'] = (
    df_rank['effect_r'].fillna(0)   / df_rank['effect_r'].max()   * 0.4 +
    df_rank['cohens_d'].fillna(0)   / df_rank['cohens_d'].max()   * 0.4 +
    df_rank['pb_corr'].fillna(0).abs() / df_rank['pb_corr'].abs().max() * 0.2
)
df_rank = df_rank.sort_values('composite_score', ascending=False)
df_rank['rank'] = range(1, len(df_rank)+1)

df_rank.to_csv(f'{REPORT_DIR}/final_feature_ranking.csv', index=False)

print(f"\n  {'Rank':<6} {'Feature':<35} {'Score':>8} {'Effect':>8} {'p-value':>12} {'Direction'}")
print(f"  {'-'*80}")
for _, row in df_rank.iterrows():
    print(f"  {int(row['rank']):<6} {row['feature']:<35} "
          f"{row['composite_score']:>8.4f} {row['effect_size']:>8} "
          f"{row['p_value']:>12.4e}  {row['direction']}")

print(f"\n  Saved: {REPORT_DIR}/final_feature_ranking.csv")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("ALL DONE — Reports and Graphs Saved")
print("=" * 60)

print(f"\n  Reports saved to '{REPORT_DIR}/':")
print(f"    descriptive_statistics.csv      — mean/median/std per feature per class")
print(f"    mann_whitney_results.csv        — U-statistic, p-value, effect size")
print(f"    naive_bayes_fold_metrics.csv    — F1/accuracy/precision/recall per fold")
print(f"    benign_malignancy_probabilities.csv — every benign record with NB prob")
print(f"    feature_contributions.csv       — Cohen's d per feature")
print(f"    pointbiserial_correlations.csv  — correlation with malignant label")
print(f"    final_feature_ranking.csv       — combined ranking of all features")

print(f"\n  Graphs saved to '{GRAPH_DIR}/':")
print(f"    mann_whitney_results.png        — significance + effect size chart")
print(f"    naive_bayes_performance.png     — confusion matrix + metric bars")
print(f"    nb_probability_distribution.png — distribution of NB probabilities")
print(f"    feature_contributions.png       — Cohen's d ranking chart")

print(f"\n  Key findings to report:")
print(f"    Significant features  : {df_mw['significant'].sum()} / {len(CONTINUOUS_COLS)}")
print(f"    Top feature           : {df_rank.iloc[0]['feature']} (score={df_rank.iloc[0]['composite_score']:.4f})")
print(f"    NB Mean F1            : {df_fold['f1'].mean():.4f}")
print(f"    NB Mean Recall        : {df_fold['recall'].mean():.4f}")
print(f"    High-risk benign (≥0.50 prob): {high_risk:,}")
print("=" * 60)
