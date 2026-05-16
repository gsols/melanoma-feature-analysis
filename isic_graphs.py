# =============================================================================
# ISIC 2024 — Graphing Script
# =============================================================================
# Reads merged_cleaned.csv (output of isic_data_prep.py)
# Produces graphs for every feature, saved as PNG in a graphs/ folder.
#
# Graphs produced:
#   - Histogram pair (benign vs malignant) for every continuous feature
#   - Box plot for every continuous feature
#   - Pie chart pair for every categorical feature
#   - Grouped bar chart for every categorical feature
#   - Correlation heatmap
#   - Feature range overlap bar chart
#   - Risk score distribution (from datasetv3.csv)
#
# Install once:
#   !pip install pandas numpy matplotlib seaborn scipy
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import mannwhitneyu
import os
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE   = 'merged_cleaned.csv'    # output from isic_data_prep.py
RISK_FILE    = 'datasetv3.csv'         # has risk_score column
GRAPH_DIR    = 'graphs'                # folder where all PNGs are saved
DPI          = 150                     # image quality (150 = good balance)

# Colors — consistent across all graphs
COLOR_BENIGN    = '#2E86AB'    # blue
COLOR_MALIGNANT = '#E84855'    # red
COLOR_BOTH      = '#F4A261'    # orange (for combined views)

# Continuous features to graph
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

# Categorical features to graph
CATEGORICAL_COLS = [
    'sex',
    'anatom_site_general',
    'tbp_lv_location_simple',
]

# Human-readable labels for graph titles
COL_LABELS = {
    'age_approx':               'Age (approx)',
    'clin_size_long_diam_mm':   'Lesion Diameter (mm)',
    'tbp_lv_area_perim_ratio':  'Area-Perimeter Ratio',
    'tbp_lv_norm_border':       'Border Irregularity (normalized)',
    'tbp_lv_symm_2axis':        'Symmetry Score (2-axis)',
    'tbp_lv_eccentricity':      'Eccentricity',
    'tbp_lv_color_std_mean':    'Color Std Mean (within lesion)',
    'tbp_lv_norm_color':        'Color Irregularity (normalized)',
    'tbp_lv_deltaLBnorm':       'Color Contrast vs Skin (normalized)',
    'tbp_lv_radial_color_std_max': 'Radial Color Std Max',
    'tbp_lv_nevi_confidence':   'Nevi Confidence Score',
    'color_contrast_3d':        '3D Color Contrast (engineered)',
    'elongation':               'Lesion Elongation (engineered)',
    'nevi_color_tension':       'Nevi-Color Tension (engineered)',
    'log_area':                 'Log Area (engineered)',
    'stdL_ratio':               'Lightness Ratio (engineered)',
    'compactness':              'Shape Compactness (engineered)',
    'chroma_contrast':          'Chroma Contrast (engineered)',
    'radial_color_ratio':       'Radial Color Ratio (engineered)',
    'sex':                      'Sex',
    'anatom_site_general':      'Anatomic Site (General)',
    'tbp_lv_location_simple':   'Body Location (Simplified)',
}


# =============================================================================
# SETUP
# =============================================================================

os.makedirs(GRAPH_DIR, exist_ok=True)
sns.set_theme(style='whitegrid', font='Arial')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'savefig.facecolor':'white',
    'font.family':      'sans-serif',
    'axes.titlesize':   13,
    'axes.labelsize':   11,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
})

print("=" * 55)
print("ISIC 2024 — Graphing Script")
print("=" * 55)


# =============================================================================
# LOAD DATA
# =============================================================================

print(f"\nLoading {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)

# Filter to only columns we need that actually exist
CONTINUOUS_COLS = [c for c in CONTINUOUS_COLS if c in df.columns]
CATEGORICAL_COLS = [c for c in CATEGORICAL_COLS if c in df.columns]

df_benign    = df[df['malignant'] == 0].copy()
df_malignant = df[df['malignant'] == 1].copy()

print(f"  Total rows    : {len(df):,}")
print(f"  Benign        : {len(df_benign):,}")
print(f"  Malignant     : {len(df_malignant):,}")
print(f"  Continuous    : {len(CONTINUOUS_COLS)} features")
print(f"  Categorical   : {len(CATEGORICAL_COLS)} features")
print(f"  Output folder : {GRAPH_DIR}/")
print()

saved_files = []


# =============================================================================
# HELPER — save figure
# =============================================================================

def save(fig, filename):
    path = os.path.join(GRAPH_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(filename)
    print(f"  Saved: {filename}")


# =============================================================================
# GRAPH TYPE 1 — Histogram pairs (benign vs malignant)
# One graph per continuous feature, two panels side by side
# =============================================================================

print("-" * 55)
print("Generating histograms...")
print("-" * 55)

for col in CONTINUOUS_COLS:
    label = COL_LABELS.get(col, col)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f'{label}\nBenign vs Malignant Distribution',
                 fontsize=14, fontweight='bold', y=1.02)

    # Benign panel
    sns.histplot(
        df_benign[col].dropna(),
        kde=True,
        ax=axes[0],
        color=COLOR_BENIGN,
        alpha=0.75,
        bins=40,
        edgecolor='white',
        linewidth=0.3,
    )
    axes[0].set_title(f'Benign  (n={len(df_benign):,})',
                      color=COLOR_BENIGN, fontweight='bold')
    axes[0].set_xlabel(label)
    axes[0].set_ylabel('Count')

    # Add mean line
    b_mean = df_benign[col].mean()
    axes[0].axvline(b_mean, color='navy', linestyle='--',
                    linewidth=1.5, label=f'Mean: {b_mean:.2f}')
    axes[0].legend(fontsize=9)

    # Malignant panel
    sns.histplot(
        df_malignant[col].dropna(),
        kde=True,
        ax=axes[1],
        color=COLOR_MALIGNANT,
        alpha=0.75,
        bins=40,
        edgecolor='white',
        linewidth=0.3,
    )
    axes[1].set_title(f'Malignant  (n={len(df_malignant):,})',
                      color=COLOR_MALIGNANT, fontweight='bold')
    axes[1].set_xlabel(label)
    axes[1].set_ylabel('Count')

    # Add mean line
    m_mean = df_malignant[col].mean()
    axes[1].axvline(m_mean, color='darkred', linestyle='--',
                    linewidth=1.5, label=f'Mean: {m_mean:.2f}')
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    save(fig, f'hist_{col}.png')


# =============================================================================
# GRAPH TYPE 2 — Box plots (benign vs malignant side by side)
# One graph per continuous feature
# =============================================================================

print("\n" + "-" * 55)
print("Generating box plots...")
print("-" * 55)

for col in CONTINUOUS_COLS:
    label = COL_LABELS.get(col, col)

    # Build a combined dataframe with a Class column
    df_plot = pd.concat([
        df_benign[col].rename('value').to_frame().assign(Class='Benign'),
        df_malignant[col].rename('value').to_frame().assign(Class='Malignant'),
    ], ignore_index=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.boxplot(
        data=df_plot,
        x='Class',
        y='value',
        palette={'Benign': COLOR_BENIGN, 'Malignant': COLOR_MALIGNANT},
        width=0.5,
        flierprops=dict(marker='o', markersize=2, alpha=0.3),
        ax=ax,
    )

    ax.set_title(f'{label}\nBox Plot: Benign vs Malignant',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Class')
    ax.set_ylabel(label)

    # Annotate medians
    for i, cls in enumerate(['Benign', 'Malignant']):
        med = df_plot[df_plot['Class'] == cls]['value'].median()
        ax.text(i, med, f' {med:.2f}', va='center',
                fontsize=9, color='white', fontweight='bold')

    plt.tight_layout()
    save(fig, f'box_{col}.png')


# =============================================================================
# GRAPH TYPE 3 — Overlapping histogram (both classes on same axes)
# Easier to see overlap between benign and malignant distributions
# =============================================================================

print("\n" + "-" * 55)
print("Generating overlap histograms...")
print("-" * 55)

for col in CONTINUOUS_COLS:
    label = COL_LABELS.get(col, col)

    fig, ax = plt.subplots(figsize=(9, 4))

    # Use density=True so the scales are comparable despite different n
    ax.hist(df_benign[col].dropna(), bins=60, density=True,
            alpha=0.5, color=COLOR_BENIGN, label='Benign', edgecolor='none')
    ax.hist(df_malignant[col].dropna(), bins=60, density=True,
            alpha=0.7, color=COLOR_MALIGNANT, label='Malignant', edgecolor='none')

    # KDE lines
    df_benign[col].dropna().plot.kde(ax=ax, color=COLOR_BENIGN,
                                     linewidth=2, label='_nolegend_')
    df_malignant[col].dropna().plot.kde(ax=ax, color=COLOR_MALIGNANT,
                                        linewidth=2, label='_nolegend_')

    # Mann-Whitney p-value
    try:
        _, p = mannwhitneyu(
            df_malignant[col].dropna(),
            df_benign[col].dropna(),
            alternative='two-sided'
        )
        p_text = f'p < 0.001' if p < 0.001 else f'p = {p:.4f}'
        sig    = '★ Significant' if p < 0.05 else 'Not significant'
        ax.set_title(f'{label}\nOverlap: Benign vs Malignant  |  {p_text}  ({sig})',
                     fontsize=12, fontweight='bold')
    except Exception:
        ax.set_title(f'{label}\nOverlap: Benign vs Malignant', fontsize=12, fontweight='bold')

    ax.set_xlabel(label)
    ax.set_ylabel('Density')
    ax.legend(fontsize=10)

    plt.tight_layout()
    save(fig, f'overlap_{col}.png')


# =============================================================================
# GRAPH TYPE 4 — Pie charts for categorical features
# =============================================================================

print("\n" + "-" * 55)
print("Generating pie charts...")
print("-" * 55)

PIE_COLORS = [
    '#2E86AB', '#E84855', '#F4A261', '#A23B72',
    '#57CC99', '#F9C74F', '#577590', '#C77DFF',
    '#80B918', '#FF6B6B',
]

for col in CATEGORICAL_COLS:
    label = COL_LABELS.get(col, col)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'{label}\nProportion: Benign vs Malignant',
                 fontsize=14, fontweight='bold')

    for ax, (subset, title, color) in zip(axes, [
        (df_benign,    f'Benign  (n={len(df_benign):,})',    COLOR_BENIGN),
        (df_malignant, f'Malignant  (n={len(df_malignant):,})', COLOR_MALIGNANT),
    ]):
        counts = subset[col].value_counts()
        wedge_colors = PIE_COLORS[:len(counts)]

        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=counts.index,
            autopct='%1.1f%%',
            colors=wedge_colors,
            startangle=140,
            pctdistance=0.82,
            wedgeprops=dict(edgecolor='white', linewidth=1.5),
        )
        for text in texts:
            text.set_fontsize(9)
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_fontweight('bold')

        ax.set_title(title, color=color, fontweight='bold', pad=12)

    plt.tight_layout()
    save(fig, f'pie_{col}.png')


# =============================================================================
# GRAPH TYPE 5 — Grouped bar charts for categorical features
# =============================================================================

print("\n" + "-" * 55)
print("Generating grouped bar charts...")
print("-" * 55)

for col in CATEGORICAL_COLS:
    label = COL_LABELS.get(col, col)

    # Count per category per class
    ben_counts = df_benign[col].value_counts().sort_index()
    mal_counts = df_malignant[col].value_counts().sort_index()

    # Align categories
    all_cats = sorted(set(ben_counts.index) | set(mal_counts.index))
    ben_vals = [ben_counts.get(c, 0) for c in all_cats]
    mal_vals = [mal_counts.get(c, 0) for c in all_cats]

    x     = np.arange(len(all_cats))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{label}\nGrouped Bar Chart: Benign vs Malignant',
                 fontsize=14, fontweight='bold')

    # Raw counts
    axes[0].bar(x - width/2, ben_vals, width, label='Benign',
                color=COLOR_BENIGN, alpha=0.85, edgecolor='white')
    axes[0].bar(x + width/2, mal_vals, width, label='Malignant',
                color=COLOR_MALIGNANT, alpha=0.85, edgecolor='white')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(all_cats, rotation=30, ha='right', fontsize=9)
    axes[0].set_title('Raw Counts')
    axes[0].set_ylabel('Count')
    axes[0].legend()

    # Percentage within each class (normalised)
    ben_pct = [v / len(df_benign)    * 100 for v in ben_vals]
    mal_pct = [v / len(df_malignant) * 100 for v in mal_vals]

    axes[1].bar(x - width/2, ben_pct, width, label='Benign',
                color=COLOR_BENIGN, alpha=0.85, edgecolor='white')
    axes[1].bar(x + width/2, mal_pct, width, label='Malignant',
                color=COLOR_MALIGNANT, alpha=0.85, edgecolor='white')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(all_cats, rotation=30, ha='right', fontsize=9)
    axes[1].set_title('Percentage Within Class (%)')
    axes[1].set_ylabel('% of class')
    axes[1].legend()

    plt.tight_layout()
    save(fig, f'bar_{col}.png')


# =============================================================================
# GRAPH TYPE 6 — Correlation heatmap
# =============================================================================

print("\n" + "-" * 55)
print("Generating correlation heatmap...")
print("-" * 55)

fig, ax = plt.subplots(figsize=(16, 13))

corr_cols = CONTINUOUS_COLS + ['malignant']
corr      = df[corr_cols].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))   # upper triangle mask

sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='RdBu_r',
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    linecolor='white',
    ax=ax,
    annot_kws={'size': 7},
    cbar_kws={'shrink': 0.8},
)

ax.set_title('Feature Correlation Heatmap\n(includes malignant label)',
             fontsize=14, fontweight='bold', pad=16)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

plt.tight_layout()
save(fig, 'correlation_heatmap.png')


# =============================================================================
# GRAPH TYPE 7 — Mean comparison bar chart (all features at once)
# =============================================================================

print("\n" + "-" * 55)
print("Generating mean comparison chart...")
print("-" * 55)

# Normalize each feature to 0-1 so they can be plotted on one axis
from sklearn.preprocessing import MinMaxScaler

scaler   = MinMaxScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df[CONTINUOUS_COLS]),
    columns=CONTINUOUS_COLS
)
df_scaled['malignant'] = df['malignant'].values

ben_means = df_scaled[df_scaled['malignant']==0][CONTINUOUS_COLS].mean()
mal_means = df_scaled[df_scaled['malignant']==1][CONTINUOUS_COLS].mean()

x     = np.arange(len(CONTINUOUS_COLS))
width = 0.38

fig, ax = plt.subplots(figsize=(16, 6))

ax.bar(x - width/2, ben_means, width, label='Benign',
       color=COLOR_BENIGN, alpha=0.85, edgecolor='white')
ax.bar(x + width/2, mal_means, width, label='Malignant',
       color=COLOR_MALIGNANT, alpha=0.85, edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(
    [COL_LABELS.get(c, c) for c in CONTINUOUS_COLS],
    rotation=45, ha='right', fontsize=8
)
ax.set_ylabel('Normalized Mean (0–1)')
ax.set_title('Normalized Feature Means: Benign vs Malignant\n(all features on same scale)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0, 1)

plt.tight_layout()
save(fig, 'mean_comparison_all_features.png')


# =============================================================================
# GRAPH TYPE 8 — Statistical significance chart (Mann-Whitney p-values)
# =============================================================================

print("\n" + "-" * 55)
print("Generating statistical significance chart...")
print("-" * 55)

p_values  = {}
for col in CONTINUOUS_COLS:
    try:
        _, p = mannwhitneyu(
            df_malignant[col].dropna(),
            df_benign[col].dropna(),
            alternative='two-sided'
        )
        p_values[col] = p
    except Exception:
        p_values[col] = 1.0

p_series = pd.Series(p_values).sort_values()
neg_log_p = -np.log10(p_series.clip(1e-300))   # -log10(p) for readability

colors = [COLOR_MALIGNANT if p < 0.05 else '#AAAAAA' for p in p_series]

fig, ax = plt.subplots(figsize=(13, 6))

bars = ax.barh(
    [COL_LABELS.get(c, c) for c in p_series.index],
    neg_log_p.values,
    color=colors,
    edgecolor='white',
    height=0.65,
)

# Significance threshold line
ax.axvline(-np.log10(0.05), color='black', linestyle='--',
           linewidth=1.5, label='p = 0.05 threshold')

ax.set_xlabel('-log10(p-value)  |  Higher = more significant')
ax.set_title('Feature Statistical Significance\n(Mann-Whitney U test, Benign vs Malignant)',
             fontsize=13, fontweight='bold')

sig_patch   = mpatches.Patch(color=COLOR_MALIGNANT, label='Significant (p < 0.05)')
insig_patch = mpatches.Patch(color='#AAAAAA',       label='Not significant')
ax.legend(handles=[sig_patch, insig_patch, ax.get_lines()[0]], fontsize=9)

plt.tight_layout()
save(fig, 'statistical_significance.png')


# =============================================================================
# GRAPH TYPE 9 — Risk score distribution (benign records only)
# =============================================================================

print("\n" + "-" * 55)
print("Generating risk score distribution...")
print("-" * 55)

try:
    df_v3      = pd.read_csv(RISK_FILE)
    df_ben_v3  = df_v3[df_v3['malignant'] == 0].dropna(subset=['risk_score'])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle('Malignancy Risk Score Distribution\n(Benign Records Only)',
                 fontsize=14, fontweight='bold')

    # Histogram
    sns.histplot(df_ben_v3['risk_score'], bins=50, kde=True,
                 ax=axes[0], color=COLOR_BOTH, edgecolor='white', linewidth=0.3)
    axes[0].set_title('Risk Score Histogram')
    axes[0].set_xlabel('Risk Score (0 = low risk, 1 = high risk)')
    axes[0].set_ylabel('Count')
    mean_risk = df_ben_v3['risk_score'].mean()
    axes[0].axvline(mean_risk, color='darkred', linestyle='--',
                    linewidth=1.5, label=f'Mean: {mean_risk:.3f}')
    axes[0].legend()

    # Cumulative distribution
    sorted_scores = np.sort(df_ben_v3['risk_score'])
    cdf = np.arange(1, len(sorted_scores)+1) / len(sorted_scores)
    axes[1].plot(sorted_scores, cdf, color=COLOR_BOTH, linewidth=2)
    axes[1].set_title('Cumulative Distribution of Risk Scores')
    axes[1].set_xlabel('Risk Score')
    axes[1].set_ylabel('Cumulative Proportion')
    axes[1].axvline(0.9, color='darkred', linestyle='--',
                    linewidth=1, label='Score = 0.9')
    axes[1].legend()

    plt.tight_layout()
    save(fig, 'risk_score_distribution.png')

except FileNotFoundError:
    print(f"  SKIP: {RISK_FILE} not found — run isic_data_prep.py first")


# =============================================================================
# GRAPH TYPE 10 — Top 5 most discriminating features (violin plots)
# =============================================================================

print("\n" + "-" * 55)
print("Generating violin plots for top features...")
print("-" * 55)

# Use the top 5 most significant features from the p-value analysis
top5 = list(p_series.head(5).index)

fig, axes = plt.subplots(1, len(top5), figsize=(16, 5))
fig.suptitle('Top 5 Most Statistically Significant Features\n(Violin Plots)',
             fontsize=14, fontweight='bold')

if len(top5) == 1:
    axes = [axes]

for ax, col in zip(axes, top5):
    label = COL_LABELS.get(col, col)

    df_vio = pd.concat([
        df_benign[col].rename('value').to_frame().assign(Class='Benign'),
        df_malignant[col].rename('value').to_frame().assign(Class='Malignant'),
    ], ignore_index=True)

    sns.violinplot(
        data=df_vio,
        x='Class',
        y='value',
        palette={'Benign': COLOR_BENIGN, 'Malignant': COLOR_MALIGNANT},
        inner='box',
        ax=ax,
    )
    ax.set_title(label, fontsize=9, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')

plt.tight_layout()
save(fig, 'violin_top5_features.png')


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 55)
print("ALL GRAPHS SAVED")
print("=" * 55)
print(f"\n  Output folder : {GRAPH_DIR}/")
print(f"  Total files   : {len(saved_files)}\n")

categories = {
    'Histograms (benign vs malignant)':  [f for f in saved_files if f.startswith('hist_')],
    'Box plots':                          [f for f in saved_files if f.startswith('box_')],
    'Overlap histograms':                 [f for f in saved_files if f.startswith('overlap_')],
    'Pie charts':                         [f for f in saved_files if f.startswith('pie_')],
    'Grouped bar charts':                 [f for f in saved_files if f.startswith('bar_')],
    'Summary charts':                     [f for f in saved_files if not any(
                                            f.startswith(p) for p in
                                            ['hist_','box_','overlap_','pie_','bar_'])],
}

for cat, files in categories.items():
    if files:
        print(f"  {cat} ({len(files)}):")
        for f in files:
            print(f"    {f}")
        print()

print("  Download from Colab:")
print("  Right-click each file in the Files panel → Download")
print("  Or run the cell below to zip everything at once:")
print()
print("  import shutil")
print("  shutil.make_archive('all_graphs', 'zip', 'graphs')")
print("  from google.colab import files")
print("  files.download('all_graphs.zip')")
print("=" * 55)
