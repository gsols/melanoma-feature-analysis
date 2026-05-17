"""
Statistical Analysis: Do Malignant and Benign Lesions Differ?

This script performs comprehensive statistical tests to determine if
malignant and benign records have significantly different feature distributions.

Tests performed:
1. Mann-Whitney U test (non-parametric, robust)
2. Effect size (Cohen's d) - practical significance
3. Feature overlap analysis - how much do distributions overlap?
4. Visualization - see the actual distributions

Usage:
    python statistical_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SETUP
# =============================================================================

TRAIN_FILE = 'train_final.csv'
REPORT_DIR = 'reports'
GRAPH_DIR = 'graphs'

print("=" * 80)
print("STATISTICAL ANALYSIS: Malignant vs Benign Distributions")
print("=" * 80)

# Load data
print("\nLoading training data...")
df = pd.read_csv(TRAIN_FILE)

# Separate malignant and benign
mal = df[df['malignant'] == 1]
ben = df[df['malignant'] == 0]

print(f"  Malignant records: {len(mal)}")
print(f"  Benign records:    {len(ben)}")
print(f"  Imbalance ratio:   {len(ben)/len(mal):.0f}:1")

# =============================================================================
# STATISTICAL TESTS
# =============================================================================

print("\n" + "=" * 80)
print("MANN-WHITNEY U TEST (Are distributions different?)")
print("=" * 80)
print("\nInterpretation:")
print("  p-value < 0.05 → Distributions ARE significantly different")
print("  p-value ≥ 0.05 → Distributions are NOT significantly different")
print("  Effect size: |d| < 0.2 (small), 0.2-0.8 (medium), > 0.8 (large)")

# Get numeric features (exclude categorical and target)
exclude_cols = {'malignant', 'risk_score', 'sex', 'anatom_site_general',
                'tbp_lv_location_simple'}
numeric_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype != 'object']

results = []

for col in sorted(numeric_cols):
    mal_vals = mal[col].dropna()
    ben_vals = ben[col].dropna()

    if len(mal_vals) == 0 or len(ben_vals) == 0:
        continue

    # Mann-Whitney U test
    stat, pval = stats.mannwhitneyu(mal_vals, ben_vals, alternative='two-sided')

    # Cohen's d (effect size)
    mean_diff = mal_vals.mean() - ben_vals.mean()
    pooled_std = np.sqrt(((len(mal_vals)-1)*mal_vals.std()**2 +
                          (len(ben_vals)-1)*ben_vals.std()**2) /
                         (len(mal_vals) + len(ben_vals) - 2))
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

    # Median and IQR
    mal_median = mal_vals.median()
    ben_median = ben_vals.median()
    mal_iqr = mal_vals.quantile(0.75) - mal_vals.quantile(0.25)
    ben_iqr = ben_vals.quantile(0.75) - ben_vals.quantile(0.25)

    # Overlap percentage (how much do distributions overlap?)
    mal_min, mal_max = mal_vals.min(), mal_vals.max()
    ben_min, ben_max = ben_vals.min(), ben_vals.max()
    overlap_min = max(mal_min, ben_min)
    overlap_max = min(mal_max, ben_max)
    overlap_pct = max(0, (overlap_max - overlap_min) / (max(mal_max, ben_max) - min(mal_min, ben_min)) * 100)

    results.append({
        'feature': col,
        'mal_mean': mal_vals.mean(),
        'ben_mean': ben_vals.mean(),
        'mal_median': mal_median,
        'ben_median': ben_median,
        'cohens_d': cohens_d,
        'pvalue': pval,
        'significant': pval < 0.05,
        'overlap_pct': overlap_pct,
        'mal_iqr': mal_iqr,
        'ben_iqr': ben_iqr,
    })

df_results = pd.DataFrame(results).sort_values('cohens_d', key=abs, ascending=False)

# Display results
print(f"\n{'Feature':<35} {'Effect Size':<12} {'p-value':<12} {'Significant':<12} {'Overlap %':<10}")
print("-" * 85)

significant_count = 0
for _, row in df_results.iterrows():
    sig_marker = "✓ YES" if row['significant'] else "  NO"
    effect_size = f"{row['cohens_d']:+.3f}"
    if abs(row['cohens_d']) < 0.2:
        effect_label = "(tiny)"
    elif abs(row['cohens_d']) < 0.5:
        effect_label = "(small)"
    elif abs(row['cohens_d']) < 0.8:
        effect_label = "(med)"
    else:
        effect_label = "(LARGE)"

    print(f"{row['feature']:<35} {effect_size:>6} {effect_label:<5} {row['pvalue']:>10.4f}  {sig_marker:<12} {row['overlap_pct']:>7.1f}%")

    if row['significant']:
        significant_count += 1

print("\n" + "=" * 80)
print(f"RESULTS SUMMARY")
print("=" * 80)
print(f"  Total features tested          : {len(df_results)}")
print(f"  Statistically significant      : {significant_count} ({significant_count/len(df_results)*100:.1f}%)")
print(f"  Average effect size (Cohen's d): {df_results['cohens_d'].abs().mean():.3f}")
print(f"  Max effect size                : {df_results['cohens_d'].abs().max():.3f}")
print(f"  Average overlap percentage     : {df_results['overlap_pct'].mean():.1f}%")

if significant_count / len(df_results) < 0.3:
    print("\n  ⚠️  WARNING: Less than 30% of features show significant differences!")
    print("     This suggests weak signal for separation.")
elif significant_count / len(df_results) < 0.6:
    print("\n  ⚡ MODERATE: About half the features differ significantly.")
    print("     Some separation is possible but not great.")
else:
    print("\n  ✅ STRONG: Most features differ significantly.")
    print("     Good signal for building classifiers.")

# =============================================================================
# TOP DISCRIMINATIVE FEATURES
# =============================================================================

print("\n" + "=" * 80)
print("TOP 10 MOST DISCRIMINATIVE FEATURES (by effect size)")
print("=" * 80)

for i, (_, row) in enumerate(df_results.head(10).iterrows(), 1):
    print(f"\n{i}. {row['feature']}")
    print(f"   Effect size (Cohen's d) : {row['cohens_d']:+.3f}")
    print(f"   p-value                 : {row['pvalue']:.2e}")
    print(f"   Malignant median        : {row['mal_median']:.4f}")
    print(f"   Benign median           : {row['ben_median']:.4f}")
    print(f"   Distribution overlap    : {row['overlap_pct']:.1f}%")

    if abs(row['cohens_d']) < 0.2:
        strength = "TINY (can't rely on this feature)"
    elif abs(row['cohens_d']) < 0.5:
        strength = "SMALL (weak separation)"
    elif abs(row['cohens_d']) < 0.8:
        strength = "MEDIUM (decent separation)"
    else:
        strength = "LARGE (good separation)"
    print(f"   Strength                : {strength}")

# Save to CSV
df_results.to_csv(f'{REPORT_DIR}/statistical_test_results.csv', index=False)
print(f"\n\nSaved to: {REPORT_DIR}/statistical_test_results.csv")

# =============================================================================
# VISUALIZATION
# =============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# Plot 1: Top 8 features - Distribution comparison
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Malignant vs Benign Distribution Comparison\n(Top 8 Most Discriminative Features)',
             fontsize=14, fontweight='bold')

for idx, (ax, (_, row)) in enumerate(zip(axes.flatten(), df_results.head(8).iterrows())):
    col = row['feature']
    mal_data = mal[col].dropna()
    ben_data = ben[col].dropna()

    ax.hist(ben_data, bins=50, alpha=0.6, label='Benign', color='#2E86AB', edgecolor='white')
    ax.hist(mal_data, bins=30, alpha=0.6, label='Malignant', color='#E84855', edgecolor='white')

    ax.set_xlabel(col, fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.set_title(f"d={row['cohens_d']:.2f} | p={row['pvalue']:.2e}", fontsize=9)
    if idx == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{GRAPH_DIR}/statistical_distributions_top8.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: statistical_distributions_top8.png")

# Plot 2: Effect size scatter
fig, ax = plt.subplots(figsize=(12, 6))

colors = ['#E84855' if p < 0.05 else '#2E86AB' for p in df_results['pvalue']]
scatter = ax.scatter(range(len(df_results)), df_results['cohens_d'].abs(),
                     s=100, alpha=0.6, c=colors, edgecolor='white', linewidth=1)

ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect (d=0.2)')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium effect (d=0.5)')
ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Large effect (d=0.8)')

ax.set_xlabel('Feature', fontsize=11)
ax.set_ylabel('|Cohen\'s d| (Effect Size)', fontsize=11)
ax.set_title('Effect Size Distribution: Which Features Best Separate Malignant from Benign?',
             fontsize=12, fontweight='bold')
ax.set_xticks(range(0, len(df_results), 2))
ax.set_xticklabels(df_results['feature'].iloc[::2], rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(alpha=0.3)

# Add color legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#E84855', alpha=0.6, label='Significant (p<0.05)'),
                  Patch(facecolor='#2E86AB', alpha=0.6, label='Not significant')]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig(f'{GRAPH_DIR}/statistical_effect_sizes.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: statistical_effect_sizes.png")

# Plot 3: P-value distribution
fig, ax = plt.subplots(figsize=(10, 6))
pvals = df_results['pvalue'].values

ax.hist(pvals, bins=30, alpha=0.7, color='#2E86AB', edgecolor='white')
ax.axvline(0.05, color='#E84855', linestyle='--', linewidth=2, label='Significance threshold (p=0.05)')
ax.set_xlabel('p-value', fontsize=11)
ax.set_ylabel('Number of Features', fontsize=11)
ax.set_title('P-value Distribution: Are Features Significantly Different?',
             fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{GRAPH_DIR}/statistical_pvalues.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: statistical_pvalues.png")

# Plot 4: Overlap analysis
fig, ax = plt.subplots(figsize=(12, 6))

sorted_by_overlap = df_results.sort_values('overlap_pct', ascending=False).head(15)
colors_overlap = ['#E84855' if x > 70 else '#F4A261' if x > 50 else '#2E86AB'
                  for x in sorted_by_overlap['overlap_pct']]

ax.barh(range(len(sorted_by_overlap)), sorted_by_overlap['overlap_pct'],
        color=colors_overlap, edgecolor='white', linewidth=1)
ax.set_yticks(range(len(sorted_by_overlap)))
ax.set_yticklabels(sorted_by_overlap['feature'], fontsize=9)
ax.set_xlabel('Distribution Overlap (%)', fontsize=11)
ax.set_title('How Much Do Benign and Malignant Distributions Overlap?\n(Higher = More Overlap = Harder to Separate)',
             fontsize=12, fontweight='bold')
ax.axvline(50, color='gray', linestyle='--', alpha=0.5)
ax.set_xlim(0, 100)

# Color legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2E86AB', label='Good separation (<50% overlap)'),
                  Patch(facecolor='#F4A261', label='Moderate overlap (50-70%)'),
                  Patch(facecolor='#E84855', label='High overlap (>70%)')]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig(f'{GRAPH_DIR}/statistical_overlap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: statistical_overlap.png")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print("""
What these statistics tell us:

1. MANN-WHITNEY U TEST
   Tests if the distributions are truly different (not just random variation)
   ✓ If p < 0.05, the difference is statistically significant
   ✗ If p ≥ 0.05, the difference could be due to chance

2. COHEN'S D (Effect Size)
   How large is the practical difference between groups?
   - |d| < 0.2  : Tiny (negligible difference)
   - |d| < 0.5  : Small (detectable but small difference)
   - |d| < 0.8  : Medium (noticeable difference)
   - |d| ≥ 0.8  : Large (substantial difference)

   Our result: Average d = {:.3f} → {}

3. DISTRIBUTION OVERLAP
   If two distributions heavily overlap, they're hard to separate
   - < 50%: Good separation
   - 50-70%: Moderate separation
   - > 70%: Poor separation, hard to classify

   Our result: Average overlap = {:.1f}% → {}

CONCLUSION FOR YOUR MODEL:
If most features show:
  • Low effect sizes (|d| < 0.2) → weak signal, hard to build good classifier
  • High p-values (p > 0.05) → not significantly different
  • High overlap (> 70%) → benign and malignant look very similar

Then your model's 50% recall performance is realistic, not a failure!
""".format(df_results['cohens_d'].abs().mean(),
           "WEAK signal" if df_results['cohens_d'].abs().mean() < 0.3 else "MODERATE signal",
           df_results['overlap_pct'].mean(),
           "Poor separation (as expected)" if df_results['overlap_pct'].mean() > 60 else "Reasonable separation"))

print("=" * 80)
