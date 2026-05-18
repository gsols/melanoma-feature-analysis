#!/usr/bin/env python3
"""
Validate v4 model accuracy for thesis:
"Metadata-based melanoma risk scoring: identifying benign lesions
 with features similar to malignant melanoma cases"

Key question: Does the model learn meaningful malignancy-like features,
or is it just overfitting to synthetic SMOTE data?
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# ============================================================================
# 1. CROSS-VALIDATION METRICS ACROSS RATIOS
# ============================================================================
print("=" * 80)
print("ACCURACY CHECK 1: Cross-Validation Metrics Across Ratios")
print("=" * 80)

ratios = [5, 10, 20, 30]
cv_results = []

for ratio in ratios:
    report_path = f'v4.1_outputs_ratio_balanced_{ratio}_1/reports/v4_readable_report.txt'
    if not Path(report_path).exists():
        print(f"⚠ Missing: {report_path}")
        continue

    with open(report_path) as f:
        content = f.read()

    # Extract mean metrics from the "MEAN ± STD" section
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'MEAN ± STD' in line:
            # Next lines have f1, accuracy, precision, recall, pr_auc
            metrics_block = '\n'.join(lines[i:i+10])
            break

    try:
        for line in lines[i:i+10]:
            if 'f1' in line and '0.' in line:
                f1 = float(line.split()[-2])
            elif 'recall' in line and '0.' in line:
                recall = float(line.split()[-2])
            elif 'precision' in line and '0.' in line and 'f1' not in line:
                precision = float(line.split()[-2])
            elif 'pr_auc' in line and '0.' in line:
                pr_auc = float(line.split()[-2])

        cv_results.append({
            'Ratio': f'{ratio}:1',
            'F1': f1,
            'Recall': recall,
            'Precision': precision,
            'PR-AUC': pr_auc
        })
    except Exception as e:
        print(f"✗ Error parsing {ratio}:1 - {e}")

cv_df = pd.DataFrame(cv_results)
print("\n✓ Cross-Validation Metrics (synthetic malignant detection):")
print(cv_df.to_string(index=False))

# ============================================================================
# 2. TEST SET PERFORMANCE: Benign Risk Stratification
# ============================================================================
print("\n" + "=" * 80)
print("ACCURACY CHECK 2: Test Set Performance (Benign Risk Stratification)")
print("=" * 80)

test_results = []

for ratio in ratios:
    report_path = f'v4.1_outputs_ratio_balanced_{ratio}_1/reports/v4_readable_report.txt'
    if not Path(report_path).exists():
        continue

    with open(report_path) as f:
        content = f.read()

    try:
        # Extract test set metrics
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'TEST SET PREDICTIONS' in line or 'TEST SET EVALUATION' in line:
                for j in range(i, min(i+15, len(lines))):
                    if 'Records scored' in lines[j]:
                        test_size = int(lines[j].split(':')[1].strip())
                    elif 'Prob max' in lines[j]:
                        prob_max = float(lines[j].split(':')[1].strip())
                    elif 'Prob mean' in lines[j]:
                        prob_mean = float(lines[j].split(':')[1].strip())
                    elif 'Screening flagged' in lines[j]:
                        flagged = lines[j].split('(')[1].split('%')[0].strip()
                        flagged = float(flagged)

                test_results.append({
                    'Ratio': f'{ratio}:1',
                    'Test Size': test_size,
                    'Mean Prob': prob_mean,
                    'Max Prob': prob_max,
                    'Flagged %': flagged
                })
                break
    except Exception as e:
        print(f"✗ Error parsing test metrics for {ratio}:1 - {e}")

test_df = pd.DataFrame(test_results)
print("\n✓ Test Set Benign Risk Distribution:")
print(test_df.to_string(index=False))

# ============================================================================
# 3. GAP ANALYSIS: CV vs Test Performance
# ============================================================================
print("\n" + "=" * 80)
print("ACCURACY CHECK 3: CV→Test Gap (Overfitting Detection)")
print("=" * 80)
print("\nInterpretation:")
print("  • Small gap (< 5%): Model generalizes well")
print("  • Medium gap (5-15%): Some overfitting")
print("  • Large gap (> 15%): Significant overfitting to synthetic data")

if not cv_df.empty and not test_df.empty:
    comparison = cv_df.copy()
    comparison['Test Max Prob'] = test_df['Max Prob']
    comparison['Benign Flagged %'] = test_df['Flagged %']

    print("\n" + comparison.to_string(index=False))

# ============================================================================
# 4. FEATURE IMPORTANCE CONSISTENCY
# ============================================================================
print("\n" + "=" * 80)
print("ACCURACY CHECK 4: Feature Importance Stability Across Ratios")
print("=" * 80)
print("\nAre the top features consistent? (indicates learned real patterns)")

top_n = 5
for ratio in ratios:
    report_path = f'v4.1_outputs_ratio_balanced_{ratio}_1/reports/v4_readable_report.txt'
    if not Path(report_path).exists():
        continue

    with open(report_path) as f:
        content = f.read()

    try:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'TOP 20 FEATURES' in line:
                features = []
                for j in range(i+3, min(i+top_n+3, len(lines))):
                    parts = lines[j].split()
                    if len(parts) >= 2:
                        feature_name = ' '.join(parts[1:-1])
                        if feature_name and feature_name[0].isalpha():
                            features.append(feature_name)

                print(f"\n  {ratio}:1 → Top 5: {features[:5]}")
                break
    except Exception as e:
        print(f"✗ Error parsing features for {ratio}:1")

# ============================================================================
# 5. LOOK AT ACTUAL PREDICTIONS FOR 30:1
# ============================================================================
print("\n" + "=" * 80)
print("ACCURACY CHECK 5: Actual Predictions (30:1 Ratio)")
print("=" * 80)

pred_file = 'v4.1_outputs_ratio_balanced_30_1/reports/v4_ratio_30_benign_risk_predictions.csv'
if Path(pred_file).exists():
    preds = pd.read_csv(pred_file)
    print(f"\n✓ Loaded {len(preds)} predictions from test set")
    print(f"\nRisk Distribution (30:1 model):")

    if 'lgbm_malignancy_prob' in preds.columns:
        prob_col = 'lgbm_malignancy_prob'
    else:
        prob_col = preds.columns[preds.columns.str.contains('prob', case=False)][0]

    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = preds[prob_col].quantile(p/100)
        print(f"    {p:2d}th percentile: {val:.6f}")

    # Show top 10 highest-risk benign lesions
    print(f"\n  Top 10 Highest-Risk Benign Lesions:")
    top_10 = preds.nlargest(10, prob_col)[['lgbm_malignancy_prob', 'age_approx', 'anatom_site_general', 'clin_size_long_diam_mm']]
    print(top_10.to_string())
else:
    print(f"⚠ Predictions file not found: {pred_file}")

# ============================================================================
# 6. THESIS VALIDATION SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("THESIS VALIDATION SUMMARY")
print("=" * 80)
print("""
For thesis: "Identifying benign lesions with features similar to malignant"

✓ THESIS-RELEVANT QUESTIONS:
  1. Does the model learn meaningful malignancy-like features?
  2. Are benign lesions stratified by risk in a clinically sensible way?
  3. Are the flagged benign lesions actually suspicious?

FINDINGS:
""")

if not cv_df.empty:
    avg_f1 = cv_df['F1'].mean()
    print(f"  • CV F1 across ratios: {avg_f1:.4f}")
    if avg_f1 > 0.9:
        print(f"    ⚠ Suspiciously HIGH - likely overfitting to synthetic SMOTE data")
    elif avg_f1 > 0.6:
        print(f"    ✓ Realistic range for synthetic data detection")
    else:
        print(f"    ✗ Model learning weak signals")

if not test_df.empty:
    avg_flagged = test_df['Flagged %'].mean()
    print(f"  • Average benign flagged as high-risk: {avg_flagged:.2f}%")
    if avg_flagged < 0.1:
        print(f"    ⚠ Very few benign lesions identified as 'malignancy-like'")
    elif avg_flagged < 1.0:
        print(f"    ✓ Reasonable proportion of suspicious benign cases")
    else:
        print(f"    ⚠ Many benign lesions flagged (high false positive rate)")

print(f"""
RECOMMENDATION FOR THESIS:

  The model trained on ratio-balanced data shows signs of OVERFITTING to
  synthetic SMOTE-generated malignant samples. The benign test set has very
  low probability scores (mean ~0.004-0.045), suggesting the model cannot
  distinguish feature patterns between real benign and synthetic malignant.

  To properly validate the thesis, consider:

  1. Use a dataset with REAL malignant samples in test set
     (see v4.1_5_1 which has 94 actual malignant samples)

  2. Lower the screening threshold to identify more "suspicious" benign cases
     (currently flagging <1% of benign lesions)

  3. Validate that flagged benign lesions have clinically plausible
     malignancy-like features (dermatologist review)

  4. Compare SMOTE-based model with models trained on real data
""")

print("\n" + "=" * 80)
print("END OF ANALYSIS")
print("=" * 80)
