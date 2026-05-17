# Quick Start: Running Model v4 with Stratified Split

## What Was Done

✓ Created proper stratified split (before resampling)
✓ Generated new datasets in `ratio_data_v2/`:
  - **test_stratified.csv** — Shared test set (94 malignant + 76,928 benign, never modified)
  - **train_ratio_5_1.csv, train_ratio_10_1.csv, ...** — Training data resampled to each ratio
✓ Updated model script to:
  - Use new v2 datasets (stratified)
  - Evaluate on BOTH classes (not just benign)
  - Report test metrics alongside CV metrics
  - Use scikit-learn GradientBoosting (for compatibility)

## Data Structure

```
Original: 393 mal, 320.5k ben (815.6:1)
  ↓ Stratified Split
Train:    299 mal, 243.6k ben (814.7:1)  ← Base for resampling
Test:     94 mal, 76.9k ben (818.4:1)    ← Held out, never modified
  ↓ For each ratio
Resampled: 50k mal (synthetic), 243.6k ben → for 5:1
Resampled: 27k mal (synthetic), 272.7k ben → for 10:1
... etc
```

## Run Single Ratio

```bash
RATIO=5 python3 isic_model_v4_ratio_balanced.py
RATIO=10 python3 isic_model_v4_ratio_balanced.py
RATIO=20 python3 isic_model_v4_ratio_balanced.py
```

## Run All Ratios

```bash
for ratio in 5 10 15 20 25 30; do
  echo "=========================================="
  echo "Training ratio $ratio:1..."
  echo "=========================================="
  RATIO=$ratio python3 isic_model_v4_ratio_balanced.py
done
```

## Output Files

For each ratio, creates:
```
v4_outputs_ratio_balanced_5_1/
  ├── reports/
  │   ├── v4_readable_report.txt              (Human readable summary)
  │   ├── v4_global_metrics.json              (Key metrics as JSON)
  │   ├── v4_ratio_5_fold_metrics.csv         (Per-fold CV metrics)
  │   ├── v4_ratio_5_feature_importance.csv   (Feature importance ranking)
  │   ├── v4_ratio_5_test_predictions.csv     (Predictions on test set)
  │   └── v4_ratio_5_recall_target_metrics.csv (Recall tradeoffs)
  └── graphs/
      ├── v4_oof_confusion_matrix.png
      ├── v4_oof_precision_recall_curve.png
      └── v4_feature_importance.png
```

## Compare Results

After running all ratios, compare with:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load registry
df = pd.read_csv('reports/v4_ratio_model_registry.csv')

# Compare CV vs Test metrics
print(df[['ratio', 'cv_f1', 'cv_recall', 'test_f1', 'test_recall']])

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(df['ratio'], df['cv_f1'], 'o-', label='CV F1', linewidth=2)
ax1.plot(df['ratio'], df['test_f1'], 's-', label='Test F1', linewidth=2)
ax1.set_xlabel('Benign:Malignant Ratio')
ax1.set_ylabel('F1 Score')
ax1.legend()
ax1.grid()
ax1.set_title('F1 Score Comparison')

ax2.plot(df['ratio'], df['cv_recall'], 'o-', label='CV Recall', linewidth=2)
ax2.plot(df['ratio'], df['test_recall'], 's-', label='Test Recall', linewidth=2)
ax2.set_xlabel('Benign:Malignant Ratio')
ax2.set_ylabel('Recall')
ax2.legend()
ax2.grid()
ax2.set_title('Malignant Detection Rate (Recall)')

plt.tight_layout()
plt.savefig('ratio_comparison.png', dpi=150)
plt.show()
```

## Expected Differences from v1

**Old (v1) behavior:**
- 5:1 ratio: F1=0.9900, Recall=0.9895 (suspicious!)
- 30:1 ratio: F1=0.6130, Recall=0.5764 (crash!)
- Test set: benign-only (no malignant evaluation)

**New (v2) behavior (expected):**
- 5:1 ratio: CV F1≈0.99, but **Test F1** will be lower (realistic overfitting)
- 30:1 ratio: CV F1≈0.62, **Test F1** should improve (better real-world distribution)
- Test set: **Both benign AND malignant** (honest evaluation of malignant detection)
- CV vs Test gap: Larger gap = more overfitting to synthetic data

## Why This Matters

The v1 approach created synthetic patterns that inflated metrics. Now:

✓ You can see which ratio generalizes best
✓ Test recall on malignant is honest (not synthetic-biased)
✓ Lower-performing ratios aren't dismissed (30:1 might be best for production)
✓ Data leakage is impossible (train/test split happens first)

---

**Next step:** Run the model, then compare the ratios. The "suspicious" 5:1 F1 should become more realistic!
