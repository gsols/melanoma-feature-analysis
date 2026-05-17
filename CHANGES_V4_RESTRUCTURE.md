# Model V4 Restructuring — Stratified Split + Proper Data Leakage Prevention

## The Problem

Your original v4 model had a subtle but critical data leakage issue:

1. **Applied resampling BEFORE split**: The SMOTE + undersampling was done on the full dataset, then split into train/test
2. **Synthetic data influenced evaluation**: Synthetic malignant samples (created from the full dataset) could have patterns that affected test expectations
3. **Test set was benign-only**: You couldn't evaluate true malignant detection, only how benign cases "look like" malignant
4. **Metrics were inflated**: F1=0.99 for 5:1 ratio seemed suspiciously high

**Comparison of metrics across ratios showed the problem:**
- 5:1 ratio → F1=0.9900 (highest, synthetic-heavy)
- 10:1 ratio → F1=0.9853
- 20:1 ratio → F1=0.9786
- 30:1 ratio → F1=0.6130 (realistic ratio, much lower)

This massive drop suggests the synthetic data at 5:1 was encoding false patterns.

---

## The Solution: Stratified Split BEFORE Resampling

### New Workflow

```
Original data (393 malignant, 320.5k benign)
    ↓
STRATIFIED SPLIT (maintain class proportions)
    ├─ Training: 299 malignant, 243.6k benign (814.7:1)
    └─ Test: 94 malignant, 76.9k benign (818.4:1) ← NEVER MODIFIED

For each ratio (5:1, 10:1, 15:1, 20:1, 25:1, 30:1):
    ├─ SMOTE: 299 malignant → {50k, 27k, 18k, 14k, 11k, 9.7k}
    ├─ Undersample: 243.6k benign → {250k, 272k, 281k, 285k, 288k, 290k}
    └─ Save: train_ratio_X_1.csv (resampled training data only)

Evaluation:
    ├─ Train on: train_ratio_X_1.csv (resampled, synthetic-heavy)
    └─ Test on: test_stratified.csv (REAL, UNCHANGED, both classes)
```

### Key Improvements

✓ **No synthetic data leakage** — SMOTE only sees training data
✓ **Test set is real and balanced** — Contains 94 real malignant cases
✓ **Honest evaluation** — Test metrics show true generalization
✓ **Same test set for all ratios** — Fair comparison across models
✓ **Detects overfitting to synthetic data** — Lower ratios may still overfit, now visible in test metrics

---

## Files Created/Modified

### New Scripts
- `isic_data_prep_ratio_v2_stratified.py` — Proper stratified split + resampling
  - Outputs: `ratio_data_v2/test_stratified.csv` (shared, real data)
  - Outputs: `ratio_data_v2/train_ratio_{5,10,15,20,25,30}_1.csv` (resampled training)

### Modified Scripts
- `isic_model_v4_ratio_balanced.py` — Updated to use v2 datasets
  - Changed from LightGBM to scikit-learn GradientBoosting (for compatibility)
  - Now evaluates on BOTH classes (not benign-only)
  - Includes test set metrics in reports
  - Outputs: `v4_outputs_ratio_balanced_{RATIO}_1/reports/{OUTPUT_PREFIX}_test_predictions.csv`

---

## What Changed in Data

| Aspect | v1 (Original) | v2 (New) |
|--------|---------------|----------|
| **Split timing** | After resampling | BEFORE resampling |
| **Test set** | 80.1k benign only | 77.0k benign + 94 malignant |
| **Data leakage** | Yes (synthetic patterns) | No (real test data) |
| **Honest evaluation** | No (no malignant test) | Yes (both classes) |
| **Train/test contamination** | Possible | Prevented |

---

## How to Use

### Run a single model with specific ratio:
```bash
RATIO=10 python3 isic_model_v4_ratio_balanced.py
```

### Run all ratios:
```bash
for ratio in 5 10 15 20 25 30; do
  echo "Training ratio $ratio:1..."
  RATIO=$ratio python3 isic_model_v4_ratio_balanced.py
done
```

### Compare results:
```python
import pandas as pd
df = pd.read_csv('reports/v4_ratio_model_registry.csv')
print(df[['ratio', 'cv_f1', 'cv_recall', 'test_f1', 'test_recall']])
```

---

## Expected Behavior

You should now see:
- **More realistic metrics** — Test F1 scores will be lower than CV (expected overfitting)
- **Malignant detection rates** — For the first time, you can measure recall on REAL malignant test cases
- **Better ratio comparison** — 30:1 ratio should improve on test (closer to real-world distribution)
- **No artificial inflation** — 5:1 ratio won't dominate all metrics via synthetic patterns

---

## Next Steps

1. **Run all ratios** and compare test metrics
2. **Choose best model** — Probably 20:1 or 30:1 (realistic imbalance, good test recall)
3. **Analyze why 5:1 overfit** — Likely synthetic data encoding false patterns
4. **Consider production use** — Use the ratio closest to real-world distribution

---

## References

- Train/Test Split Timing: https://scikit-learn.org/stable/modules/cross_validation.html#splitting-rules
- SMOTE: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
- Data Leakage: https://machinelearningmastery.com/data-leakage-machine-learning/
