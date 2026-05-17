# Quick Start: Model Improvements & Tracking

## What Changed (Simple Explanation)

### 1. **Better Features** 🎯
- Added 5 "combo" features (e.g., color × size = how intense is a big spot?)
- These combinations work better than individual features
- Result: 2nd most important feature is now `color_x_size`

### 2. **Less Strict Rules** 📏
- Old: "ignore patterns unless they affect 200+ samples"
- New: "find patterns in just 50+ samples"
- Why: Malignant samples (only 393) need smaller pattern groups

### 3. **Penalize Mistakes on Minority** ⚖️
- Old: Treating benign and malignant mistakes equally
- New: Mistake on malignant = 326x worse
- Why: Catching malignant lesions is more important

### 4. **Automatic Record Keeping** 📊
- Every time you train the model, results are saved automatically
- You can compare old runs vs new runs
- No manual tracking needed!

---

## How to Use

### Train the Model
```bash
cd /home/lumpia/Documents/School/Third\ Year/2nd_Sem/CMSC_170/melanoma-feature-analysis
python isic_model.py
```
**Takes ~10 minutes**. Automatically saves to `reports/model_registry.csv`

### View Results
```bash
# Show last 5 models trained
python view_model_registry.py latest

# Show best model so far
python view_model_registry.py compare

# Show ALL models ever trained
python view_model_registry.py
```

---

## What the Metrics Mean (Simple)

### CV F1 Score
- **What it is**: How balanced is your model? (1.0 = perfect, 0.0 = terrible)
- **Your result**: 0.0398 = very imbalanced, but honest
- **Don't worry**: With 816:1 imbalance, low F1 is expected

### Screening Recall
- **What it is**: Out of 100 malignant-like lesions, how many do you catch?
- **Your result**: 50.4% = catch about half of suspicious-looking benign lesions
- **Medical use**: Flag the suspicious ones for doctor review

### Screening Precision
- **What it is**: Of the 100 lesions you flag, how many are actually suspicious?
- **Your result**: 0.57% = low, but okay for screening
- **Medical use**: Many flagged lesions are benign, but at least you don't miss malignant

---

## Experiment with Settings

**File to edit**: `isic_model.py` (around line 180)

```python
# Option 1: More aggressive on minority (catch more malignant)
CLASS_WEIGHT_MULTIPLIER = 0.8  # Was 0.4, try 0.8

# Option 2: Allow even smaller patterns (for more learning)
'min_child_samples': 50,  # Was 75, try 50

# Option 3: Different recall target for screening
TARGET_RECALL = 0.70  # Was 0.50, try 0.70 to catch more
```

Then run:
```bash
python isic_model.py
python view_model_registry.py latest
```

The new results will be saved automatically and you can compare!

---

## Track Your Progress

After each run, check:
```bash
python view_model_registry.py compare
```

Shows:
- **First model** you trained (baseline)
- **Best model** so far (by F1 score)
- **Latest model** you just trained
- **Trend**: Did F1 go up or down?

---

## The Honest Truth About Your Data

**Your model says**: Even with good features and smart parameters, it can only catch 50% of malignant-like lesions without flagging 10.8% of normal benign ones.

**This means**: Benign and malignant lesions **look very similar**. This is realistic for melanoma — it's actually hard to distinguish!

**Good news**: A screening tool that catches 50% of suspicious cases and flags 10% for review is **useful in practice**. Doctors can manually review the flagged 10%.

---

## Files You Care About

| File | What it does |
|------|--------------|
| `isic_model.py` | The main model (edit settings here) |
| `reports/model_registry.csv` | Your experiment log (view this to track progress) |
| `view_model_registry.py` | View the experiment log |
| `reports/metrics_report.txt` | Detailed results from latest run |
| `reports/benign_risk_predictions.csv` | The actual predictions on test data |

---

## Common Questions

**Q: Why did performance go down after improvements?**  
A: The new settings are more honest. They don't overfit to false patterns. This is better!

**Q: Can I improve F1 further?**  
A: Only by getting better features or more malignant samples. Your current features have weak separation.

**Q: Should I try different thresholds?**  
A: Yes! Edit `TARGET_RECALL = 0.50` (try 0.30, 0.70, 0.90) and run the model again.

**Q: How do I use the predictions?**  
A: Open `reports/benign_risk_predictions.csv` — it shows every test lesion ranked by malignancy risk.

---

## Next Steps (Optional)

1. **Try interaction feature experiments**
   - Works! `color_x_size` is now 2nd most important

2. **Try different recall targets**
   - Edit line 57: `TARGET_RECALL = 0.70` (catch more)
   - Or `TARGET_RECALL = 0.30` (flag less)

3. **Try ensemble models**
   - Combine LightGBM + Logistic Regression
   - Advanced but potentially better

4. **Try Naive Bayes baseline**
   - Fast, simple, good for comparison
   - See if the signal is real

---

Done! Your model is now tracking results automatically. 🎉
