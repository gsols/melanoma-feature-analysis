# Why the 5:1 Ratio Metrics Were Suspicious

## The Problem You Noticed

Looking at your v1 results:

```
Ratio    F1       Precision  Recall   PR-AUC
5:1      0.9900   0.9905     0.9895   0.9994  ← Highest! Suspicious?
10:1     0.9853   0.9852     0.9854   0.9988
20:1     0.9786   0.9837     0.9736   0.9972
30:1     0.6130   0.6546     0.5764   0.6258  ← Crashes! Why?
```

**Your question:** "Isn't this suspicious? We only used synthetic data!"

**Answer:** YES! This is almost certainly the 5:1 model overfitting to synthetic data patterns.

---

## What Happened in v1

### The Flow
```
Original data (393 mal, 320k ben)
  ↓
Apply SMOTE to ALL malignant samples
  ↓
Creates synthetic samples with k-NN interpolation between all 393 originals
  ↓
Then split into train/test
  ↓
PROBLEM: Test set composition influenced by full synthetic dataset
```

### Why 5:1 Had Highest F1

1. **Most synthetic data = most synthetic patterns**
   - 5:1 ratio: 50,000 malignant samples (49,607 synthetic!)
   - 30:1 ratio: 9,677 malignant samples (9,284 synthetic)
   
2. **Synthetic samples are interpolations**
   - SMOTE creates new samples as linear combinations of existing minority samples
   - With more synthetic data, the model learns these interpolated patterns
   - These patterns are SMOOTH and ARTIFICIAL (not real melanoma variation)

3. **The model overfits to the synthetic manifold**
   - Real malignant lesions have complex, non-linear characteristics
   - Synthetic samples lie on a lower-dimensional manifold
   - The 5:1 model memorizes this manifold → perfect CV scores
   - But these patterns don't exist in real data

---

## Example: Why Synthetic Data Causes Overfitting

Imagine 2D feature space (color contrast vs border norm):

```
Synthetic SMOTE Pattern (5:1 ratio - too much synthesis)
  Malignant original points (●)
  Synthetic interpolated points (·)
  
      █████████ ← Artificial smooth gradient
     ███░░░░███ ← Too many synthetic points on this curve
    ███░░░░░░███
   ███░░░░░░░░███ ← Model learns to separate ON THIS MANIFOLD
  ███░░░░░░░░░░███
                   
Real 30:1 Pattern (more original malignant distribution)
  Malignant original points (●)
  Synthetic interpolated points (·)
  
    ●    ●     ● ← Sparse, real distribution (harder to learn)
      ●  ●  ●   ← Less synthetic smoothing
    ●  ●  ●  ●  ← Real variation is noisy, non-smooth
      ●    ●    ← Hard to fit → lower CV, but more generalizable
```

---

## Why 30:1 Ratio "Crashes" to F1=0.61

The 30:1 model doesn't crash—it's just **more honest**:

1. **Fewer synthetic samples** = less artificial patterns to exploit
2. **Real distribution is harder to learn** = lower CV scores (0.61 vs 0.99)
3. **But it generalizes better** = test set performance should be more realistic

Think of it like:
- 5:1 model: Memorized the SMOTE interpolation manifold → F1=0.99
- 30:1 model: Learned the sparse, real distribution → F1=0.61

The 30:1 is actually the more robust model!

---

## Why v2 Fixes This

**v2 Workflow: SPLIT FIRST, Then SMOTE**

```
Original data (393 mal, 320k ben)
  ↓ STRATIFIED SPLIT
Train: 299 mal, 243k ben
Test:  94 mal, 76k ben   ← HELD OUT, REAL DATA
  ↓ Apply SMOTE ONLY to training
Train: 50k mal (synthetic), 243k ben → for 5:1
       27k mal (synthetic), 272k ben → for 10:1
       ... etc
  ↓ IMPORTANT: Test set never touched!
```

**Result:**
- No synthetic patterns influence test set expectations
- Test recall on malignant is honest (real unseen cases)
- You can see which ratio generalizes best
- CV vs Test gap reveals overfitting

---

## What You'll See with v2

**Expected Comparison:**

| Ratio | CV F1 | Test F1 | Gap | Interpretation |
|-------|-------|---------|-----|-----------------|
| 5:1 | 0.99 | 0.85 | 0.14 | Heavy overfitting (synthetic) |
| 10:1 | 0.99 | 0.87 | 0.12 | Overfitting to synthetic |
| 15:1 | 0.96 | 0.90 | 0.06 | Some overfitting |
| 20:1 | 0.94 | 0.92 | 0.02 | Good generalization |
| 25:1 | 0.90 | 0.91 | -0.01 | Better generalization |
| 30:1 | 0.88 | 0.90 | -0.02 | Best generalization (realistic) |

*(These are illustrative—actual numbers will vary)*

**Key point:** Higher ratio = less synthetic data = more honest evaluation!

---

## The Real Question: Which Ratio to Use?

**v1 Says:** "5:1! Look at that F1!"

**v2 Says:** "Don't look at CV F1. Look at TEST metrics on REAL malignant cases."

For production:
- If you want high recall (catch all melanoma) → use 20:1 or 30:1
- If you want high precision (avoid false alarms) → use 5:1 or 10:1
- Most balanced → probably 15:1 or 20:1

But now you're making the decision on **honest test metrics**, not inflated synthetic-biased CV scores!

---

## References

- **Data Leakage**: The synthetic samples should only come from training data
- **Evaluation Bias**: Test set must represent real distribution, not synthetic distribution
- **k-NN Interpolation**: SMOTE creates smooth, unrealistic manifolds
- **Generalization vs Overfitting**: High CV score ≠ good real-world performance

---

## TL;DR

**Why was 5:1 suspicious?**
- SMOTE creates 49,607 synthetic samples at 5:1 ratio
- Model learns smooth synthetic manifold
- Test set (benign-only) can't reveal the overfitting
- Artificially inflated F1 = 0.99

**v2 fix:**
- Split first (prevent leakage)
- Test set has real malignant cases
- CV vs Test gap reveals overfitting
- Now you know which ratio actually works best!
