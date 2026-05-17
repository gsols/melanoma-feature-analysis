#!/bin/bash
# Run all model_v4 ratios and generate comparison

echo "=========================================="
echo "Running Model v4 - All Ratios"
echo "=========================================="
echo ""

for ratio in 5 10 15 20 25 30; do
  echo "=========================================="
  echo "Training ratio $ratio:1..."
  echo "=========================================="
  RATIO=$ratio python3 isic_model_v4_ratio_balanced.py
  echo ""
done

echo "=========================================="
echo "✓ All models trained!"
echo "=========================================="
echo ""
echo "Comparing results..."
python3 << 'PYTHON_EOF'
import pandas as pd

# Load registry
df = pd.read_csv('reports/v4_ratio_model_registry.csv')

# Sort by ratio
df = df.sort_values('ratio')

print("\n" + "=" * 100)
print("MODEL COMPARISON — v4 ALL RATIOS")
print("=" * 100)
print()
print(df[['ratio', 'cv_f1', 'cv_recall', 'cv_precision',
           'test_f1', 'test_recall', 'test_precision']].to_string(index=False))

print()
print("Interpretation:")
print("  • CV metrics are on training set (synthetic-heavy)")
print("  • Test metrics are on REAL unseen malignant cases (94 samples)")
print("  • Large gap (CV - Test) = overfitting to synthetic patterns")
print("  • Best production model = largest test recall with reasonable false positive rate")
print()
print("=" * 100)
PYTHON_EOF

echo ""
echo "To compare visually, run:"
echo "  python3 << 'EOF'"
echo "  import pandas as pd"
echo "  import matplotlib.pyplot as plt"
echo "  "
echo "  df = pd.read_csv('reports/v4_ratio_model_registry.csv')"
echo "  df = df.sort_values('ratio')"
echo "  "
echo "  fig, axes = plt.subplots(2, 2, figsize=(14, 10))"
echo "  "
echo "  axes[0,0].plot(df['ratio'], df['cv_f1'], 'o-', label='CV', linewidth=2)"
echo "  axes[0,0].plot(df['ratio'], df['test_f1'], 's-', label='Test', linewidth=2)"
echo "  axes[0,0].set_ylabel('F1 Score')"
echo "  axes[0,0].set_title('F1 Score Comparison')"
echo "  axes[0,0].legend()"
echo "  axes[0,0].grid()"
echo "  "
echo "  axes[0,1].plot(df['ratio'], df['cv_recall'], 'o-', label='CV', linewidth=2)"
echo "  axes[0,1].plot(df['ratio'], df['test_recall'], 's-', label='Test', linewidth=2)"
echo "  axes[0,1].set_ylabel('Recall')"
echo "  axes[0,1].set_title('Malignant Detection Rate (Recall)')"
echo "  axes[0,1].legend()"
echo "  axes[0,1].grid()"
echo "  "
echo "  axes[1,0].plot(df['ratio'], df['cv_f1'] - df['test_f1'], 'o-', color='red', linewidth=2)"
echo "  axes[1,0].set_ylabel('F1 Gap')"
echo "  axes[1,0].set_title('Overfitting Gap (CV - Test F1)')"
echo "  axes[1,0].grid()"
echo "  "
echo "  axes[1,1].plot(df['ratio'], df['test_recall'], 'o-', label='Recall', linewidth=2)"
echo "  axes[1,1].axhline(0.8, color='green', linestyle='--', label='80% target')"
echo "  axes[1,1].set_xlabel('Benign:Malignant Ratio')"
echo "  axes[1,1].set_ylabel('Recall')"
echo "  axes[1,1].set_title('Real Malignant Detection (80% target)')"
echo "  axes[1,1].legend()"
echo "  axes[1,1].grid()"
echo "  "
echo "  plt.tight_layout()"
echo "  plt.savefig('ratio_comparison_all.png', dpi=150, bbox_inches='tight')"
echo "  plt.show()"
echo "  EOF"
