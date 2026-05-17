#!/usr/bin/env python
"""
View and compare all recorded models from the model registry.
This helps track performance improvements over time.

Usage:
    python view_model_registry.py              # Show all models
    python view_model_registry.py latest       # Show last 5 models
    python view_model_registry.py compare      # Compare current with previous best
"""

import pandas as pd
import sys
from pathlib import Path

REGISTRY_FILE = 'reports/model_registry.csv'

def main():
    if not Path(REGISTRY_FILE).exists():
        print(f"❌ No model registry found at {REGISTRY_FILE}")
        print("   Run isic_model.py first to create it.")
        return

    df = pd.read_csv(REGISTRY_FILE)

    # Determine what to show
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if mode == 'latest':
        print("\n" + "="*100)
        print("LATEST 5 MODELS")
        print("="*100)
        df_show = df.tail(5)
    elif mode == 'compare':
        print("\n" + "="*100)
        print("COMPARISON: BEST vs LATEST vs FIRST")
        print("="*100)
        best_idx = df['cv_f1'].idxmax()
        df_show = df.iloc[[0, best_idx, -1]]
    else:
        print("\n" + "="*100)
        print(f"ALL {len(df)} MODELS IN REGISTRY")
        print("="*100)
        df_show = df

    # Display with nice formatting
    display_cols = [
        'timestamp',
        'cv_f1', 'cv_recall', 'cv_precision', 'cv_pr_auc',
        'screening_f1', 'screening_recall', 'screening_precision',
        'min_child_samples', 'scale_pos_weight', 'num_interaction_features'
    ]

    cols_to_show = [c for c in display_cols if c in df_show.columns]

    print("\n" + df_show[cols_to_show].to_string(index=True))

    # Show best model details
    print("\n" + "="*100)
    print("BEST MODEL (by CV F1)")
    print("="*100)
    best = df.loc[df['cv_f1'].idxmax()]
    print(f"  Timestamp              : {best['timestamp']}")
    print(f"  CV F1                  : {best['cv_f1']}")
    print(f"  CV Recall              : {best['cv_recall']}")
    print(f"  CV Precision           : {best['cv_precision']}")
    print(f"  CV PR-AUC              : {best['cv_pr_auc']}")
    print(f"  Screening Recall       : {best['screening_recall']}")
    print(f"  Screening Precision    : {best['screening_precision']}")
    print(f"  Min Child Samples      : {int(best['min_child_samples'])}")
    print(f"  Scale Pos Weight       : {best['scale_pos_weight']:.1f}")
    print(f"  Num Features           : {int(best['num_features_total'])}")
    print(f"  Interaction Features   : {int(best['num_interaction_features'])}")

    # Show trend
    if len(df) > 1:
        print("\n" + "="*100)
        print("TREND (Latest vs Best)")
        print("="*100)
        latest = df.iloc[-1]
        f1_delta = ((latest['cv_f1'] - best['cv_f1']) / best['cv_f1'] * 100) if best['cv_f1'] > 0 else 0
        recall_delta = ((latest['cv_recall'] - best['cv_recall']) / best['cv_recall'] * 100) if best['cv_recall'] > 0 else 0
        print(f"  F1 change          : {f1_delta:+.1f}%")
        print(f"  Recall change      : {recall_delta:+.1f}%")
        print(f"  Models trained     : {len(df)}")

    print("\n")

if __name__ == '__main__':
    main()
