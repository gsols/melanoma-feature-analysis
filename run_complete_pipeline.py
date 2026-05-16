# =============================================================================
# ISIC 2024 — Complete Analysis Pipeline
# =============================================================================
# Runs the entire workflow in sequence:
#   1. isic_data_prep.py      — Data loading, cleaning, preprocessing
#   2. isic_graphs.py         — Generate visualization graphs
#   3. isic_analysis.py       — Statistical analysis & feature ranking
#   4. isic_model.py          — ML model training & evaluation
#
# Usage:
#   python run_complete_pipeline.py
#
# =============================================================================

import subprocess
import sys
import os
from pathlib import Path

# Define script directory
SCRIPT_DIR = Path(__file__).parent
SCRIPTS = [
    ('Step 1: Data Preparation', 'isic_data_prep.py'),
    ('Step 2: Generate Graphs', 'isic_graphs.py'),
    ('Step 3: Statistical Analysis', 'isic_analysis.py'),
    ('Step 4: Model Training', 'isic_model.py'),
]

def run_script(step_name, script_name):
    """Run a Python script and handle errors."""
    script_path = SCRIPT_DIR / script_name
    
    if not script_path.exists():
        print(f"\n❌ ERROR: {script_name} not found at {script_path}")
        return False
    
    print(f"\n{'='*70}")
    print(f"{step_name}")
    print(f"{'='*70}")
    print(f"Running: {script_name}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=SCRIPT_DIR,
            check=True,
            capture_output=False
        )
        print(f"\n✓ {step_name} completed successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR in {script_name}: Exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR running {script_name}: {str(e)}")
        return False


def main():
    """Execute the full pipeline."""
    print("\n" + "="*70)
    print("ISIC 2024 — COMPLETE ANALYSIS PIPELINE")
    print("="*70)
    print(f"Working directory: {SCRIPT_DIR}")
    print(f"Total steps: {len(SCRIPTS)}\n")
    
    completed = []
    failed = []
    
    for step_name, script_name in SCRIPTS:
        if run_script(step_name, script_name):
            completed.append(step_name)
        else:
            failed.append(step_name)
            print(f"\n⚠️  Pipeline halted at {step_name}")
            break
    
    # Final summary
    print(f"\n{'='*70}")
    print("PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"Completed: {len(completed)}/{len(SCRIPTS)}")
    
    if completed:
        print(f"\n✓ Successful steps:")
        for i, step in enumerate(completed, 1):
            print(f"  {i}. {step}")
    
    if failed:
        print(f"\n❌ Failed steps:")
        for i, step in enumerate(failed, 1):
            print(f"  {i}. {step}")
        print(f"\nCheck the error messages above for details.")
        sys.exit(1)
    else:
        print(f"\n✓ All steps completed successfully!")
        print(f"\nOutput files generated:")
        print(f"  • Datasets:  datasetv1.csv, datasetv2.csv, datasetv3.csv")
        print(f"  • Cleaned:   merged_cleaned.csv, merged_cleaned_binned.csv")
        print(f"  • Reports:   reports/ directory")
        print(f"  • Graphs:    graphs/ directory")
        print(f"  • Models:    models/ directory (if created)")
        print(f"\n" + "="*70)
        sys.exit(0)


if __name__ == '__main__':
    main()
