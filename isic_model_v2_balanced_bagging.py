"""
ISIC 2024 Metadata Model V2 — honest melanoma risk ranking

What this fixes versus the earlier pipeline:
1. The test set contains BOTH benign and malignant rows.
2. It keeps all non-ID metadata columns instead of dropping many TBP features.
3. It trains a balanced undersampling ensemble:
   each bag uses all malignant training rows + a random benign subset.
4. It reports PR-AUC, ROC-AUC, recall, precision, F1, confusion matrix,
   top-k recall, and high-recall operating points.

Usage:
    python isic_model_v2_balanced_bagging.py

Optional faster testing:
    N_BAGS=5 N_ESTIMATORS=80 python isic_model_v2_balanced_bagging.py
"""

from __future__ import annotations

import os
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(os.getenv("DATA_DIR", "."))
METADATA_FILE = DATA_DIR / "metadata.csv"
LABELS_FILE = DATA_DIR / "labels.csv"

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "v2_outputs"))
REPORT_DIR = OUTPUT_DIR / "reports"
GRAPH_DIR = OUTPUT_DIR / "graphs"

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.20"))

# Balanced bagging settings
N_BAGS = int(os.getenv("N_BAGS", "30"))
BENIGN_TO_MALIGNANT_RATIO = int(os.getenv("BENIGN_RATIO", "10"))  # try 5, 10, 20
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", "200"))

# Use patient-level split when possible to reduce patient leakage.
USE_PATIENT_GROUP_SPLIT = os.getenv("USE_PATIENT_GROUP_SPLIT", "1") == "1"

# LightGBM settings tuned for small balanced bags, not the full 400k:393 ratio.
LGBM_PARAMS = {
    "objective": "binary",
    "n_estimators": N_ESTIMATORS,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 8,
    "subsample": 0.90,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.20,
    "reg_lambda": 1.00,
    "max_bin": 63,
    "force_col_wise": True,
    "n_jobs": 1,
    "verbose": -1,
}

DROP_FEATURES = {"isic_id", "patient_id", "malignant"}
ID_COLS_TO_KEEP_IN_OUTPUT = ["isic_id", "patient_id"]

TOP_K_VALUES = [100, 500, 1000, 5000]
TARGET_RECALLS = [0.70, 0.80, 0.90]


# =============================================================================
# HELPERS
# =============================================================================

def make_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)


def make_ohe() -> OneHotEncoder:
    """Compatibility helper for sklearn versions before/after sparse_output."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the previous engineered features while keeping the original TBP columns."""
    df = df.copy()
    eps = 1e-6

    def has(*cols: str) -> bool:
        return all(c in df.columns for c in cols)

    if has("tbp_lv_deltaA", "tbp_lv_deltaB", "tbp_lv_deltaL"):
        df["color_contrast_3d"] = np.sqrt(
            df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2
        )

    if has("clin_size_long_diam_mm", "tbp_lv_minorAxisMM"):
        df["elongation"] = df["clin_size_long_diam_mm"] / (df["tbp_lv_minorAxisMM"] + eps)

    if has("tbp_lv_nevi_confidence", "tbp_lv_color_std_mean"):
        df["nevi_color_tension"] = df["tbp_lv_nevi_confidence"] - df["tbp_lv_color_std_mean"]

    if has("tbp_lv_areaMM2"):
        df["log_area"] = np.log1p(df["tbp_lv_areaMM2"])

    if has("tbp_lv_stdL", "tbp_lv_stdLExt"):
        df["stdL_ratio"] = df["tbp_lv_stdL"] / (df["tbp_lv_stdLExt"] + eps)

    if has("tbp_lv_areaMM2", "tbp_lv_perimeterMM"):
        df["compactness"] = 4 * np.pi * df["tbp_lv_areaMM2"] / (df["tbp_lv_perimeterMM"] ** 2 + eps)

    if has("tbp_lv_C", "tbp_lv_Cext"):
        df["chroma_contrast"] = df["tbp_lv_C"] - df["tbp_lv_Cext"]

    if has("tbp_lv_radial_color_std_max", "tbp_lv_color_std_mean"):
        df["radial_color_ratio"] = df["tbp_lv_radial_color_std_max"] / (
            df["tbp_lv_color_std_mean"] + eps
        )

    return df


def load_data() -> pd.DataFrame:
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Missing {METADATA_FILE}")
    if not LABELS_FILE.exists():
        raise FileNotFoundError(f"Missing {LABELS_FILE}")

    metadata = pd.read_csv(METADATA_FILE)
    labels = pd.read_csv(LABELS_FILE)
    df = metadata.merge(labels, on="isic_id", how="inner")
    df["malignant"] = df["malignant"].astype(int)
    df = add_engineered_features(df)
    return df


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df["malignant"].values

    if USE_PATIENT_GROUP_SPLIT and "patient_id" in df.columns:
        # 5 folds approximates an 80/20 train/test split.
        groups = df["patient_id"].fillna("missing_patient").astype(str).values
        sgkf = StratifiedGroupKFold(n_splits=int(round(1 / TEST_SIZE)), shuffle=True, random_state=RANDOM_STATE)
        train_idx, test_idx = next(sgkf.split(df, y, groups))
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(df)), test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df


def build_preprocessor(train_df: pd.DataFrame, feature_cols: list[str]) -> tuple[ColumnTransformer, list[str], list[str]]:
    # Be strict here: only truly numeric columns should enter the median-imputation path.
    # Pandas may store text columns as object, string, category, or mixed dtype;
    # anything non-numeric must be treated as categorical.
    numeric_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(train_df[c])
    ]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_ohe()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.30,
    )

    return preprocessor, numeric_cols, categorical_cols


def get_transformed_feature_names(preprocessor: ColumnTransformer, numeric_cols: list[str], categorical_cols: list[str]) -> list[str]:
    names = list(numeric_cols)
    if categorical_cols:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        try:
            cat_names = list(ohe.get_feature_names_out(categorical_cols))
        except AttributeError:
            cat_names = list(ohe.get_feature_names(categorical_cols))
        names.extend(cat_names)
    return names


def evaluate_predictions(y_true: np.ndarray, probs: np.ndarray) -> tuple[dict, pd.DataFrame]:
    metrics: dict[str, float | int] = {}
    metrics["n_test"] = int(len(y_true))
    metrics["n_malignant_test"] = int(y_true.sum())
    metrics["n_benign_test"] = int((y_true == 0).sum())
    metrics["pr_auc"] = float(average_precision_score(y_true, probs))
    metrics["roc_auc"] = float(roc_auc_score(y_true, probs))

    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1_values = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.argmax(f1_values))
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    preds = (probs >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    metrics.update({
        "best_f1_threshold": best_threshold,
        "best_f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision_at_best_f1": float(precision_score(y_true, preds, zero_division=0)),
        "recall_at_best_f1": float(recall_score(y_true, preds, zero_division=0)),
        "tp_at_best_f1": int(tp),
        "fp_at_best_f1": int(fp),
        "fn_at_best_f1": int(fn),
        "tn_at_best_f1": int(tn),
    })

    order = np.argsort(probs)[::-1]
    top_rows = []
    for k in TOP_K_VALUES + [int(0.01 * len(y_true)), int(0.05 * len(y_true))]:
        k = max(1, min(int(k), len(y_true)))
        caught = int(y_true[order[:k]].sum())
        top_rows.append({
            "top_k": k,
            "malignant_caught": caught,
            "top_k_recall": caught / max(1, int(y_true.sum())),
            "top_k_precision": caught / k,
        })

    operating_rows = []
    for target_recall in TARGET_RECALLS:
        valid = np.where(recall >= target_recall)[0]
        if len(valid) == 0:
            continue
        # Among thresholds that hit target recall, choose highest precision.
        idx = valid[np.argmax(precision[valid])]
        threshold = float(thresholds[idx]) if idx < len(thresholds) else 0.0
        pred_target = (probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred_target).ravel()
        operating_rows.append({
            "target_recall": target_recall,
            "threshold": threshold,
            "actual_recall": tp / max(1, tp + fn),
            "precision": tp / max(1, tp + fp),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        })

    top_df = pd.DataFrame(top_rows)
    op_df = pd.DataFrame(operating_rows)
    extra_tables = {"top_k": top_df, "operating_points": op_df}
    return metrics, extra_tables


def save_graphs(y_true: np.ndarray, probs: np.ndarray, feature_importance: pd.DataFrame, metrics: dict) -> None:
    precision, recall, _ = precision_recall_curve(y_true, probs)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, linewidth=2, label=f"V2 PR-AUC = {metrics['pr_auc']:.4f}")
    ax.axhline(baseline, linestyle="--", color="gray", label=f"Random baseline = {baseline:.5f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — V2 Balanced Bagging")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / "v2_precision_recall_curve.png", dpi=160)
    plt.close(fig)

    preds = (probs >= metrics["best_f1_threshold"]).astype(int)
    cm = confusion_matrix(y_true, preds)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"]).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix — Best F1 Threshold")
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / "v2_confusion_matrix_best_f1.png", dpi=160)
    plt.close(fig)

    top = feature_importance.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Mean LightGBM importance across bags")
    ax.set_title("Top 25 Features — V2 Balanced Bagging")
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / "v2_feature_importance.png", dpi=160)
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    make_dirs()

    print("=" * 72)
    print("ISIC 2024 — V2 Balanced Bagging Metadata Model")
    print("=" * 72)

    df = load_data()
    print(f"Loaded: {len(df):,} rows x {df.shape[1]} columns")
    print(f"Benign: {(df['malignant'] == 0).sum():,}")
    print(f"Malignant: {df['malignant'].sum():,}")

    feature_cols = [c for c in df.columns if c not in DROP_FEATURES]
    train_df, test_df = split_train_test(df)

    print("\nSplit summary")
    print(f"  Train rows: {len(train_df):,} | malignant: {train_df['malignant'].sum():,} | benign: {(train_df['malignant'] == 0).sum():,}")
    print(f"  Test rows : {len(test_df):,} | malignant: {test_df['malignant'].sum():,} | benign: {(test_df['malignant'] == 0).sum():,}")

    if test_df["malignant"].sum() == 0:
        raise RuntimeError("Test set has no malignant rows. Evaluation would be invalid.")

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(train_df, feature_cols)
    X_train = preprocessor.fit_transform(train_df[feature_cols])
    X_test = preprocessor.transform(test_df[feature_cols])
    y_train = train_df["malignant"].values.astype(int)
    y_test = test_df["malignant"].values.astype(int)

    feature_names = get_transformed_feature_names(preprocessor, numeric_cols, categorical_cols)

    print("\nFeature summary")
    print(f"  Original feature columns: {len(feature_cols)}")
    print(f"  Numeric columns         : {len(numeric_cols)}")
    print(f"  Categorical columns    : {len(categorical_cols)}")
    print(f"  Transformed features   : {X_train.shape[1]}")
    print(f"  Sparse matrix          : {sparse.issparse(X_train)}")

    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    n_benign_per_bag = min(len(neg_idx), BENIGN_TO_MALIGNANT_RATIO * len(pos_idx))

    print("\nTraining balanced bagging LightGBM")
    print(f"  Bags                  : {N_BAGS}")
    print(f"  Per bag malignant     : {len(pos_idx):,}")
    print(f"  Per bag benign        : {n_benign_per_bag:,}")
    print(f"  Per bag ratio         : about {n_benign_per_bag / max(1, len(pos_idx)):.1f}:1")
    print(f"  Estimators per bag    : {N_ESTIMATORS}")

    rng = np.random.default_rng(RANDOM_STATE)
    test_probs = np.zeros(len(test_df), dtype=np.float64)
    importances = []

    for bag in range(N_BAGS):
        benign_sample = rng.choice(neg_idx, size=n_benign_per_bag, replace=False)
        bag_idx = np.concatenate([pos_idx, benign_sample])
        rng.shuffle(bag_idx)
        y_bag = y_train[bag_idx]

        params = dict(LGBM_PARAMS)
        params["scale_pos_weight"] = (y_bag == 0).sum() / max(1, (y_bag == 1).sum())
        params["random_state"] = RANDOM_STATE + bag

        model = LGBMClassifier(**params)
        model.fit(X_train[bag_idx], y_bag)

        test_probs += model.predict_proba(X_test)[:, 1] / N_BAGS
        importances.append(model.feature_importances_)

        if (bag + 1) % max(1, N_BAGS // 10) == 0 or bag == 0:
            print(f"  Finished bag {bag + 1:>3}/{N_BAGS} | elapsed {time.time() - t0:.1f}s")

    metrics, tables = evaluate_predictions(y_test, test_probs)

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": np.mean(importances, axis=0),
        "importance_std": np.std(importances, axis=0),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    importance["rank"] = np.arange(1, len(importance) + 1)

    # Save predictions with useful ID columns if available.
    output_cols = [c for c in ID_COLS_TO_KEEP_IN_OUTPUT if c in test_df.columns]
    output_cols += ["malignant"]
    pred_out = test_df[output_cols].copy()
    pred_out["v2_malignancy_score"] = test_probs
    pred_out = pred_out.sort_values("v2_malignancy_score", ascending=False).reset_index(drop=True)
    pred_out.index += 1
    pred_out.index.name = "risk_rank"

    # Save all outputs.
    with open(REPORT_DIR / "v2_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame([metrics]).to_csv(REPORT_DIR / "v2_metrics_summary.csv", index=False)
    tables["top_k"].to_csv(REPORT_DIR / "v2_top_k_recall.csv", index=False)
    tables["operating_points"].to_csv(REPORT_DIR / "v2_high_recall_operating_points.csv", index=False)
    importance.to_csv(REPORT_DIR / "v2_feature_importance.csv", index=False)
    pred_out.to_csv(REPORT_DIR / "v2_test_predictions_ranked.csv")

    save_graphs(y_test, test_probs, importance, metrics)

    # Human-readable report.
    report_lines = [
        "=" * 72,
        "ISIC 2024 — V2 Balanced Bagging Metadata Model",
        "=" * 72,
        "",
        "DATASET",
        f"  Total rows        : {len(df):,}",
        f"  Benign            : {(df['malignant'] == 0).sum():,}",
        f"  Malignant         : {df['malignant'].sum():,}",
        f"  Train rows        : {len(train_df):,}",
        f"  Test rows         : {len(test_df):,}",
        f"  Test malignant    : {int(y_test.sum()):,}",
        f"  Test benign       : {int((y_test == 0).sum()):,}",
        f"  Patient group split: {USE_PATIENT_GROUP_SPLIT}",
        "",
        "MODEL",
        f"  Algorithm         : LightGBM balanced undersampling ensemble",
        f"  Bags              : {N_BAGS}",
        f"  Benign ratio/bag  : {BENIGN_TO_MALIGNANT_RATIO}:1",
        f"  Estimators/bag    : {N_ESTIMATORS}",
        f"  Original features : {len(feature_cols)} non-ID metadata + engineered features",
        f"  Transformed feats : {X_train.shape[1]}",
        "",
        "MAIN TEST METRICS",
        f"  PR-AUC            : {metrics['pr_auc']:.6f}",
        f"  ROC-AUC           : {metrics['roc_auc']:.6f}",
        f"  Best-F1 threshold : {metrics['best_f1_threshold']:.6f}",
        f"  F1                : {metrics['best_f1']:.6f}",
        f"  Precision         : {metrics['precision_at_best_f1']:.6f}",
        f"  Recall            : {metrics['recall_at_best_f1']:.6f}",
        f"  Confusion matrix  : TP={metrics['tp_at_best_f1']}, FP={metrics['fp_at_best_f1']}, FN={metrics['fn_at_best_f1']}, TN={metrics['tn_at_best_f1']}",
        "",
        "TOP-K RECALL",
        tables["top_k"].to_string(index=False),
        "",
        "HIGH-RECALL OPERATING POINTS",
        "  These are evaluation operating points. In a final system, choose thresholds using validation/CV, not the test set.",
        tables["operating_points"].to_string(index=False),
        "",
        "TOP 20 FEATURES",
        importance.head(20)[["rank", "feature", "importance", "importance_std"]].to_string(index=False),
        "",
        "OUTPUT FILES",
        f"  {REPORT_DIR / 'v2_metrics_summary.csv'}",
        f"  {REPORT_DIR / 'v2_top_k_recall.csv'}",
        f"  {REPORT_DIR / 'v2_high_recall_operating_points.csv'}",
        f"  {REPORT_DIR / 'v2_feature_importance.csv'}",
        f"  {REPORT_DIR / 'v2_test_predictions_ranked.csv'}",
        f"  {GRAPH_DIR / 'v2_precision_recall_curve.png'}",
        f"  {GRAPH_DIR / 'v2_confusion_matrix_best_f1.png'}",
        f"  {GRAPH_DIR / 'v2_feature_importance.png'}",
    ]

    report_text = "\n".join(report_lines)
    with open(REPORT_DIR / "v2_readable_report.txt", "w") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\nDone. Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
