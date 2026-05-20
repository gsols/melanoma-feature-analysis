"""
ISIC 2024 Metadata Model V3 — repeated patient-group CV balanced bagging

Why this exists:
- V2 was a major improvement because it used a real test set with malignant cases.
- V3 makes the evaluation more defensible by using patient-group cross-validation:
  every fold becomes a held-out test fold, so all rows receive out-of-fold scores.
- Thresholds are selected from an inner validation split, not from the outer test fold.
- The main recommended results remain ranking metrics: PR-AUC and top-k recall.

Usage:
    python isic_model_v3_repeated_group_cv.py

Fast smoke test:
    N_SPLITS=3 N_REPEATS=1 N_BAGS=3 N_ESTIMATORS=40 python isic_model_v3_repeated_group_cv.py

Report run examples:
    BENIGN_RATIO=5  N_SPLITS=5 N_REPEATS=1 N_BAGS=30 N_ESTIMATORS=200 python isic_model_v3_repeated_group_cv.py
    BENIGN_RATIO=10 N_SPLITS=5 N_REPEATS=1 N_BAGS=30 N_ESTIMATORS=200 python isic_model_v3_repeated_group_cv.py
    BENIGN_RATIO=20 N_SPLITS=5 N_REPEATS=1 N_BAGS=30 N_ESTIMATORS=200 python isic_model_v3_repeated_group_cv.py

Optional repeated run for stronger stability:
    BENIGN_RATIO=10 N_SPLITS=5 N_REPEATS=3 N_BAGS=20 N_ESTIMATORS=160 python isic_model_v3_repeated_group_cv.py
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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(os.getenv("DATA_DIR", str((_HERE / "../../datasets").resolve())))
DATASET_FILE = DATA_DIR / "datasetv3_raw_metadata_labels_merged.csv"

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

N_SPLITS = int(os.getenv("N_SPLITS", "5"))
N_REPEATS = int(os.getenv("N_REPEATS", "1"))

# Balanced bagging settings
N_BAGS = int(os.getenv("N_BAGS", "30"))
RATIOS = [5, 10, 20]
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", "200"))

# Per-ratio dirs — updated inside the main() loop
BENIGN_TO_MALIGNANT_RATIO = RATIOS[0]
_V3_OUTPUTS = (_HERE / "../../outputs/v3_outputs").resolve()
OUTPUT_DIR = _V3_OUTPUTS / f"v3_outputs_ratio{BENIGN_TO_MALIGNANT_RATIO}_1"
REPORT_DIR = OUTPUT_DIR / "reports"
GRAPH_DIR = OUTPUT_DIR / "graphs"
PROCESSED_DATA_DIR = _HERE / "processed_data"

# Inner validation thresholding
THRESHOLD_TARGET_RECALL = float(os.getenv("THRESHOLD_TARGET_RECALL", "0.80"))

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

# Same clip ranges used by data_preprocessing_pipeline.py to produce datasetv4.
CLIP_RANGES = {
    "age_approx":              (0,   85),
    "clin_size_long_diam_mm":  (0,  150),
    "tbp_lv_nevi_confidence":  (0,  100),
    "tbp_lv_eccentricity":     (0,    1),
    "tbp_lv_symm_2axis":       (0,    1),
    "tbp_lv_norm_border":      (0,   10),
    "tbp_lv_norm_color":       (0,   10),
    "tbp_lv_area_perim_ratio": (0,  100),
    "tbp_lv_areaMM2":          (0, 5000),
    "tbp_lv_perimeterMM":      (0,  500),
    "tbp_lv_minorAxisMM":      (0,  100),
}

_CAT_COLS_FOR_IMPUTE = ["sex", "anatom_site_general", "tbp_lv_location",
                        "tbp_lv_location_simple", "image_type", "tbp_tile_type"]


# =============================================================================
# HELPERS
# =============================================================================

def make_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def make_ohe() -> OneHotEncoder:
    """Compatibility helper for sklearn versions before/after sparse_output."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
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
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Missing {DATASET_FILE}")

    df = pd.read_csv(DATASET_FILE)
    df["malignant"] = df["malignant"].astype(int)

    # Step 1: impute — same strategy as data_preprocessing_pipeline.py
    cat_src = [c for c in _CAT_COLS_FOR_IMPUTE if c in df.columns]
    num_src = [
        c for c in df.columns
        if c not in cat_src + ["isic_id", "patient_id", "malignant"]
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if num_src:
        df[num_src] = SimpleImputer(strategy="median").fit_transform(df[num_src])
    if cat_src:
        df[cat_src] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_src])

    # Step 2: clip outliers — same ranges as data_preprocessing_pipeline.py
    for col, (lo, hi) in CLIP_RANGES.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    # Step 3: engineer features using the same formulas as data_preprocessing_pipeline.py
    df = add_engineered_features(df)
    return df


def build_preprocessor(train_df: pd.DataFrame, feature_cols: list[str]) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Only truly numeric columns go into median imputation. Everything else is categorical."""
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


def make_cv_splits(df: pd.DataFrame, y: np.ndarray, repeat: int):
    """Prefer patient-group CV. Fall back to normal stratified CV if patient_id is absent."""
    if "patient_id" in df.columns:
        groups = df["patient_id"].fillna("missing_patient").astype(str).values
        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE + repeat,
        )
        return splitter.split(df, y, groups), groups

    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE + repeat,
    )
    return splitter.split(df, y), None


def train_bagging_predict(
    X_train,
    y_train: np.ndarray,
    X_test,
    seed: int,
    feature_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Train balanced bagging ensemble and return averaged scores + mean feature importance."""
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]

    if len(pos_idx) == 0:
        raise RuntimeError("Training fold contains no malignant records.")

    n_benign_per_bag = min(len(neg_idx), BENIGN_TO_MALIGNANT_RATIO * len(pos_idx))

    rng = np.random.default_rng(seed)
    probs = np.zeros(X_test.shape[0], dtype=np.float64)
    importances = []

    for bag in range(N_BAGS):
        benign_sample = rng.choice(neg_idx, size=n_benign_per_bag, replace=False)
        bag_idx = np.concatenate([pos_idx, benign_sample])
        rng.shuffle(bag_idx)

        y_bag = y_train[bag_idx]
        params = dict(LGBM_PARAMS)
        params["scale_pos_weight"] = (y_bag == 0).sum() / max(1, (y_bag == 1).sum())
        params["random_state"] = seed + bag

        model = LGBMClassifier(**params)
        model.fit(X_train[bag_idx], y_bag)

        probs += model.predict_proba(X_test)[:, 1] / N_BAGS
        importances.append(model.feature_importances_)

    if importances:
        mean_importance = np.mean(importances, axis=0)
    else:
        mean_importance = np.zeros(feature_count)

    return probs, mean_importance


def choose_threshold_from_inner_validation(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
) -> tuple[float, dict]:
    """
    Choose a threshold from inner validation only.
    This avoids choosing a classification threshold using the outer held-out test fold.
    """
    y = train_df["malignant"].values.astype(int)

    if y.sum() < 2:
        return 0.5, {"inner_note": "not enough malignant cases; used 0.5"}

    # One inner split is enough to avoid test-threshold leakage without doubling the full runtime too much.
    if "patient_id" in train_df.columns:
        groups = train_df["patient_id"].fillna("missing_patient").astype(str).values
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        inner_train_idx, inner_val_idx = next(inner.split(train_df, y, groups))
    else:
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        inner_train_idx, inner_val_idx = next(inner.split(train_df, y))

    inner_train_df = train_df.iloc[inner_train_idx].reset_index(drop=True)
    inner_val_df = train_df.iloc[inner_val_idx].reset_index(drop=True)

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(inner_train_df, feature_cols)
    X_inner_train = preprocessor.fit_transform(inner_train_df[feature_cols])
    X_inner_val = preprocessor.transform(inner_val_df[feature_cols])

    y_inner_train = inner_train_df["malignant"].values.astype(int)
    y_inner_val = inner_val_df["malignant"].values.astype(int)

    if y_inner_val.sum() == 0:
        return 0.5, {"inner_note": "inner validation had no malignant cases; used 0.5"}

    probs_val, _ = train_bagging_predict(
        X_inner_train,
        y_inner_train,
        X_inner_val,
        seed=seed + 10_000,
        feature_count=X_inner_train.shape[1],
    )

    precision, recall, thresholds = precision_recall_curve(y_inner_val, probs_val)

    valid = np.where(recall >= THRESHOLD_TARGET_RECALL)[0]
    if len(valid) > 0:
        # Among thresholds reaching the target recall, choose the one with best precision.
        idx = valid[np.argmax(precision[valid])]
        threshold = float(thresholds[idx]) if idx < len(thresholds) else 0.0
        strategy = f"max precision while recall >= {THRESHOLD_TARGET_RECALL:.2f}"
    else:
        # Fallback to best F1 on inner validation.
        f1_values = 2 * precision * recall / (precision + recall + 1e-12)
        idx = int(np.argmax(f1_values))
        threshold = float(thresholds[idx]) if idx < len(thresholds) else 0.5
        strategy = "best F1 fallback"

    preds_val = (probs_val >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_inner_val, preds_val).ravel()

    details = {
        "inner_strategy": strategy,
        "inner_threshold": threshold,
        "inner_pr_auc": float(average_precision_score(y_inner_val, probs_val)),
        "inner_recall": float(recall_score(y_inner_val, preds_val, zero_division=0)),
        "inner_precision": float(precision_score(y_inner_val, preds_val, zero_division=0)),
        "inner_f1": float(f1_score(y_inner_val, preds_val, zero_division=0)),
        "inner_tp": int(tp),
        "inner_fp": int(fp),
        "inner_fn": int(fn),
        "inner_tn": int(tn),
    }

    return threshold, details


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    return {
        "n": int(len(y_true)),
        "n_malignant": int(y_true.sum()),
        "n_benign": int((y_true == 0).sum()),
        "prevalence": float(y_true.mean()),
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def compute_top_k(y_true: np.ndarray, probs: np.ndarray, prefix: str = "") -> pd.DataFrame:
    order = np.argsort(probs)[::-1]
    rows = []
    total_mal = max(1, int(y_true.sum()))

    for k in TOP_K_VALUES + [int(0.01 * len(y_true)), int(0.05 * len(y_true))]:
        k = max(1, min(int(k), len(y_true)))
        caught = int(y_true[order[:k]].sum())
        rows.append({
            f"{prefix}top_k": k,
            f"{prefix}malignant_caught": caught,
            f"{prefix}top_k_recall": caught / total_mal,
            f"{prefix}top_k_precision": caught / k,
        })

    return pd.DataFrame(rows)


def summarize_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        rows.append({
            "metric": col,
            "mean": float(df[col].mean()),
            "std": float(df[col].std(ddof=1)) if len(df) > 1 else 0.0,
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        })
    return pd.DataFrame(rows)


def save_graphs(y_true: np.ndarray, probs: np.ndarray, global_metrics: dict, feature_importance: pd.DataFrame) -> None:
    precision, recall, _ = precision_recall_curve(y_true, probs)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, linewidth=2, label=f"V3 OOF PR-AUC = {global_metrics['pr_auc']:.4f}")
    ax.axhline(baseline, linestyle="--", color="gray", label=f"Random baseline = {baseline:.5f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — V3 Out-of-Fold Scores")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / "v3_oof_precision_recall_curve.png", dpi=160)
    plt.close(fig)

    preds = (probs >= global_metrics["threshold"]).astype(int)
    cm = confusion_matrix(y_true, preds)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title("Confusion Matrix — V3 Inner-Validation Threshold")
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / "v3_oof_confusion_matrix.png", dpi=160)
    plt.close(fig)

    top = feature_importance.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Mean LightGBM importance across outer folds and bags")
    ax.set_title("Top 25 Features — V3 Balanced Bagging CV")
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / "v3_feature_importance.png", dpi=160)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    global BENIGN_TO_MALIGNANT_RATIO, OUTPUT_DIR, REPORT_DIR, GRAPH_DIR
    t0 = time.time()

    print("=" * 76)
    print("ISIC 2024 — V3 Repeated Patient-Group CV Balanced Bagging")
    print("=" * 76)
    print(f"\nRatios to run: {RATIOS}")

    # Load data once — shared across all ratio runs
    df = load_data()
    y_all = df["malignant"].values.astype(int)
    feature_cols = [c for c in df.columns if c not in DROP_FEATURES]

    print(f"\nLoaded: {len(df):,} rows x {df.shape[1]} columns")
    print(f"Benign: {(df['malignant'] == 0).sum():,}")
    print(f"Malignant: {df['malignant'].sum():,}")
    print(f"Original feature columns: {len(feature_cols)}")
    print(f"CV: {N_SPLITS} folds x {N_REPEATS} repeat(s)")
    print(f"Threshold strategy: inner validation, target recall {THRESHOLD_TARGET_RECALL:.2f}")

    # Save processed data once
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_DIR / "v3_processed.csv", index=False)
    print(f"✓ Processed data saved to {PROCESSED_DATA_DIR}")

    # =========================================================================
    # MAIN LOOP — iterate over each ratio
    # =========================================================================

    for ratio in RATIOS:
        BENIGN_TO_MALIGNANT_RATIO = ratio
        OUTPUT_DIR = _V3_OUTPUTS / f"v3_outputs_ratio{ratio}_1"
        REPORT_DIR = OUTPUT_DIR / "reports"
        GRAPH_DIR = OUTPUT_DIR / "graphs"
        make_dirs()

        print(f"\n{'='*76}")
        print(f"RATIO {ratio}:1  (benign:malignant)")
        print(f"{'='*76}")
        print(f"Bagging: {N_BAGS} bags, benign ratio {ratio}:1, estimators {N_ESTIMATORS}")
        print()

        all_prediction_frames = []
        fold_metric_rows = []
        threshold_rows = []
        topk_fold_rows = []
        feature_importance_rows = []

        for repeat in range(N_REPEATS):
            splits, groups = make_cv_splits(df, y_all, repeat)

            for fold, (train_idx, test_idx) in enumerate(splits, start=1):
                fold_start = time.time()
                train_df = df.iloc[train_idx].reset_index(drop=True)
                test_df = df.iloc[test_idx].reset_index(drop=True)

                y_train = train_df["malignant"].values.astype(int)
                y_test = test_df["malignant"].values.astype(int)

                print("-" * 76)
                print(
                    f"Repeat {repeat + 1}/{N_REPEATS}, fold {fold}/{N_SPLITS} | "
                    f"train malignant={y_train.sum():,}, test malignant={y_test.sum():,}"
                )

                if y_test.sum() == 0:
                    print("  WARNING: held-out fold has no malignant cases; skipping fold metrics.")
                    continue

                threshold, threshold_details = choose_threshold_from_inner_validation(
                    train_df=train_df,
                    feature_cols=feature_cols,
                    seed=RANDOM_STATE + repeat * 1000 + fold * 100,
                )

                preprocessor, numeric_cols, categorical_cols = build_preprocessor(train_df, feature_cols)
                X_train = preprocessor.fit_transform(train_df[feature_cols])
                X_test = preprocessor.transform(test_df[feature_cols])
                feature_names = get_transformed_feature_names(preprocessor, numeric_cols, categorical_cols)

                print(f"  Transformed features: {X_train.shape[1]} | sparse={sparse.issparse(X_train)}")
                print(f"  Inner threshold: {threshold:.6f}")

                probs, importance = train_bagging_predict(
                    X_train,
                    y_train,
                    X_test,
                    seed=RANDOM_STATE + repeat * 1000 + fold * 100,
                    feature_count=X_train.shape[1],
                )

                metrics = compute_metrics(y_test, probs, threshold)
                metrics.update({
                    "repeat": repeat + 1,
                    "fold": fold,
                    "train_n": int(len(train_df)),
                    "train_malignant": int(y_train.sum()),
                    "train_benign": int((y_train == 0).sum()),
                    "elapsed_seconds": float(time.time() - fold_start),
                })
                fold_metric_rows.append(metrics)

                threshold_row = {"repeat": repeat + 1, "fold": fold}
                threshold_row.update(threshold_details)
                threshold_rows.append(threshold_row)

                topk = compute_top_k(y_test, probs)
                topk.insert(0, "fold", fold)
                topk.insert(0, "repeat", repeat + 1)
                topk_fold_rows.append(topk)

                for name, imp in zip(feature_names, importance):
                    feature_importance_rows.append({
                        "repeat": repeat + 1,
                        "fold": fold,
                        "feature": name,
                        "importance": float(imp),
                    })

                output_cols = [c for c in ID_COLS_TO_KEEP_IN_OUTPUT if c in test_df.columns]
                pred = test_df[output_cols + ["malignant"]].copy()
                pred["repeat"] = repeat + 1
                pred["fold"] = fold
                pred["v3_malignancy_score"] = probs
                pred["inner_validation_threshold"] = threshold
                pred["v3_predicted_positive"] = (probs >= threshold).astype(int)
                all_prediction_frames.append(pred)

                print(
                    f"  Fold metrics: PR-AUC={metrics['pr_auc']:.5f}, ROC-AUC={metrics['roc_auc']:.5f}, "
                    f"Precision={metrics['precision']:.5f}, Recall={metrics['recall']:.5f}, F1={metrics['f1']:.5f}"
                )
                print(f"  Finished fold in {time.time() - fold_start:.1f}s")

        if not fold_metric_rows:
            print(f"  WARNING: No valid folds completed for ratio {ratio}. Skipping.")
            continue

        fold_metrics = pd.DataFrame(fold_metric_rows)
        threshold_df = pd.DataFrame(threshold_rows)
        topk_by_fold = pd.concat(topk_fold_rows, ignore_index=True) if topk_fold_rows else pd.DataFrame()

        preds_all = pd.concat(all_prediction_frames, ignore_index=True)
        preds_all_sorted = preds_all.sort_values("v3_malignancy_score", ascending=False).reset_index(drop=True)
        preds_all_sorted.index += 1
        preds_all_sorted.index.name = "risk_rank"

        if "isic_id" in preds_all.columns and N_REPEATS > 1:
            agg_cols = ["isic_id"]
            keep_cols = [c for c in ["patient_id"] if c in preds_all.columns]
            pred_mean = (
                preds_all
                .groupby(agg_cols, as_index=False)
                .agg({
                    **{c: "first" for c in keep_cols},
                    "malignant": "first",
                    "v3_malignancy_score": "mean",
                    "v3_predicted_positive": "mean",
                })
            )
            y_global = pred_mean["malignant"].values.astype(int)
            probs_global = pred_mean["v3_malignancy_score"].values
            threshold_global = float(fold_metrics["threshold"].median())
            pred_mean["v3_predicted_positive"] = (pred_mean["v3_malignancy_score"] >= threshold_global).astype(int)
            pred_mean = pred_mean.sort_values("v3_malignancy_score", ascending=False).reset_index(drop=True)
            pred_mean.index += 1
            pred_mean.index.name = "risk_rank"
        else:
            pred_mean = preds_all_sorted.copy()
            y_global = pred_mean["malignant"].values.astype(int)
            probs_global = pred_mean["v3_malignancy_score"].values
            threshold_global = float(fold_metrics["threshold"].median())

        global_metrics = compute_metrics(y_global, probs_global, threshold_global)
        topk_global = compute_top_k(y_global, probs_global)

        metric_summary = summarize_numeric(
            fold_metrics,
            ["pr_auc", "roc_auc", "f1", "precision", "recall", "tp", "fp", "fn", "tn"],
        )

        feature_importance = (
            pd.DataFrame(feature_importance_rows)
            .groupby("feature", as_index=False)
            .agg(
                importance=("importance", "mean"),
                importance_std=("importance", "std"),
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        feature_importance["rank"] = np.arange(1, len(feature_importance) + 1)

        fold_metrics.to_csv(REPORT_DIR / "v3_cv_metrics_by_fold.csv", index=False)
        metric_summary.to_csv(REPORT_DIR / "v3_cv_metric_mean_std.csv", index=False)
        threshold_df.to_csv(REPORT_DIR / "v3_inner_threshold_details.csv", index=False)
        topk_by_fold.to_csv(REPORT_DIR / "v3_top_k_by_fold.csv", index=False)
        topk_global.to_csv(REPORT_DIR / "v3_top_k_global_oof.csv", index=False)
        feature_importance.to_csv(REPORT_DIR / "v3_feature_importance.csv", index=False)
        preds_all_sorted.to_csv(REPORT_DIR / "v3_oof_predictions_all_repeats_ranked.csv")
        pred_mean.to_csv(REPORT_DIR / "v3_oof_predictions_mean_ranked.csv")

        with open(REPORT_DIR / "v3_global_metrics.json", "w") as f:
            json.dump(global_metrics, f, indent=2)

        pd.DataFrame([global_metrics]).to_csv(REPORT_DIR / "v3_global_metrics.csv", index=False)

        save_graphs(y_global, probs_global, global_metrics, feature_importance)

        report_lines = [
            "=" * 76,
            f"ISIC 2024 — V3 Repeated Patient-Group CV Balanced Bagging ({ratio}:1 Ratio)",
            "=" * 76,
            "",
            "DATASET",
            f"  Total rows         : {len(df):,}",
            f"  Benign             : {(df['malignant'] == 0).sum():,}",
            f"  Malignant          : {df['malignant'].sum():,}",
            f"  Patient-group CV   : {'patient_id' in df.columns}",
            "",
            "MODEL",
            f"  CV design          : {N_SPLITS} folds x {N_REPEATS} repeat(s)",
            f"  Algorithm          : LightGBM balanced undersampling ensemble",
            f"  Bags per fold      : {N_BAGS}",
            f"  Benign ratio/bag   : {ratio}:1",
            f"  Estimators/bag     : {N_ESTIMATORS}",
            f"  Original features  : {len(feature_cols)} non-ID metadata + engineered features",
            "",
            "THRESHOLD POLICY",
            "  Thresholds for fold-level classification were selected from inner validation,",
            "  not from the outer held-out test fold.",
            f"  Inner target recall: {THRESHOLD_TARGET_RECALL:.2f}",
            f"  Global report threshold: median inner threshold = {threshold_global:.6f}",
            "",
            "GLOBAL OUT-OF-FOLD METRICS",
            f"  PR-AUC             : {global_metrics['pr_auc']:.6f}",
            f"  ROC-AUC            : {global_metrics['roc_auc']:.6f}",
            f"  F1                 : {global_metrics['f1']:.6f}",
            f"  Precision          : {global_metrics['precision']:.6f}",
            f"  Recall             : {global_metrics['recall']:.6f}",
            f"  Confusion matrix   : TP={global_metrics['tp']}, FP={global_metrics['fp']}, FN={global_metrics['fn']}, TN={global_metrics['tn']}",
            "",
            "MEAN ± STD ACROSS OUTER FOLDS",
            metric_summary.to_string(index=False),
            "",
            "GLOBAL TOP-K OUT-OF-FOLD RECALL",
            topk_global.to_string(index=False),
            "",
            "TOP 20 FEATURES",
            feature_importance.head(20)[["rank", "feature", "importance", "importance_std"]].to_string(index=False),
            "",
            "OUTPUT FILES",
            f"  {PROCESSED_DATA_DIR / 'v3_processed.csv'}",
            f"  {REPORT_DIR / 'v3_global_metrics.csv'}",
            f"  {REPORT_DIR / 'v3_cv_metrics_by_fold.csv'}",
            f"  {REPORT_DIR / 'v3_cv_metric_mean_std.csv'}",
            f"  {REPORT_DIR / 'v3_top_k_global_oof.csv'}",
            f"  {REPORT_DIR / 'v3_feature_importance.csv'}",
            f"  {REPORT_DIR / 'v3_oof_predictions_mean_ranked.csv'}",
            f"  {GRAPH_DIR / 'v3_oof_precision_recall_curve.png'}",
            f"  {GRAPH_DIR / 'v3_oof_confusion_matrix.png'}",
            f"  {GRAPH_DIR / 'v3_feature_importance.png'}",
        ]

        report_text = "\n".join(report_lines)
        with open(REPORT_DIR / "v3_readable_report.txt", "w") as f:
            f.write(report_text)

        print("\n" + report_text)

        print(f"\n{'='*76}")
        print(f"RATIO {ratio}:1 COMPLETE  |  Output: {OUTPUT_DIR}")
        print(f"{'='*76}")

    print(f"\nCompleted {len(RATIOS)} ratio run(s): {RATIOS}")
    print(f"Done. Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
