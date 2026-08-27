import argparse
import json
import logging
import os
import subprocess
import sys

# Dynamic dependency resolution for SageMaker pre-built containers
try:
    import imblearn
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imbalanced-learn"])
    import imblearn

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

from preprocess import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_COLUMN = "is_fraud"
MODEL_FILENAME = "model.joblib"
CONFIG_FILENAME = "model_config.json"


def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--n-estimators", "--n_estimators", type=int, default=500, dest="n_estimators")
    parser.add_argument("--max-depth", "--max_depth", type=int, default=8, dest="max_depth")
    parser.add_argument("--learning-rate", "--learning_rate", type=float, default=0.05, dest="learning_rate")
    parser.add_argument("--smote-sampling-strategy", "--smote_sampling_strategy", type=float, default=0.5, dest="smote_sampling_strategy")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)

    # Metadata
    parser.add_argument("--team-id", type=str, default=os.environ.get("TEAM_ID", "unknown-team"))
    parser.add_argument("--student-id", type=str, default=os.environ.get("STUDENT_ID", "s000"))
    parser.add_argument("--semester", type=str, default=os.environ.get("SEMESTER", "26S1"))
    parser.add_argument("--run-name", type=str, default="sagemaker_pipeline_run")

    # SageMaker channels / dirs
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))

    args, _ = parser.parse_known_args()
    return args


def _best_f1_threshold(y_true, y_prob):
    """Return the probability threshold that maximises the F1 score on y_true."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5


def _load_data(channel_dir, split_name):
    """Safely loads feature/label files or splits a single CSV if features/labels aren't separate."""
    features_path = os.path.join(channel_dir, f"{split_name}_features.csv")
    labels_path = os.path.join(channel_dir, f"{split_name}_labels.csv")
    combined_path = os.path.join(channel_dir, f"{split_name}.csv")

    if os.path.exists(features_path) and os.path.exists(labels_path):
        X = pd.read_csv(features_path)
        y = pd.read_csv(labels_path).squeeze("columns")
    elif os.path.exists(combined_path):
        df = pd.read_csv(combined_path)
        y = df[TARGET_COLUMN]
        X = df.drop(columns=[TARGET_COLUMN], errors="ignore")
    else:
        raise FileNotFoundError(f"Could not find valid dataset files in channel directory: {channel_dir}")

    X = X.drop(columns=[TARGET_COLUMN], errors="ignore")
    return X, y


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    X_train, y_train = _load_data(args.train, "train")
    X_test, y_test = _load_data(args.test, "test")

    if len(pd.Series(y_train).unique()) < 2:
        raise ValueError("Training labels contain fewer than two classes.")

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=args.random_state,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), [c for c in NUMERIC_FEATURES if c in X_fit.columns]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
                [c for c in CATEGORICAL_FEATURES if c in X_fit.columns],
            ),
        ],
        remainder="passthrough",
    )

    # Compute actual minority balance safely
    class_counts = pd.Series(y_fit).value_counts()
    minority_count = class_counts.min()
    majority_count = class_counts.max()
    current_ratio = minority_count / majority_count

    # Validate SMOTE pre-conditions
    # Target ratio must strictly exceed current ratio and minority count must be > k_neighbors (default 5)
    target_smote_ratio = args.smote_sampling_strategy
    if current_ratio < target_smote_ratio and minority_count > 5:
        k_neighbors = min(5, minority_count - 1)
        smote_step = SMOTE(
            sampling_strategy=target_smote_ratio,
            k_neighbors=k_neighbors,
            random_state=args.random_state,
        )
        logger.info(f"Applying SMOTE with target ratio {target_smote_ratio:.2f} (current: {current_ratio:.2f})")
    else:
        smote_step = "passthrough"
        logger.info(
            f"Bypassing SMOTE: current ratio ({current_ratio:.2f}) meets/exceeds target ({target_smote_ratio:.2f}) "
            f"or insufficient minority samples ({minority_count})."
        )

    model = ImbPipeline([
        ("prep", preprocessor),
        ("smote", smote_step),
        ("clf", XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=args.random_state,
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=args.n_jobs,
        )),
    ])
    model.fit(X_fit, y_fit)

    val_probs = model.predict_proba(X_val)[:, 1]
    best_threshold = _best_f1_threshold(y_val, val_probs)
    logger.info(f"Best F1 threshold on validation: {best_threshold:.4f}")

    all_metrics = {}
    for split, X, y in [("train", X_fit, y_fit), ("test", X_test, y_test)]:
        probabilities = model.predict_proba(X)[:, 1]
        predictions = (probabilities >= best_threshold).astype(int)

        all_metrics.update(
            {
                f"{split}_accuracy": round(accuracy_score(y, predictions), 4),
                f"{split}_f1": round(f1_score(y, predictions, zero_division=0), 4),
                f"{split}_precision": round(precision_score(y, predictions, zero_division=0), 4),
                f"{split}_recall": round(recall_score(y, predictions, zero_division=0), 4),
            }
        )

        if len(pd.Series(y).unique()) >= 2:
            all_metrics[f"{split}_roc_auc"] = round(roc_auc_score(y, probabilities), 4)
            all_metrics[f"{split}_pr_auc"] = round(average_precision_score(y, probabilities), 4)
        else:
            all_metrics[f"{split}_roc_auc"] = None
            all_metrics[f"{split}_pr_auc"] = None
            logger.warning(f"{split} split has only one class; AUC-ROC unavailable.")

    logger.info("=== Metrics ===")
    for metric_name, metric_value in sorted(all_metrics.items()):
        logger.info(f"  {metric_name:<20}: {metric_value}")

    model_path = os.path.join(args.model_dir, MODEL_FILENAME)
    joblib.dump(model, model_path)
    logger.info(f"Model saved: {model_path}")

    config = {
        "threshold": round(best_threshold, 4),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "label_map": {"0": "Non-Fraud", "1": "Fraud"},
        "model_type": "SMOTE+XGBoost",
    }

    config_path = os.path.join(args.model_dir, CONFIG_FILENAME)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Model config saved: {config_path}")

    if all_metrics["test_roc_auc"] is None:
        raise ValueError("Test AUC-ROC is unavailable; cannot evaluate the quality gate.")

    print(f"test_pr_auc: {all_metrics['test_pr_auc']}")
    print(f"Test AUC-ROC: {all_metrics['test_roc_auc']}")
    print(f"test_accuracy: {all_metrics['test_accuracy']}")
    print(f"test_f1: {all_metrics['test_f1']}")
    print(f"test_recall: {all_metrics['test_recall']}")
    print(f"test_precision: {all_metrics['test_precision']}")
    print(f"best_threshold: {best_threshold:.4f}")


if __name__ == "__main__":
    main()