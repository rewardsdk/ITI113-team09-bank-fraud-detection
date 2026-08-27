import argparse
import json
import logging
import os

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
from xgboost import XGBClassifier

from preprocess import NUMERIC_FEATURES, CATEGORICAL_FEATURES, FEATURE_COLUMNS, PreprocessedModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_COLUMN = "is_fraud"

MODEL_FILENAME = "model.joblib"
CONFIG_FILENAME = "model_config.json"


def parse_args():
    parser = argparse.ArgumentParser()

    # RandomForest hyperparameters
    # XGBoost hyperparameters (with aliases for hyphen/underscore formats)
    parser.add_argument(
        "--n-estimators",
        "--n_estimators",
        type=int,
        default=500,
        dest="n_estimators",
    )
    parser.add_argument(
        "--max-depth",
        "--max_depth",
        type=int,
        default=8,
        dest="max_depth",
    )
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument(
        "--colsample-bytree",
        "--colsample_bytree",
        type=float,
        default=0.8,
        dest="colsample_bytree",
    )
    parser.add_argument(
        "--min-child-weight",
        "--min_child_weight",
        type=int,
        default=5,
        dest="min_child_weight",
    )
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument(
        "--reg-alpha",
        "--reg_alpha",
        type=float,
        default=0.1,
        dest="reg_alpha",
    )
    parser.add_argument(
        "--reg-lambda",
        "--reg_lambda",
        type=float,
        default=1.0,
        dest="reg_lambda",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        "--early_stopping_rounds",
        type=int,
        default=30,
        dest="early_stopping_rounds",
    )
    parser.add_argument(
        "--learning-rate",
        "--learning_rate",
        type=float,
        default=0.05,
        dest="learning_rate",
    )
    parser.add_argument(
        "--smote-sampling-strategy",
        "--smote_sampling_strategy",
        type=float,
        default=0.5,
        dest="smote_sampling_strategy",
    )
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

    return parser.parse_args()


def _best_f1_threshold(y_true, y_prob):
    """Return the probability threshold that maximises the F1 score on y_true."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5


def main():
    args = parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    X_train = pd.read_csv(os.path.join(args.train, "train_features.csv"))
    y_train = pd.read_csv(os.path.join(args.train, "train_labels.csv")).squeeze("columns")
    X_test = pd.read_csv(os.path.join(args.test, "test_features.csv"))
    y_test = pd.read_csv(os.path.join(args.test, "test_labels.csv")).squeeze("columns")

    X_train = X_train.drop(columns=[TARGET_COLUMN], errors="ignore")
    X_test = X_test.drop(columns=[TARGET_COLUMN], errors="ignore")

    if len(pd.Series(y_train).unique()) < 2:
        raise ValueError("Training labels contain fewer than two classes.")

    # Hold out a validation set from train or threshold tuning. This keeps the
    # test set untouched and mirros the threshold tuning approach in Notebook 2
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=args.random_state,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    # Step 1: Preprocess
    preprocessor.fit(X_fit)
    X_fit_prep = preprocessor.transform(X_fit)
    X_val_prep = preprocessor.transform(X_val)

    # Step 2: SMOTE on training data only
    smote = SMOTE(random_state=args.random_state, sampling_strategy=args.smote_sampling_strategy)
    X_fit_res, y_fit_res = smote.fit_resample(X_fit_prep, y_fit)
    logger.info(f"SMOTE: {X_fit_prep.shape[0]} -> {X_fit_res.shape[0]} rows")

    clf = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        early_stopping_rounds=args.early_stopping_rounds,

    )
    clf.fit(X_fit_res, y_fit_res, eval_set=[(X_val_prep, y_val)], verbose=False)
    logger.info(f"Early stopping: best iteration {clf.best_iteration} / {args.n_estimators}")

    model = PreprocessedModel(preprocessor, clf)

    # Tune the decision threshold on the validation set. With class_weight='balanced'
    # and a 5.5% fraud rate, the default 0.5 threshold predicts far too many
    # positives and makes test_accuracy look like ~50%. A threshold tuned for
    # F1 raises accuracy to ~80% while still catching ~40% of fraud cases.
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

    # Save model artifact using joblib as defined in MODEL_FILENAME
    model_path = os.path.join(args.model_dir, MODEL_FILENAME)
    joblib.dump(model, model_path)
    logger.info(f"Model saved: {model_path}")

    config = {
        "threshold": round(best_threshold, 4),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "label_map": {"0": "Non-Fraud", "1": "Fraud"},
        "model_type": "SMOTE+XGBoost",
        "best_iteration": int(clf.best_iteration) if clf.best_iteration is not None else args.n_estimators,
    }
    config_path = os.path.join(args.model_dir, CONFIG_FILENAME)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Model config saved: {config_path}")

    # Check metric using matching key 'test_roc_auc'
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
