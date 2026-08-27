import os
import sys
import glob
import json
import tarfile
import argparse
import subprocess

# Ensure required libraries are installed before unpickling the model
required_packages = ["imbalanced-learn", "xgboost"]
for package in required_packages:
    try:
        import_name = "imblearn" if package == "imbalanced-learn" else package
        __import__(import_name)
    except ImportError:
        print(f"Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)


def find_file(folder: str, filename: str) -> str:
    expected_path = os.path.join(folder, filename)

    if os.path.exists(expected_path):
        return expected_path

    matches = glob.glob(
        os.path.join(folder, "**", filename),
        recursive=True
    )

    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find {filename} under {folder}. "
        f"Found: {matches}. "
        f"All files: {glob.glob(os.path.join(folder, '**', '*'), recursive=True)}"
    )


def extract_model_tar_if_needed(model_dir: str):
    """
    SageMaker TrainingStep passes model.tar.gz into the evaluation step.
    Extracts archive so model.joblib is available.
    """
    print("Listing model directory before extraction:")
    for root, dirs, files in os.walk(model_dir):
        print(root, files)

    tar_files = glob.glob(
        os.path.join(model_dir, "**", "model.tar.gz"),
        recursive=True
    )

    if not tar_files:
        print("No model.tar.gz found. Looking for model.joblib directly...")
        return

    tar_path = tar_files[0]
    print("Found model.tar.gz:", tar_path)
    print("Extracting model.tar.gz to:", model_dir)

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=model_dir)

    print("Listing model directory after extraction:")
    for root, dirs, files in os.walk(model_dir):
        print(root, files)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-dir",
        type=str,
        default="/opt/ml/processing/model"
    )

    parser.add_argument(
        "--test",
        type=str,
        default="/opt/ml/processing/test"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/opt/ml/processing/evaluation"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Evaluation Environment ===")
    print("Model dir:", args.model_dir)
    print("Test dir:", args.test)
    print("Output dir:", args.output_dir)

    extract_model_tar_if_needed(args.model_dir)

    model_file = find_file(args.model_dir, "model.joblib")
    config_file = find_file(args.model_dir, "model_config.json")
    test_features_file = find_file(args.test, "test_features.csv")
    test_labels_file = find_file(args.test, "test_labels.csv")

    print("Model file:", model_file)
    print("Config file:", config_file)
    print("Test features:", test_features_file)
    print("Test labels:", test_labels_file)

    model = joblib.load(model_file)
    with open(config_file, "r", encoding='utf-8') as f:
        config = json.load(f)

    threshold = config.get('threshold', 0.5)
    target_column = config.get('target_column', 'is_fraud')

    X_test = pd.read_csv(test_features_file)
    y_test = pd.read_csv(test_labels_file).squeeze("columns")

    # Drop target column if present in feature matrix
    X_test = X_test.drop(columns=[target_column], errors='ignore')

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "f1": round(f1_score(y_test, predictions, zero_division=0), 4),
        "precision": round(precision_score(y_test, predictions, zero_division=0), 4),
        "recall": round(recall_score(y_test, predictions, zero_division=0), 4),
    }

    if len(pd.Series(y_test).unique()) >= 2:
        metrics["auc_roc"] = round(roc_auc_score(y_test, probabilities), 4)
    else:
        metrics["auc_roc"] = 0.0

    evaluation_report = {
        "classification_metrics": {
            "auc_roc": {
                "value": metrics["auc_roc"]
            },
            "accuracy": {
                "value": metrics["accuracy"]
            },
            "f1": {
                "value": metrics["f1"]
            },
            "precision": {
                "value": metrics["precision"]
            },
            "recall": {
                "value": metrics["recall"]
            }
        }
    }

    output_path = os.path.join(args.output_dir, "evaluation.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)

    print("Saved evaluation report:", output_path)
    print(json.dumps(evaluation_report, indent=2))


if __name__ == "__main__":
    main()