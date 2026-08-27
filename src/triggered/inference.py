import json
import os

import joblib
from preprocess import FEATURE_COLUMNS, prepare_transactions

MODEL_FILENAME = "model.joblib"
CONFIG_FILENAME = "model_config.json"

# Holds the tuned decision threshold loaded by model_fn so that
# predict_fn can apply the same operating point used during training.
# Falls back to 0.5 if no config is present (e.g. when testing an
# older model artifact).
MODEL_CONFIG = {}


def model_fn(model_dir):
    """Load the fitted sklearn Pipeline and its companion config file."""
    global MODEL_CONFIG
    model_path = os.path.join(model_dir, MODEL_FILENAME)
    config_path = os.path.join(model_dir, CONFIG_FILENAME)
    
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Expected model artifact at {model_path}, found: {os.listdir(model_dir)}")

    model = joblib.load(model_path)

    MODEL_CONFIG = {}
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            MODEL_CONFIG = json.load(f)
        print(f"Loaded model config: threshold={MODEL_CONFIG.get('threshold', 0.5)}")
    else:
        print("No model_config.json found, using default threshold 0.5")
    return model


def input_fn(body, content_type="application/json"):
    """Parse the request body and return a DataFrame with model-ready columns."""
    # Strip optional attributes like charset=utf-8
    content_type_clean = content_type.split(";")[0].strip().lower()

    if content_type_clean != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")

    payload = json.loads(body)
    records = payload if isinstance(payload, list) else [payload]

    # prepare_transactions already returns df[FEATURE_COLUMNS]
    return prepare_transactions(records)


def predict_fn(data, model):
    """Return both predicted class and fraud probability."""
    threshold = MODEL_CONFIG.get("threshold", 0.5)
    probs = model.predict_proba(data)[:, 1]
    preds = (probs >= threshold).astype(int)

    return preds, probs


def output_fn(prediction, accept="application/json"):
    """Serialize predictions as JSON."""
    label_map = MODEL_CONFIG.get("label_map", {"0": "Non-Fraud", "1": "Fraud"})
    preds, probs = prediction
    response = [
        {
            "prediction": int(p),
            "label": label_map.get(str(int(p)), "Fraud" if int(p) == 1 else "Non-Fraud"),
            "probability": round(float(b), 4),
        }
        for p, b in zip(preds, probs)
    ]
    return json.dumps(response), "application/json"