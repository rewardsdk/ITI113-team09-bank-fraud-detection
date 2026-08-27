import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_COLUMN = "is_fraud"


class PreprocessedModel(BaseEstimator):
    """Wrap a fitted preprocessor + classifier so joblib can serialize it
    and inference.py can call predict_proba(raw features) directly.
    """

    def __init__(self, prep, clf):
        self.prep = prep
        self.clf = clf

    def predict_proba(self, X):
        return self.clf.predict_proba(self.prep.transform(X))

    @property
    def named_steps(self):
        return {"prep": self.prep, "clf": self.clf}


NUMERIC_FEATURES = [
    "hour_of_day",
    "customer_age",
    "credit_score",
    "account_age_years",
    "account_balance",
    "transaction_amount",
    "num_prev_transactions",
    "transaction_freq_monthly",
    "distance_from_home_km",
    "time_since_last_txn_hrs",
    "failed_attempts",
    "log_transaction_amount",
    "log_account_balance",
    "amount_to_balance_ratio",
    "risk_score",
    "failed_x_transaction",
    "night_x_international",
    "pin_x_failed",
    "night_x_pin",
    "amount_x_distance",
    "amount_x_failed",
    "risk_x_amount",
    "risk_x_distance",
    "customer_mean_amount",
    "customer_std_amount",
    "customer_mean_distance",
    "customer_std_distance",
    "customer_txn_count",
    "customer_mean_balance",
    "amount_zscore",
    "distance_zscore",
    "balance_deviation",
]
CATEGORICAL_FEATURES = [
    "is_weekend",
    "is_night_transaction",
    "country",
    "city",
    "merchant_category",
    "payment_method",
    "device_type",
    "is_international",
    "pin_changed_recently",
    "customer_age_group",
    "hour_bin",
    "high_amount",
    "far_from_home",
    "rapid_txn",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DROP_COLS = [
    "transaction_id",
    "customer_id",
    "transaction_date",
    "transaction_time",
    "fraud_type",
]

# Defaults for raw transaction fields, used when a caller omits columns at inference
RAW_DEFAULTS = {
    "transaction_id": "TXN9990000001",
    "customer_id": "CUST99000000",
    "transaction_date": "2024-01-01",
    "transaction_time": "12:00:00",
    "hour_of_day": 12,
    "is_weekend": 0,
    "is_night_transaction": 0,
    "country": "USA",
    "city": "New York",
    "merchant_category": "Grocery",
    "payment_method": "Debit Card",
    "device_type": "Mobile",
    "customer_age": 35,
    "credit_score": 600,
    "account_age_years": 5.0,
    "account_balance": 1000.0,
    "transaction_amount": 100.0,
    "num_prev_transactions": 10,
    "transaction_freq_monthly": 5,
    "distance_from_home_km": 0.0,
    "time_since_last_txn_hrs": 24.0,
    "is_international": 0,
    "failed_attempts": 0,
    "pin_changed_recently": 0,
    "is_fraud": 0,
    "fraud_type": "",
}

# The exact column order of the original bank_fraud.csv, used for headerless CSV inference
RAW_COLUMNS = [
    'transaction_id',
    'customer_id',
    'transaction_date',
    'transaction_time',
    'hour_of_day',
    'is_weekend',
    'is_night_transaction',
    'country',
    'city',
    'merchant_category',
    'payment_method',
    'device_type',
    'customer_age',
    'credit_score',
    'account_age_years',
    'account_balance',
    'transaction_amount',
    'num_prev_transactions',
    'transaction_freq_monthly',
    'distance_from_home_km',
    'time_since_last_txn_hrs',
    'is_international',
    'failed_attempts',
    'pin_changed_recently',
    'is_fraud',
    'fraud_type'
]


def _get_series(df, col, default):
    """Return a column as a Series, or a constant Series if the column is missing."""
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _to_int(series):
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def _to_float(series):
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def engineer_features(df, sort=True, split_year=2022):
    df = df.copy()

    # Build a single datetime column from date and time strings
    if "transaction_datetime" not in df.columns:
        date_series = _get_series(df, "transaction_date", RAW_DEFAULTS["transaction_date"]).astype(str)
        time_series = _get_series(df, "transaction_time", RAW_DEFAULTS["transaction_time"]).astype(str)
        df["transaction_datetime"] = pd.to_datetime(date_series + " " + time_series, errors="coerce")

    # Sorting is only useful during training preprocessing. During real-time
    # inference it would reorder batch requests and misalign predictions with
    # the original input order, so prepare_transactions() calls with sort=False
    if sort:
        df = df.sort_values("transaction_datetime").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Extract year BEFORE dropping columns or returning
    df["year"] = df["transaction_datetime"].dt.year

    # Numeric Feature Engineering
    transaction_amount = _to_float(_get_series(df, "transaction_amount", RAW_DEFAULTS["transaction_amount"]))
    account_balance = _to_float(_get_series(df, "account_balance", RAW_DEFAULTS["account_balance"]))
    distance_from_home_km = _to_float(_get_series(df, "distance_from_home_km", RAW_DEFAULTS["distance_from_home_km"]))
    time_since_last_txn_hrs = _to_float(_get_series(df, "time_since_last_txn_hrs", RAW_DEFAULTS["time_since_last_txn_hrs"]))
    transaction_freq_monthly = _to_float(_get_series(df, "transaction_freq_monthly", RAW_DEFAULTS["transaction_freq_monthly"]))
    account_age_years = _to_float(_get_series(df, "account_age_years", RAW_DEFAULTS["account_age_years"]))
    num_prev_transactions = _to_float(_get_series(df, "num_prev_transactions", RAW_DEFAULTS["num_prev_transactions"]))

    df["log_transaction_amount"] = np.log1p(transaction_amount)
    df["log_account_balance"] = np.log1p(account_balance)
    df["amount_to_balance_ratio"] = transaction_amount / (account_balance + 1.0)

    # Behavioural flags
    df["is_night_transaction"] = _to_int(_get_series(df, "is_night_transaction", RAW_DEFAULTS["is_night_transaction"]))
    df["is_international"] = _to_int(_get_series(df, "is_international", RAW_DEFAULTS["is_international"]))
    df["failed_attempts"] = _to_int(_get_series(df, "failed_attempts", RAW_DEFAULTS["failed_attempts"]))
    df["pin_changed_recently"] = _to_int(_get_series(df, "pin_changed_recently", RAW_DEFAULTS["pin_changed_recently"]))
    df["is_weekend"] = _to_int(_get_series(df, "is_weekend", RAW_DEFAULTS["is_weekend"]))

    df["risk_score"] = (
        df["is_night_transaction"].astype(int)
        + df["is_international"].astype(int)
        + (df["failed_attempts"] > 0).astype(int)
        + df["pin_changed_recently"].astype(int)
    )

    customer_age = _to_float(_get_series(df, "customer_age", RAW_DEFAULTS["customer_age"]))
    df["customer_age_group"] = pd.cut(
        customer_age,
        bins=[0, 25, 40, 60, 100],
        labels=["18-25", "26-40", "41-60", "60+"],
    ).astype(str)

    hour_of_day = _to_float(_get_series(df, "hour_of_day", RAW_DEFAULTS["hour_of_day"]))
    df["hour_bin"] = pd.cut(
        hour_of_day,
        bins=[-1, 5, 11, 17, 23],
        labels=["night_0_5", "morning_6_11", "afternoon_12_17", "evening_18_23"],
    ).astype(str)

    df["failed_x_transaction"] = df["failed_attempts"] * df["is_international"]
    df["night_x_international"] = df["is_night_transaction"] * df["is_international"]
    df["pin_x_failed"] = df["pin_changed_recently"] * df["failed_attempts"]
    df["night_x_pin"] = df["is_night_transaction"] * df["pin_changed_recently"]
    df["amount_x_distance"] = df["log_transaction_amount"] * df["distance_from_home_km"]
    df["amount_x_failed"] = df["log_transaction_amount"] * df["failed_attempts"]
    df["risk_x_amount"] = df["risk_score"] * df["log_transaction_amount"]
    df["risk_x_distance"] = df["risk_score"] * df["distance_from_home_km"]

    train_period = df["transaction_datetime"].dt.year <= split_year
    if train_period.any():
        amount_p90 = df.loc[train_period, "transaction_amount"].quantile(0.90) if "transaction_amount" in df.columns else transaction_amount[train_period].quantile(0.90)
        distance_p90 = distance_from_home_km[train_period].quantile(0.90)
        time_p10 = time_since_last_txn_hrs[train_period].quantile(0.10)
    else:
        amount_p90 = transaction_amount.quantile(0.90)
        distance_p90 = distance_from_home_km.quantile(0.90)
        time_p10 = time_since_last_txn_hrs.quantile(0.10)

    df["high_amount"] = (transaction_amount > amount_p90).astype(int)
    df["far_from_home"] = (distance_from_home_km > distance_p90).astype(int)
    df["rapid_txn"] = (time_since_last_txn_hrs < time_p10).astype(int)

    has_customer_id = "customer_id" in df.columns
    if has_customer_id and train_period.any():
        customer_stats = df[train_period].groupby("customer_id").agg(
            customer_mean_amount=("transaction_amount", "mean"),
            customer_std_amount=("transaction_amount", "std"),
            customer_mean_distance=("distance_from_home_km", "mean"),
            customer_std_distance=("distance_from_home_km", "std"),
            customer_txn_count=("transaction_amount", "count"),
            customer_mean_balance=("account_balance", "mean"),
        ).reset_index()

        customer_stats["customer_std_amount"] = customer_stats["customer_std_amount"].fillna(0)
        customer_stats["customer_std_distance"] = customer_stats["customer_std_distance"].fillna(0)

        # customer behaviour features (use training period only to avoid leakage)
        df = df.merge(customer_stats, on="customer_id", how="left")

        # Global baseline imputation values
        global_mean_amount = transaction_amount[train_period].mean() if train_period.any() else transaction_amount.mean()
        global_std_amount = transaction_amount[train_period].std() if train_period.any() else transaction_amount.std()
        global_mean_distance = distance_from_home_km[train_period].mean() if train_period.any() else distance_from_home_km.mean()
        global_std_distance = distance_from_home_km[train_period].std() if train_period.any() else distance_from_home_km.std() 
        global_mean_balance = account_balance[train_period].mean() if train_period.any() else account_balance.mean()

        # Fill NaNs (unseen customers)
        df["customer_mean_amount"] = df["customer_mean_amount"].fillna(global_mean_amount)
        df["customer_std_amount"] = df["customer_std_amount"].fillna(global_std_amount)
        df["customer_mean_distance"] = df["customer_mean_distance"].fillna(global_mean_distance)
        df["customer_std_distance"] = df["customer_std_distance"].fillna(global_std_distance)
        df["customer_txn_count"] = df["customer_txn_count"].fillna(1)
        df["customer_mean_balance"] = df["customer_mean_balance"].fillna(global_mean_balance)
    else:
        df["customer_mean_amount"] = transaction_amount.mean()
        df["customer_std_amount"] = transaction_amount.std() if len(df) > 1 else 0
        df["customer_mean_distance"] = distance_from_home_km.mean()
        df["customer_std_distance"] = distance_from_home_km.std() if len(df) > 1 else 0
        df["customer_txn_count"] = 1.0
        df["customer_mean_balance"] = account_balance.mean()

    # Z-scores and deviations
    df["amount_zscore"] = (
        (transaction_amount - df["customer_mean_amount"])
        / (df["customer_std_amount"] + 1e-6)
    )
    df["distance_zscore"] = (
        (distance_from_home_km - df["customer_mean_distance"])
        / (df["customer_std_distance"] + 1e-6)
    )
    df["balance_deviation"] = (
        (account_balance - df["customer_mean_balance"])
        / (df["customer_mean_balance"] + 1e-6)
    )

    # Clean up non-feature metadata columns
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    cols_to_drop.append("transaction_datetime")
    df = df.drop(columns=cols_to_drop, errors="ignore")

    return df


def prepare_transactions(data):
    """
    Accept a dict, list of dicts, or DataFrame and return a DataFrame
    with feature columns matched to training outputs.
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        if isinstance(data, dict):
            records = [data]
        else:
            records = list(data)

        if not records:
            return pd.DataFrame(columns=FEATURE_COLUMNS)

        filled = []
        for raw in records:
            record = RAW_DEFAULTS.copy()
            for k, v in raw.items():
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    record[k] = v
            filled.append(record)
        df = pd.DataFrame(filled)

    df = engineer_features(df, sort=False)
    return df[FEATURE_COLUMNS]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-year", type=int, default=2022)
    parser.add_argument("--input-dir", type=str, default=os.environ.get("SM_INPUT_DIR", "/opt/ml/processing/input"))
    parser.add_argument("--output-dir", type=str, default=os.environ.get("SM_OUTPUT_DIR", "/opt/ml/processing/output"))
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = os.path.join(args.input_dir, "bank_fraud.csv")
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading raw data from {input_path}")
    df = pd.read_csv(input_path)

    df = engineer_features(df)

    # Split by year (year was extracted inside engineer_features)
    train_df = df[df["year"] <= args.split_year]
    test_df = df[df["year"] > args.split_year]

    # Save features and labels
    train_df[FEATURE_COLUMNS].to_csv(os.path.join(args.output_dir, "train_features.csv"), index=False)
    train_df[TARGET_COLUMN].to_frame().to_csv(os.path.join(args.output_dir, "train_labels.csv"), index=False)

    test_df[FEATURE_COLUMNS].to_csv(os.path.join(args.output_dir, "test_features.csv"), index=False)
    test_df[TARGET_COLUMN].to_frame().to_csv(os.path.join(args.output_dir, "test_labels.csv"), index=False)

    expected_files = [
        "train_features.csv",
        "train_labels.csv",
        "test_features.csv",
        "test_labels.csv"
    ]

    missing = [f for f in expected_files if not os.path.isfile(os.path.join(args.output_dir, f))]
    if missing:
        raise RuntimeError(f"Missing expected output files: {missing}")

    logger.info(f"Preprocessing complete. Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    logger.info(f"Output files in {args.output_dir}: {expected_files}")


if __name__ == "__main__":
    main()