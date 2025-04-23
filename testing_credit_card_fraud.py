"""
File: testing_credit_card_fraud.py
Author: Rithvik Nakirikanti
Date: 4/23/25
Description: 
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from testing_funcs import test, f1_score

def load_and_prepare(file):
    df = pd.read_csv(file)

    drop_cols = [
        'trans_date_trans_time', 'cc_num', 'merchant', 'first', 'last',
        'street', 'city', 'state', 'job', 'dob', 'trans_num', 'unix_time'
    ]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    df = pd.get_dummies(df, columns=["category", "gender"], drop_first=True)

    # Normalize
    X = scaler.transform(df.drop(columns=["is_fraud"]))
    y = df["is_fraud"].values.astype(bool)
    return X, y

df_train = pd.read_csv("datasets/credit_card/fraudTrain.csv").head(10000)
df_train = pd.get_dummies(df_train.drop(columns=[
    'trans_date_trans_time', 'cc_num', 'merchant', 'first', 'last',
    'street', 'city', 'state', 'job', 'dob', 'trans_num', 'unix_time'
], errors='ignore'), columns=["category", "gender"], drop_first=True)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(df_train.drop(columns=["is_fraud"]))
y_train = df_train["is_fraud"].values.astype(bool)

# Load and transform test set using same scaler
#X_test, y_test = load_and_prepare("datasets/credit_card/fraudTest.csv").head(5000)
X_test_raw, y_test_raw = load_and_prepare("datasets/credit_card/fraudTest.csv")

# Take only the first 5000 rows
X_test = X_test_raw[:5000]
y_test = y_test_raw[:5000]


methods = ["IncrementalLOF", "DILOF", "MILOF", "KDEWR"]
params = {
    "IncrementalLOF": {"k": 20, "threshold": 1.4},
    "DILOF": {"k": 20, "threshold": 1.4, "window_size": 200, "skipping_enabled": True},
    "MILOF": {"k": 20, "threshold": 1.4, "window_size": 200, "num_clusters": 10},
    "KDEWR": {"threshold": 2, "window_size": 200},
}

results = {}

# Warm-Up & Evaluate
for method in methods:
    print(f"\nRunning: {method}")

    # Warm-up phase with training data (no scoring)
    warmup_predictions, warmup_scores = test(
        data=X_train,
        method=method,
        k=params[method].get("k", 30),
        threshold=params[method]["threshold"],
        window_size=params[method].get("window_size", 200),
        skipping_enabled=params[method].get("skipping_enabled", False),
        num_clusters=params[method].get("num_clusters", 10),
    )

    # Evaluate on test data
    test_predictions, test_scores = test(
        data=X_test,
        method=method,
        k=params[method].get("k", 30),
        threshold=params[method]["threshold"],
        window_size=params[method].get("window_size", 200),
        skipping_enabled=params[method].get("skipping_enabled", False),
        num_clusters=params[method].get("num_clusters", 10),
    )

    f1 = f1_score(test_predictions, y_test)
    results[method] = round(f1, 4)
    print(f"F1 Score: {f1:.4f}")

print("\nSummary of F1 Scores")
for method, score in results.items():
    print(f"{method}: {score:.4f}")