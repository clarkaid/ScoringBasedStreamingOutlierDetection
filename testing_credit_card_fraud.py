"""
File: testing_credit_card_fraud.py
Author: Rithvik Nakirikanti
Date: 4/27/25
Description: Test IncrementalLOF, DILOF, MILOF, and KDEWR on a selected cardholder's transactions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from testing_funcs import test, f1_score
from sklearn.preprocessing import MinMaxScaler

df_train = pd.read_csv("datasets/credit_card/fraudTrain.csv")
df_test = pd.read_csv("datasets/credit_card/fraudTest.csv")
df_full = pd.concat([df_train, df_test], ignore_index=True)

fraud_counts = df_full[df_full["is_fraud"] == 1]["cc_num"].value_counts()
fraud_counts = fraud_counts[fraud_counts >= 3]
chosen_cc_num = fraud_counts.idxmax()


df_cardholder = df_full[df_full["cc_num"] == chosen_cc_num]
df_cardholder = df_cardholder.sample(frac=1.0, random_state=42)
split_idx = int(0.7 * len(df_cardholder))
df_train = df_cardholder.iloc[:split_idx]
df_test = df_cardholder.iloc[split_idx:]

def load_and_prepare(df, scaler=None):
    df = df.copy()
    drop_cols = [
        'trans_date_trans_time', 'merchant', 'first', 'last',
        'street', 'city', 'state', 'job', 'dob', 'trans_num', 'unix_time'
    ]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    df.drop(columns=['cc_num'], inplace=True, errors='ignore')
    df = pd.get_dummies(df, columns=["category", "gender"], drop_first=True)
    if scaler is None:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(df.drop(columns=["is_fraud"]))
    else:
        X = scaler.transform(df.drop(columns=["is_fraud"]))
    y = df["is_fraud"].values.astype(bool)
    return X, y, scaler

X_train, y_train, scaler = load_and_prepare(df_train)
X_test, y_test, _ = load_and_prepare(df_test, scaler)


window_sizes = [100, 200, 300, 400, 500]

methods = ["IncrementalLOF", "DILOF", "MILOF", "KDEWR"]
colors = {
    "IncrementalLOF": "black",
    "DILOF": "red",
    "MILOF": "blue",
    "KDEWR": "green",
}
f1_scores = {method: [] for method in methods}

for window_size in window_sizes:
    print(f"\nWindow size: {window_size}")
    for method in methods:
        print(f"  Running: {method}")
        _warmup_preds, _warmup_scores = test(
            data=X_train,
            method=method,
            k=20,
            threshold=1.4 if method != "KDEWR" else 2,
            window_size=window_size,
            skipping_enabled=(method == "DILOF"),
            num_clusters=10
        )

        # Test
        test_preds, test_scores = test(
            data=X_test,
            method=method,
            k=20,
            threshold=1.4 if method != "KDEWR" else 2,
            window_size=window_size,
            skipping_enabled=(method == "DILOF"),
            num_clusters=10
        )

        try:
            f1 = f1_score(test_preds, y_test)
        except ZeroDivisionError:
            f1 = 0.0

        f1_scores[method].append(f1)
        print(f"    F1 Score: {f1:.4f}")

plt.figure(figsize=(10, 6))

for method in methods:
    plt.plot(
        window_sizes, f1_scores[method],
        marker='o',
        color=colors[method],
        label=method,
        markersize=7,
        linewidth=2
    )

plt.title("Performance on Credit Card Fraud Dataset", fontsize=14)
plt.xlabel("Window Size", fontsize=12)
plt.ylabel("F1-Score", fontsize=12)
plt.xticks(window_sizes)
plt.ylim(-0.02, 0.2)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()