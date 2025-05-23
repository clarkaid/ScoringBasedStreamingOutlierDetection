import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from kdwer import KDEWRStreaming
from milof import milof
from dilof import DILOF
from IncrementalLOF import IncrementalLOF

def f1_score(estimated_idxs, true_idxs, total_len):
    estimated = [0] * total_len
    actual = [0] * total_len
    for i in estimated_idxs:
        estimated[i] = 1
    for i in true_idxs:
        actual[i] = 1

    tp = sum(1 for i in range(total_len) if estimated[i] == 1 and actual[i] == 1)
    fp = sum(1 for i in range(total_len) if estimated[i] == 1 and actual[i] == 0)
    fn = sum(1 for i in range(total_len) if estimated[i] == 0 and actual[i] == 1)

    return 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0

# Load and preprocess
df = pd.read_csv("./datasets/NF-UQ-NIDS-V2.csv").head(1000)
labels = df["Label"].values if "Label" in df.columns else np.zeros(len(df))
drop_cols = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label', 'Attack', 'Dataset']
df_clean = df.drop(columns=drop_cols, errors='ignore')
df_clean = df_clean.select_dtypes(include=[np.number]).fillna(0)
data = MinMaxScaler().fit_transform(df_clean)
true_outlier_idxs = [i for i, lbl in enumerate(labels) if lbl == 1]

# Parameter sets
kdewr_params = [
    {"windowSize": 50, "r": 1, "xi": 0.5, "forgetting": 0.9},
    {"windowSize": 75, "r": 1, "xi": 0.3, "forgetting": 0.95},
    {"windowSize": 100, "r": 2, "xi": 0.2, "forgetting": 0.92},
]

milof_params = [
    {"k": 10, "b": 100, "c": 10, "threshold": 1.5},
    {"k": 15, "b": 120, "c": 15, "threshold": 1.3},
    {"k": 20, "b": 150, "c": 20, "threshold": 1.2},
]

dilof_params = [
    {"k": 10, "window_size": 100, "threshold": 1.5, "step_size": 0.01, "regularizer": 0.1, "max_iter": 10},
    {"k": 15, "window_size": 120, "threshold": 1.3, "step_size": 0.01, "regularizer": 0.1, "max_iter": 10},
    {"k": 20, "window_size": 150, "threshold": 1.2, "step_size": 0.01, "regularizer": 0.1, "max_iter": 10},
]

inclof_params = [
    {"k": 10, "threshold": 1.5},
    {"k": 15, "threshold": 1.3},
    {"k": 20, "threshold": 1.2},
]

summary = {}

# Run KDEWR
for i, params in enumerate(kdewr_params, 1):
    name = f"KDEWR-{i}"
    model = KDEWRStreaming(
        windowSize=params["windowSize"],
        dim=data.shape[1],
        r=params["r"],
        xi=params["xi"],
        forgetting=params["forgetting"]
    )
    print(f"\nRunning {name} with {params}")
    detected = [i for i, point in enumerate(data) if model.insert(point)[0]]
    correct = set(detected).intersection(true_outlier_idxs)
    f1 = f1_score(detected, true_outlier_idxs, len(data))
    summary[name] = {
        "detected": len(detected),
        "true": len(true_outlier_idxs),
        "correct": len(correct),
        "accuracy": f"{len(correct)} / {len(true_outlier_idxs)}",
        "f1_score": round(f1, 3)
    }

# Run MiLOF
for i, params in enumerate(milof_params, 1):
    name = f"MiLOF-{i}"
    model = milof(**params)
    print(f"\nRunning {name} with {params}")
    detected = [i for i, point in enumerate(data) if model.insert(point)[0]]
    correct = set(detected).intersection(true_outlier_idxs)
    f1 = f1_score(detected, true_outlier_idxs, len(data))
    summary[name] = {
        "detected": len(detected),
        "true": len(true_outlier_idxs),
        "correct": len(correct),
        "accuracy": f"{len(correct)} / {len(true_outlier_idxs)}",
        "f1_score": round(f1, 3)
    }

# Run DILOF
for i, params in enumerate(dilof_params, 1):
    name = f"DILOF-{i}"
    model = DILOF(**params)
    print(f"\nRunning {name} with {params}")
    detected = [i for i, point in enumerate(data) if model.insert(point)[0]]
    correct = set(detected).intersection(true_outlier_idxs)
    f1 = f1_score(detected, true_outlier_idxs, len(data))
    summary[name] = {
        "detected": len(detected),
        "true": len(true_outlier_idxs),
        "correct": len(correct),
        "accuracy": f"{len(correct)} / {len(true_outlier_idxs)}",
        "f1_score": round(f1, 3)
    }

# Run IncrementalLOF
for i, params in enumerate(inclof_params, 1):
    name = f"IncrementalLOF-{i}"
    model = IncrementalLOF(**params)
    print(f"\nRunning {name} with {params}")
    detected = [i for i, point in enumerate(data) if model.insert(point)[0]]
    correct = set(detected).intersection(true_outlier_idxs)
    f1 = f1_score(detected, true_outlier_idxs, len(data))
    summary[name] = {
        "detected": len(detected),
        "true": len(true_outlier_idxs),
        "correct": len(correct),
        "accuracy": f"{len(correct)} / {len(true_outlier_idxs)}",
        "f1_score": round(f1, 3)
    }

# Print all results
print("\n========== Summary ==========")
for name, result in summary.items():
    print(f"{name}:")
    print(f"  Total Detected:       {result['detected']}")
    print(f"  True Outliers:        {result['true']}")
    print(f"  Correctly Detected:   {result['correct']}")
    print(f"  Accuracy:             {result['accuracy']}")
    print(f"  F1 Score:             {result['f1_score']}\n")
