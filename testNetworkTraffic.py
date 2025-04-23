import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from dilof import DILOF
from milof import milof
from IncrementalLOF import IncrementalLOF
from SlidingWindow import KDEWRStreaming

# Load dataset
df = pd.read_csv("./datasets/NF-UQ-NIDS-V2.csv")

# Drop non-numeric and metadata columns
drop_cols = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label', 'Attack', 'Dataset']
df = df.drop(columns=drop_cols, errors='ignore')

# Keep only numeric columns
df = df.select_dtypes(include=[np.number])

# Fill missing values (if any)
df = df.fillna(0)

# Normalize features to [0,1]
scaler = MinMaxScaler()
data = scaler.fit_transform(df)

# ======================
# Test DILOF
# ======================
dilof_detector = DILOF(k=10, window_size=200, threshold=1.5, step_size=0.01, regularizer=0.1, max_iter=10)
dilof_detected = []
for point in data:
    is_outlier, score = dilof_detector.insert(point)
    if is_outlier:
        dilof_detected.append(point)

print(f"Total Detected Outliers by DILOF: {len(dilof_detected)}")

# ======================
# Test MiLOF
# ======================
milof_detector = milof(k=10, b=100, c=10, threshold=1.5)
milof_detected = []
for point in data:
    result = milof_detector.insert(point)
    if len(milof_detector.outliers) > len(milof_detected):
        milof_detected.append(point)

print(f"Total Detected Outliers by MiLOF: {len(milof_detected)}")

# ======================
# Test Incremental LOF
# ======================
inclof_detector = IncrementalLOF(k=10, threshold=1.5)
inclof_detected = []
for point in data:
    is_outlier, score = inclof_detector.insert(point)
    if is_outlier:
        inclof_detected.append(point)

print(f"Total Detected Outliers by Incremental LOF: {len(inclof_detected)}")

# ======================
# Test KDEWRStreaming
# ======================
kdewr_detector = KDEWRStreaming(windowSize=200, dim=data.shape[1], r=2, xi=0.1, forgetting=0.9)
kdewr_detected = []
for point in data:
    is_outlier, score = kdewr_detector.insert(point)
    if is_outlier:
        kdewr_detected.append(point)

print(f"Total Detected Outliers by KDEWRStreaming: {len(kdewr_detected)}")
