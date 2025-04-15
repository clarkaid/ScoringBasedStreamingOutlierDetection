import numpy as np
from collections import deque, defaultdict
from math import exp, pi
import matplotlib.pyplot as plt

# Streaming outlier detection using KDE with a sliding window and binned summary
class KDEWRStreaming:
    def __init__(self, windowSize=100, binSplit=10, dim=2, forgetting=0.5, r=3, xi=0.1):
        # Initialize parameters and internal data structures
        self.W = windowSize
        self.k = binSplit
        self.dim = dim
        self.lam = forgetting  
        self.r = r
        self.xi = xi
        self.window = deque()
        self.bins = {}
        self.binTime = {}
        self.candidates = defaultdict(int)
        self.totalCount = 0

    # Gaussian kernel used for density estimation
    def _gaussian_kernel(self, x, y, h):
        diff = (x - y) / h
        exponent = -0.5 * np.dot(diff, diff)
        denom = (2 * pi) ** (self.dim / 2) * np.prod(h)
        return exp(exponent) / denom

    # Scott's rule to determine bandwidth per dimension
    def _scott_bandwidth(self, data):
        std_dev = np.std(data, axis=0)
        n = len(data)
        h = std_dev * n ** (-1 / (self.dim + 4))
        return np.where(h == 0, 1e-6, h) 

    # Estimate density from the sliding window
    def _window_density(self, x, h):
        if len(self.window) == 0:
            return 0.0
        density = 0.0
        for p in self.window:
            density += self._gaussian_kernel(x, p, h)
        return density / len(self.window)

    # Estimate density from binned summary of older data
    def _bin_density(self, x, h):
        density = 0.0
        weighted_sum = 0.0
        for idx, (count, mean) in self.bins.items():
            age = self.totalCount - self.binTime.get(idx, self.totalCount)
            weight = (self.lam ** age) * count
            density += weight * self._gaussian_kernel(x, mean, h)
            weighted_sum += weight
        return density / weighted_sum if weighted_sum > 0 else 0.0

    # Assign a bin index to a point based on its coordinates
    def _get_bin_index(self, x):
        index = 0
        bounds = (0, 1)
        delta = (bounds[1] - bounds[0]) / self.k
        idxs = [int((xi - bounds[0]) / delta) for xi in x]
        for i, idx in enumerate(idxs):
            index += idx * (self.k ** i)
        return index

    # Update binned summary with points removed from the window
    def _update_bins(self, data):
        for point in data:
            idx = self._get_bin_index(point)
            if idx not in self.bins:
                self.bins[idx] = [1, np.array(point)]
            else:
                count, mean = self.bins[idx]
                new_mean = (mean * count + point) / (count + 1)
                self.bins[idx] = [count + 1, new_mean]
            self.binTime[idx] = self.totalCount

    # Insert a new point into the stream and return outlier status and score
    def insert(self, x):
        self.totalCount += 1
        x = np.array(x)
        self.window.append(x)
        if len(self.window) > self.W:
            removed = [self.window.popleft() for _ in range(self.W // 2)]
            self._update_bins(removed)

        h = self._scott_bandwidth(np.array(self.window))
        f_window = self._window_density(x, h)
        f_bin = self._bin_density(x, h)
        f_cumulative = f_window + f_bin

        if f_cumulative == 0:
            return False, 0.0

        outlier_score = 1 / f_cumulative
        avg_density = np.mean([self._window_density(p, h) for p in self.window])
        threshold = 1 / (avg_density * self.xi)

        if outlier_score > threshold:
            self.candidates[tuple(x)] += 1
        else:
            self.candidates[tuple(x)] = max(0, self.candidates[tuple(x)] - 1)

        is_outlier = self.candidates[tuple(x)] >= self.r
        return is_outlier, outlier_score

### TESTING
np.random.seed(543)

# Generate clustered inliers and separated outliers
inliers = np.random.normal(loc=0.5, scale=0.05, size=(300, 2))
outliers = np.random.uniform(low=0.8, high=1.0, size=(10, 2))
stream = np.vstack((inliers, outliers))
np.random.shuffle(stream)

# Initialize the detector with tuned parameters
detector = KDEWRStreaming(windowSize=100, dim=2, r=1, xi=0.1, forgetting=0.9)
detected = []
scores = []

# Stream data point by point and collect outliers
for point in stream:
    is_outlier, score = detector.insert(point)
    scores.append(score)
    if is_outlier:
        detected.append(point)

# Plot results and save to file
x, y = stream[:, 0], stream[:, 1]
plt.figure(figsize=(8, 6))
plt.scatter(x, y, s=10, label="All Points")
if detected:
    dx, dy = zip(*detected)
    plt.scatter(dx, dy, c='red', s=40, label="Detected Outliers")
# plt.title("KDEWRStreaming: Outlier Detection")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.grid(True)
# plt.savefig("outlier_plot.png")
# plt.show()

print(f"Total Detected Outliers: {len(detected)}")
