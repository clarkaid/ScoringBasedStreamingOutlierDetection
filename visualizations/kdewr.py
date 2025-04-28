import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.append(os.path.abspath('..'))

from kdwer import KDEWRStreaming

# Fake 2D data (some clusters, some outliers)
np.random.seed(0)
cluster = np.random.normal(0.5, 0.05, (100, 2))
outliers = np.random.uniform(0, 1, (10, 2))
data = np.vstack([cluster, outliers])

detector = KDEWRStreaming(windowSize=30, dim=2, r=1, xi=0.5, forgetting=0.9)

outlier_points = []
frames = []

fig, ax = plt.subplots(figsize=(6, 6))

def update(i):
    ax.clear()
    point = data[i]
    is_outlier, _ = detector.insert(point)

    # Plot existing points
    x_vals, y_vals = zip(*list(detector.window))
    ax.scatter(x_vals, y_vals, color="black", s=20, label="Window Points")

    # Plot bins (mean of compressed points)
    bin_means = [v[1] for v in detector.bins.values()]
    if bin_means:
        x_bin, y_bin = zip(*bin_means)
        ax.scatter(x_bin, y_bin, color="red", marker='x', s=50, label="Bin Summaries")

    # Highlight the current point
    ax.scatter(point[0], point[1], color="blue", edgecolors='white', s=100, label="New Point")
    if is_outlier:
        outlier_points.append(point)
    
    if outlier_points:
        outlier_x, outlier_y = zip(*outlier_points)
        ax.scatter(outlier_x, outlier_y, color="purple", marker='*', s=150, label="Detected Outliers")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"KDEWR Streaming (Frame {i})")
    ax.legend(loc="upper left")
    ax.grid(True)

ani = animation.FuncAnimation(fig, update, frames=len(data), interval=300)

ani.save('kdewr_streaming.gif', writer='pillow')
