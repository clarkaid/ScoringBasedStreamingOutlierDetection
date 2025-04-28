import matplotlib.pyplot as plt

# Example summarized results manually entered
results = {
    'KDEWR-1': 0.171,
    'KDEWR-2': 0.039,
    'KDEWR-3': 0.0,
    'MiLOF-1': 0.328,
    'MiLOF-2': 0.456,
    'MiLOF-3': 0.491,
    'DILOF-1': 0.129,
    'DILOF-2': 0.221,
    'DILOF-3': 0.368,
    'IncrementalLOF-1': 0.125,
    'IncrementalLOF-2': 0.194,
    'IncrementalLOF-3': 0.301
}

sorted_results = dict(sorted(results.items(), key=lambda item: item[0]))

# Create the line plot
plt.figure(figsize=(8, 6))
plt.plot(list(sorted_results.keys()), list(sorted_results.values()), marker='o', color='red', linewidth=2)
plt.fill_between(range(len(sorted_results)), list(sorted_results.values()), color='black', alpha=0.1)

plt.xticks(rotation=45, ha='right')
plt.ylabel('F1 Score')
plt.ylim(0, 0.6)
plt.title('F1 Score Comparison Across Streaming Outlier Algorithms')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('network_traffic_eval.png', dpi=300)
plt.show()
