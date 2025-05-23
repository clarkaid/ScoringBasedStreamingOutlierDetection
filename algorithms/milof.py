"""
File: milof.py
Author: Rithvik Nakirikanti
Updated: 4/15/2025
Description: An implementation of memory efficient incremental local outlier (MiLOF) detection algorithm for data streams,
as described in this paper:
https://www.researchgate.net/profile/James-Bezdek/publication/305825504_Fast_Memory_Efficient_Local_Outlier_Detection_in_Data_Streams/links/5a1ec39baca272cbfbc06f3d/Fast-Memory-Efficient-Local-Outlier-Detection-in-Data-Streams.pdf
"""
from sklearn.cluster import KMeans
import numpy as np
from item import Item
from IncrementalLOF import IncrementalLOF
from collections import deque

class ClusterSummary:
    def __init__(self, center, k_distance, lrd, lof, weight):
        self.center = center 
        self.k_distance = k_distance
        self.lrd = lrd
        self.lof = lof
        self.weight = weight

class milof(IncrementalLOF):
    def __init__(self, k, b=100, c=10, threshold=1.0, print_outliers=False, skipping_enabled=False):
        super().__init__(k, threshold, print_outliers)
        self.b = b  #number of recent data points
        self.c = c  #max number of cluster summaries
        self.recent_data = deque()  #deque of Item objects
        self.cluster_summaries = []  #list of ClusterSummary objects
        self.skipping_enabled = skipping_enabled
        self.try_skip_next = False

    def insert(self, point):
        item = Item(point)
        res = self.lod(item)
        if res >= 0:
            self.recent_data.append(item)
            if len(self.recent_data) >= self.b:
                self._summarize_oldest_data()

        if res < 0 or res > self.threshold:
            #Outlier
            return (True, res)
        else:
            return (False, res)

    def lod(self, item):
        """
        Computes the LOF of the given item using summaries + recent data.
        Returns the LOF of the point or -1 if skipped (for future skipping logic).
        """
        if self.skipping_enabled and self.skipping_scheme(item):
            return -1

        item.neighbors = self._find_k_neighbors(item)
        for neighbor in item.neighbors:
            if isinstance(neighbor, Item):
                if neighbor.k_distance is None:
                    neighbor.k_distance = 0.0

        x_update = item.reverse_knn.copy()
        x_update_lrd = x_update[:]

        for j in x_update:
            for i in j.neighbors:
                if i != item and j in i.neighbors and j not in x_update_lrd:
                    x_update_lrd.append(i)

        x_update_lof = x_update_lrd[:]

        for m in x_update_lrd:
            m.set_lrd(self.k)
            for y in m.neighbors:
                if y not in x_update_lof:
                    x_update_lof.append(y)

        for l in x_update_lof:
            l.set_lof(self.k)

        item.set_lrd(self.k)
        item.set_lof(self.k)

        if item.lof > self.threshold:
            self.outliers.append(item)
            if self.print_outliers:
                print("Outlier Detected:", item.tuple, "with a LOF of", item.lof)
            self.try_skip_next = True
        else:
            self.try_skip_next = False

        return item.lof

    def skipping_scheme(self, item):
        """
        Skips insertion of items that are part of a detected outlier sequence.
        """
        if not self.try_skip_next:
            return False

        if not self.outliers:
            return False

        sum_d1 = 0
        count = 0
        for x in self.recent_data:
            if x.neighbors:
                sum_d1 += x.dist(x.neighbors[0])
                count += 1

        if count == 0:
            return False

        d1_bar = sum_d1 / count

        if item.dist(self.outliers[-1]) < d1_bar:
            self.outliers.append(item)
            if self.print_outliers:
                print("Outlier Detected:", item.tuple, "as part of a sequence")
            return True

        return False

    def _find_k_neighbors(self, p):
        dists = []
        for item in self.recent_data:
            dist = p.dist(item)
            dists.append((dist, item))

        for summary in self.cluster_summaries:
            dist = np.linalg.norm(np.array(p.tuple) - summary.center)
            dists.append((dist, summary))

        dists.sort(key=lambda x: x[0])

        neighbors = []
        for _, neighbor in dists[:self.k]:
            if isinstance(neighbor, ClusterSummary):
                fake_item = Item(neighbor.center)
                fake_item.k_distance = neighbor.k_distance if neighbor.k_distance is not None else 0.0
                fake_item.lrd = neighbor.lrd if neighbor.lrd is not None else 0.0
                fake_item.lof = neighbor.lof if neighbor.lof is not None else 1.0
                neighbors.append(fake_item)
            else:
                neighbors.append(neighbor)

        return neighbors

    def _summarize_oldest_data(self):
        old_items = [self.recent_data.popleft() for _ in range(self.b // 2)]
        points = np.array([item.tuple for item in old_items])

        kmeans = KMeans(n_clusters=self.c, n_init=10).fit(points)
        labels = kmeans.labels_

        new_summaries = []
        for i in range(self.c):
            members = [old_items[j] for j in range(len(points)) if labels[j] == i]
            if not members:
                continue

            center = kmeans.cluster_centers_[i]

            # Only include non-None values
            kdist_vals = [m.k_distance for m in members if m.k_distance is not None]
            lrd_vals = [m.lrd for m in members if m.lrd is not None]
            lof_vals = [m.lof for m in members if m.lof is not None]

            avg_kdist = np.mean(kdist_vals) if kdist_vals else 0.0
            avg_lrd = np.mean(lrd_vals) if lrd_vals else 0.0
            avg_lof = np.mean(lof_vals) if lof_vals else 1.0

            new_summaries.append(ClusterSummary(center, avg_kdist, avg_lrd, avg_lof, len(members)))

        self._merge_summaries(new_summaries)

    def _merge_summaries(self, new_summaries):
        self.cluster_summaries.extend(new_summaries)
        if len(self.cluster_summaries) > self.c:
            self.cluster_summaries.sort(key=lambda x: x.weight, reverse=True)
            self.cluster_summaries = self.cluster_summaries[:self.c]
