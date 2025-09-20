"""
Point clustering functions for the red pillar detection pipeline.

This module handles DBSCAN clustering of red points to group them into
potential pillar candidates based on spatial proximity.
"""

import numpy as np
import time
from typing import Tuple, List
from sklearn.cluster import DBSCAN
from config import DBSCAN_EPS, DBSCAN_MIN_SAMPLES


def cluster_red_points(red_points: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
    """
    Cluster red points using DBSCAN algorithm.

    Args:
        red_points: numpy array of shape (N, 3) with red point coordinates

    Returns:
        Tuple of (clusters, cluster_labels) where:
        - clusters: List of numpy arrays, each containing points for one cluster
        - cluster_labels: List of cluster label IDs
    """
    if len(red_points) == 0:
        return [], []

    print("Clustering red points using DBSCAN...")
    start_time = time.time()

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1)
    cluster_labels = dbscan.fit_predict(red_points)

    # Extract valid clusters (exclude noise labeled as -1)
    unique_labels = np.unique(cluster_labels)
    valid_labels = unique_labels[unique_labels != -1]

    clusters = []
    cluster_ids = []

    for label in valid_labels:
        cluster_mask = cluster_labels == label
        cluster_points = red_points[cluster_mask]

        # Filter clusters by minimum size
        if len(cluster_points) >= DBSCAN_MIN_SAMPLES:
            clusters.append(cluster_points)
            cluster_ids.append(label)

    noise_points = np.sum(cluster_labels == -1)
    cluster_time = time.time() - start_time

    print(f"Found {len(clusters)} valid clusters and {noise_points:,} noise points "
          f"in {cluster_time:.2f} seconds")

    for i, cluster in enumerate(clusters):
        print(f"  Cluster {cluster_ids[i]}: {len(cluster):,} points")

    return clusters, cluster_ids
