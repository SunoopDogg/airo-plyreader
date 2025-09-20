"""
Cylinder fitting and pillar detection functions for the red pillar detection pipeline.

This module handles RANSAC-based cylinder fitting to point clusters, geometric
validation of cylinder parameters, and pillar detection with confidence scoring.
"""

import numpy as np
import time
from typing import List, Optional, Dict, Any
import pyransac3d as pyrsc
from config import (
    PILLAR_RADIUS_MIN, PILLAR_RADIUS_MAX, PILLAR_HEIGHT_MIN,
    PILLAR_AXIS_Z_ANGLE_MAX, RANSAC_THRESHOLD, RANSAC_MAX_ITERATIONS,
    MAX_POINTS_PER_CLUSTER
)


def validate_cylinder_geometry(center: np.ndarray, axis: np.ndarray, radius: float,
                               cluster_points: np.ndarray) -> bool:
    """
    Validate cylinder geometry against pillar constraints.

    Args:
        center: Cylinder center point [x, y, z]
        axis: Cylinder axis vector [ax, ay, az]
        radius: Cylinder radius
        cluster_points: Points used for fitting

    Returns:
        True if cylinder meets pillar criteria, False otherwise
    """
    # Check radius constraints
    if radius < PILLAR_RADIUS_MIN or radius > PILLAR_RADIUS_MAX:
        return False

    # Check height constraint (approximate from point spread along axis)
    # Project points onto cylinder axis to estimate height
    axis_normalized = axis / np.linalg.norm(axis)
    projections = np.dot(cluster_points - center, axis_normalized)
    height = np.max(projections) - np.min(projections)

    if height < PILLAR_HEIGHT_MIN:
        return False

    # Check axis alignment with Z-axis (vertical orientation)
    z_axis = np.array([0, 0, 1])
    axis_normalized = axis / np.linalg.norm(axis)
    angle_with_z = np.arccos(np.abs(np.dot(axis_normalized, z_axis)))
    angle_degrees = np.degrees(angle_with_z)

    if angle_degrees > PILLAR_AXIS_Z_ANGLE_MAX:
        return False

    return True


def fit_cylinder_to_cluster(cluster_points: np.ndarray, cluster_id: int) -> Optional[Dict[str, Any]]:
    """
    Fit cylinder to point cluster using RANSAC.

    Args:
        cluster_points: numpy array of shape (N, 3) with cluster point coordinates
        cluster_id: Cluster identifier for logging

    Returns:
        Dictionary with cylinder parameters if successful, None otherwise
    """
    if len(cluster_points) < 10:  # Minimum points for cylinder fitting
        return None

    print(
        f"  Fitting cylinder to cluster {cluster_id} ({len(cluster_points):,} points)...")

    # Limit cluster size to prevent memory issues
    if len(cluster_points) > MAX_POINTS_PER_CLUSTER:
        # Randomly sample points
        indices = np.random.choice(
            len(cluster_points), MAX_POINTS_PER_CLUSTER, replace=False)
        cluster_points = cluster_points[indices]
        print(f"    Downsampled to {len(cluster_points):,} points")

    try:
        start_time = time.time()

        # Create cylinder detector
        cylinder = pyrsc.Cylinder()

        # Fit cylinder using RANSAC
        center, axis, radius, inliers = cylinder.fit(
            cluster_points,
            thresh=RANSAC_THRESHOLD,
            maxIteration=RANSAC_MAX_ITERATIONS
        )

        fit_time = time.time() - start_time

        if center is None or axis is None or radius is None:
            print(f"    Cylinder fitting failed (no solution found)")
            return None

        # Flatten arrays if needed
        if center.ndim > 1:
            center = center.flatten()
        if axis.ndim > 1:
            axis = axis.flatten()

        # Validate geometry
        if not validate_cylinder_geometry(center, axis, radius, cluster_points):
            print(f"    Cylinder rejected (fails geometric constraints)")
            return None

        # Calculate confidence metrics
        confidence = len(inliers) / len(cluster_points)
        inlier_points = cluster_points[inliers] if len(
            inliers) > 0 else np.array([])

        print(f"    Cylinder fitted: radius={radius:.3f}m, confidence={confidence:.3f}, "
              f"time={fit_time:.2f}s")

        return {
            'center': center,
            'axis': axis,
            'radius': radius,
            'inliers': inliers,
            'inlier_points': inlier_points,
            'confidence': confidence,
            'cluster_id': cluster_id
        }

    except Exception as e:
        print(f"    Cylinder fitting error: {str(e)}")
        return None


def detect_pillars_in_clusters(clusters: List[np.ndarray], cluster_ids: List[int]) -> List[Dict[str, Any]]:
    """
    Detect pillars by fitting cylinders to each cluster.

    Args:
        clusters: List of point clusters
        cluster_ids: List of cluster identifiers

    Returns:
        List of detected pillar dictionaries
    """
    print("Detecting pillars using cylinder fitting...")
    start_time = time.time()

    detected_pillars = []

    for cluster, cluster_id in zip(clusters, cluster_ids):
        pillar = fit_cylinder_to_cluster(cluster, cluster_id)
        if pillar is not None:
            detected_pillars.append(pillar)

    detection_time = time.time() - start_time
    print(
        f"Detected {len(detected_pillars)} pillars in {detection_time:.2f} seconds")

    # Debug: Show detection statistics
    if len(detected_pillars) > 0:
        confidences = [p['confidence'] for p in detected_pillars]
        radii = [p['radius'] for p in detected_pillars]
        print(
            f"  Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
        print(f"  Radius range: {min(radii):.3f}m - {max(radii):.3f}m")
    else:
        print("  No pillars met the geometric constraints")

    # Sort pillars by confidence
    detected_pillars.sort(key=lambda p: p['confidence'], reverse=True)

    return detected_pillars