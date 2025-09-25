"""
PCA-based cluster analysis and pillar detection for the red pillar detection pipeline.

This module handles Principal Component Analysis (PCA) based cylindrical structure detection
in point clusters, geometric validation of PCA-derived parameters, and pillar detection
with confidence scoring based on eigenvalue ratios.
"""

import numpy as np
import time
from typing import List, Optional, Dict, Any
from sklearn.decomposition import PCA
from config import (
    PILLAR_RADIUS_MIN, PILLAR_RADIUS_MAX, PILLAR_HEIGHT_MIN,
    MAX_POINTS_PER_CLUSTER, PCA_CYLINDER_THRESHOLD,
    PCA_CROSS_SECTION_RATIO_THRESHOLD, PCA_MIN_SECONDARY_VARIANCE,
    PCA_MAX_TERTIARY_VARIANCE
)


def validate_pca_geometry(center: np.ndarray, axis: np.ndarray, radius: float,
                         cluster_points: np.ndarray) -> bool:
    """
    Validate PCA-derived cylinder geometry against pillar constraints.

    Args:
        center: Cylinder center point [x, y, z]
        axis: Cylinder axis vector [ax, ay, az]
        radius: Estimated cylinder radius
        cluster_points: Points used for analysis

    Returns:
        True if cylinder meets pillar criteria, False otherwise
    """
    # Check radius constraints
    if radius < PILLAR_RADIUS_MIN or radius > PILLAR_RADIUS_MAX:
        print(f"    PCA cylinder rejected (radius={radius:.3f}m)")
        return False

    # Check height constraint (approximate from point spread along axis)
    # Project points onto cylinder axis to estimate height
    axis_normalized = axis / np.linalg.norm(axis)
    projections = np.dot(cluster_points - center, axis_normalized)
    height = np.max(projections) - np.min(projections)

    if height < PILLAR_HEIGHT_MIN:
        print(f"    PCA cylinder rejected (height={height:.3f}m)")
        return False

    return True


def analyze_cluster_with_pca(cluster_points: np.ndarray, cluster_id: int) -> Optional[Dict[str, Any]]:
    """
    Analyze point cluster using PCA to detect cylindrical structures.

    Args:
        cluster_points: numpy array of shape (N, 3) with cluster point coordinates
        cluster_id: Cluster identifier for logging

    Returns:
        Dictionary with PCA-derived cylinder parameters if successful, None otherwise
    """
    if len(cluster_points) < 10:  # Minimum points for meaningful PCA
        return None

    print(
        f"  Analyzing cluster {cluster_id} with PCA ({len(cluster_points):,} points)...")

    # Limit cluster size to prevent memory issues
    if len(cluster_points) > MAX_POINTS_PER_CLUSTER:
        # Randomly sample points
        indices = np.random.choice(
            len(cluster_points), MAX_POINTS_PER_CLUSTER, replace=False)
        cluster_points = cluster_points[indices]
        print(f"    Downsampled to {len(cluster_points):,} points")

    try:
        start_time = time.time()

        # Center the data (critical for PCA)
        cluster_center = np.mean(cluster_points, axis=0)
        centered_points = cluster_points - cluster_center

        # Apply PCA
        pca = PCA(n_components=3, svd_solver='auto')
        pca.fit(centered_points)

        # Extract PCA results
        components = pca.components_
        explained_variance_ratios = pca.explained_variance_ratio_
        explained_variance = pca.explained_variance_

        analysis_time = time.time() - start_time

        # Analyze eigenvalue ratios to determine if structure is cylindrical
        λ1, λ2, λ3 = explained_variance_ratios[0], explained_variance_ratios[1], explained_variance_ratios[2]

        # Cylindrical structure criteria:
        # - Strong primary axis (λ1 > threshold)
        # - Limited cross-sectional spread (λ2, λ3 relatively small)
        # - Similar cross-sectional variance (|λ2 - λ3| small)
        is_cylindrical = (
            λ1 > PCA_CYLINDER_THRESHOLD and  # Strong primary direction
            λ2 > PCA_MIN_SECONDARY_VARIANCE and  # Some cross-sectional spread
            λ3 < PCA_MAX_TERTIARY_VARIANCE and   # Third component not too large
            abs(λ2 - λ3) < PCA_CROSS_SECTION_RATIO_THRESHOLD  # Similar cross-sectional variance
        )

        if not is_cylindrical:
            print(f"    Cluster rejected (not cylindrical): λ1={λ1:.3f}, λ2={λ2:.3f}, λ3={λ3:.3f}")
            return None

        # Extract cylinder properties
        cylinder_axis = components[0]  # First principal component (main axis)

        # Estimate radius from cross-sectional variance
        # Use average of second and third eigenvalues as radius estimate
        cross_sectional_variance = (explained_variance[1] + explained_variance[2]) / 2
        estimated_radius = np.sqrt(cross_sectional_variance)

        # Validate geometry
        if not validate_pca_geometry(cluster_center, cylinder_axis, estimated_radius, cluster_points):
            return None

        # Calculate confidence based on eigenvalue ratios and cylindrical structure quality
        # Higher λ1 and more balanced λ2, λ3 indicate better cylindrical structure
        cylindrical_quality = λ1 * (1 - abs(λ2 - λ3))  # Penalize imbalanced cross-sections
        confidence = min(cylindrical_quality, 1.0)  # Cap at 1.0

        print(f"    PCA cylinder detected: radius={estimated_radius:.3f}m, confidence={confidence:.3f}, "
              f"eigenvalues=[{λ1:.3f}, {λ2:.3f}, {λ3:.3f}], time={analysis_time:.2f}s")

        return {
            'center': cluster_center,
            'axis': cylinder_axis,
            'radius': estimated_radius,
            'inliers': np.arange(len(cluster_points)),  # PCA uses all points
            'inlier_points': cluster_points,
            'confidence': confidence,
            'cluster_id': cluster_id,
            'eigenvalue_ratios': explained_variance_ratios,
            'analysis_method': 'PCA'
        }

    except Exception as e:
        print(f"    PCA analysis error: {str(e)}")
        return None


def detect_pillars_with_pca(clusters: List[np.ndarray], cluster_ids: List[int]) -> List[Dict[str, Any]]:
    """
    Detect pillars by analyzing clusters with PCA.

    Args:
        clusters: List of point clusters
        cluster_ids: List of cluster identifiers

    Returns:
        List of detected pillar dictionaries
    """
    print("Detecting pillars using PCA analysis...")
    start_time = time.time()

    detected_pillars = []

    for cluster, cluster_id in zip(clusters, cluster_ids):
        pillar = analyze_cluster_with_pca(cluster, cluster_id)
        if pillar is not None:
            detected_pillars.append(pillar)

    detection_time = time.time() - start_time
    print(
        f"Detected {len(detected_pillars)} pillars in {detection_time:.2f} seconds")

    # Debug: Show detection statistics
    if len(detected_pillars) > 0:
        confidences = [p['confidence'] for p in detected_pillars]
        radii = [p['radius'] for p in detected_pillars]
        eigenvalue_ratios = [p['eigenvalue_ratios'][0] for p in detected_pillars]
        print(
            f"  Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
        print(f"  Radius range: {min(radii):.3f}m - {max(radii):.3f}m")
        print(f"  Primary eigenvalue range: {min(eigenvalue_ratios):.3f} - {max(eigenvalue_ratios):.3f}")
    else:
        print("  No pillars met the PCA cylindrical criteria")

    # Sort pillars by confidence
    detected_pillars.sort(key=lambda p: p['confidence'], reverse=True)

    return detected_pillars


def get_pca_cluster_statistics(cluster_points: np.ndarray) -> Dict[str, float]:
    """
    Get detailed PCA statistics for a cluster (utility function for debugging).

    Args:
        cluster_points: numpy array of shape (N, 3) with cluster point coordinates

    Returns:
        Dictionary with PCA statistics
    """
    if len(cluster_points) < 3:
        return {}

    # Center the data
    centered_points = cluster_points - np.mean(cluster_points, axis=0)

    # Apply PCA
    pca = PCA(n_components=3)
    pca.fit(centered_points)

    λ1, λ2, λ3 = pca.explained_variance_ratio_

    return {
        'eigenvalue_ratio_1': λ1,
        'eigenvalue_ratio_2': λ2,
        'eigenvalue_ratio_3': λ3,
        'cylindrical_score': λ1 * (1 - abs(λ2 - λ3)),
        'linearity_score': λ1,
        'planarity_score': λ1 + λ2,
        'sphericity_score': 1 - λ1
    }