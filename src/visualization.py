"""
Visualization functions for the multi-color pillar detection pipeline.

This module handles the creation of visualization output including cylinder
sample point generation, color-coded point cloud creation, and output
visualization preparation.
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Any
from config import GRAY_COLOR, RED_COLOR


def generate_pca_axes_points(pillar: Dict[str, Any], axis_length_factor: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate cylinder axis line for a detected pillar (main axis only).

    Args:
        pillar: Detected pillar dictionary containing analysis results
        axis_length_factor: Factor to scale axis length relative to pillar height

    Returns:
        Tuple of (axis_points, axis_colors) for cylinder main axis only
    """
    center = pillar['center']
    axis = pillar['axis']  # Cylinder axis

    # Calculate axis length based on pillar dimensions
    if len(pillar['inlier_points']) > 0:
        # Estimate height from inlier points
        axis_norm = axis / np.linalg.norm(axis)
        projections = np.dot(pillar['inlier_points'] - center, axis_norm)
        height = np.max(projections) - np.min(projections)
        axis_length = height * axis_length_factor
    else:
        axis_length = pillar['radius'] * 10  # Fallback

    # Generate only the main axis (cylinder direction)
    primary_axis = axis / np.linalg.norm(axis)
    primary_start = center - axis_length * 0.5 * primary_axis
    primary_end = center + axis_length * 0.5 * primary_axis

    # Create axis line with more points for visibility
    axis_points = np.linspace(primary_start, primary_end, 30)
    axis_colors = np.full((len(axis_points), 3), [
                          255, 255, 0], dtype=np.uint8)  # Yellow for visibility

    return axis_points, axis_colors


def create_visualization_output(points: np.ndarray, colors: np.ndarray,
                                detected_pillars: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create visualization with color-coded results including cylinder axes.

    Args:
        points: Original point cloud coordinates
        colors: Original point cloud colors
        detected_pillars: List of detected pillar dictionaries

    Returns:
        Tuple of (output_points, output_colors) for visualization
    """
    print("Creating visualization output with cylinder axes...")
    start_time = time.time()

    # Start with all points in gray
    output_points = points.copy()
    output_colors = np.full_like(colors, GRAY_COLOR)

    # Collect all pillar inlier points
    all_pillar_points = []
    all_axes_points = []
    all_axes_colors = []

    # Find the pillar with the most points
    largest_pillar = None
    max_points = 0
    for pillar in detected_pillars:
        if len(pillar['inlier_points']) > max_points:
            max_points = len(pillar['inlier_points'])
            largest_pillar = pillar

    for pillar in detected_pillars:
        if len(pillar['inlier_points']) > 0:
            all_pillar_points.append(pillar['inlier_points'])

            # Generate axis only for the largest pillar
            if pillar is largest_pillar:
                axes_points, axes_colors = generate_pca_axes_points(pillar)
                all_axes_points.append(axes_points)
                all_axes_colors.append(axes_colors)
                print(
                    f"Generating axis for largest cluster: {len(pillar['inlier_points']):,} points")

    if all_pillar_points:
        # Combine all pillar points
        pillar_points = np.vstack(all_pillar_points)

        # Color pillar points in red
        output_points = np.vstack([output_points, pillar_points])
        pillar_colors = np.full((len(pillar_points), 3),
                                RED_COLOR, dtype=np.uint8)
        output_colors = np.vstack([output_colors, pillar_colors])

        # Add PCA-based coordinate axes
        if all_axes_points:
            combined_axes_points = np.vstack(all_axes_points)
            combined_axes_colors = np.vstack(all_axes_colors)

            output_points = np.vstack([output_points, combined_axes_points])
            output_colors = np.vstack([output_colors, combined_axes_colors])

            print(
                f"Added cylinder axes for {len(detected_pillars)} pillars ({len(combined_axes_points):,} axis points)")

        # Note: Cylinder surface visualization removed - showing axes only

    viz_time = time.time() - start_time
    print(
        f"Created visualization with {len(output_points):,} points in {viz_time:.2f} seconds")

    return output_points, output_colors


def create_clustering_visualization(
    original_points: np.ndarray,
    original_colors: np.ndarray,
    clusters: List[np.ndarray],
    cluster_ids: List[int],
    cluster_indices: List[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a visualization of clustering results with color-coded clusters.

    Args:
        original_points: Original point cloud array of shape (N, 3)
        original_colors: Original color array of shape (N, 3) with values 0-255
        clusters: List of cluster point arrays
        cluster_ids: List of cluster IDs
        cluster_indices: List of arrays containing original indices for each cluster (optional)

    Returns:
        Tuple of (visualization_points, visualization_colors) for saving
    """
    print("Creating clustering visualization...")

    # Start with gray background for all original points
    viz_points = original_points.copy()
    viz_colors = np.full_like(original_colors, 128,
                              dtype=np.uint8)  # Gray background

    # Generate distinctive colors for each cluster
    cluster_colors = [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
        [255, 128, 0],  # Orange
        [128, 0, 255],  # Purple
        [255, 128, 128],  # Light Red
        [128, 255, 128],  # Light Green
    ]

    # Efficiently color cluster points using indices
    total_cluster_points = 0

    for i, (cluster_points, cluster_id) in enumerate(zip(clusters, cluster_ids)):
        if len(cluster_points) == 0:
            continue

        # Use cycling colors if we have more clusters than predefined colors
        color_idx = i % len(cluster_colors)
        cluster_color = np.array(cluster_colors[color_idx], dtype=np.uint8)

        # Use direct indexing if cluster_indices is provided (FAST path)
        if cluster_indices is not None and i < len(cluster_indices):
            # Direct index-based coloring - O(M) instead of O(N×M)
            indices = cluster_indices[i]
            viz_colors[indices] = cluster_color
        else:
            # Fallback to distance-based matching (SLOW path - backward compatibility)
            print(
                f"  Warning: Using slow distance-based matching for cluster {cluster_id}")
            tolerance = 1e-6
            for cluster_point in cluster_points:
                # Find closest point in original cloud
                distances = np.linalg.norm(
                    original_points - cluster_point, axis=1)
                closest_idx = np.argmin(distances)
                # If the distance is very small, it's a match
                if distances[closest_idx] < tolerance:
                    viz_colors[closest_idx] = cluster_color

        total_cluster_points += len(cluster_points)

    print(
        f"  Colored {total_cluster_points:,} cluster points across {len(clusters)} clusters")

    return viz_points, viz_colors
