"""
Visualization functions for the red pillar detection pipeline.

This module handles the creation of visualization output including cylinder
sample point generation, color-coded point cloud creation, and output
visualization preparation.
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Any
from config import GRAY_COLOR, RED_COLOR, CYLINDER_SAMPLE_DENSITY


def generate_pca_axes_points(pillar: Dict[str, Any], axis_length_factor: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 3D coordinate axes based on PCA components for a detected pillar.

    Args:
        pillar: Detected pillar dictionary containing PCA analysis results
        axis_length_factor: Factor to scale axis length relative to pillar radius

    Returns:
        Tuple of (axis_points, axis_colors) for PCA-based coordinate system
    """
    center = pillar['center']
    axis = pillar['axis']  # Primary PCA component (cylinder axis)
    radius = pillar['radius']

    # Calculate axis length based on pillar dimensions
    if len(pillar['inlier_points']) > 0:
        # Estimate height from inlier points
        axis_norm = axis / np.linalg.norm(axis)
        projections = np.dot(pillar['inlier_points'] - center, axis_norm)
        height = np.max(projections) - np.min(projections)
        axis_length = max(height * 0.5, radius * axis_length_factor)
    else:
        axis_length = radius * axis_length_factor

    # Get all PCA components if available (from eigenanalysis in pca_analysis.py)
    # For robust axis generation, we'll create orthogonal axes based on the primary axis
    primary_axis = axis / np.linalg.norm(axis)

    # Create two orthogonal vectors for cross-sectional axes
    # Find a vector orthogonal to primary axis
    if abs(primary_axis[2]) < 0.9:
        secondary_axis = np.cross(primary_axis, [0, 0, 1])
    else:
        secondary_axis = np.cross(primary_axis, [1, 0, 0])
    secondary_axis = secondary_axis / np.linalg.norm(secondary_axis)

    # Third axis is cross product of first two
    tertiary_axis = np.cross(primary_axis, secondary_axis)
    tertiary_axis = tertiary_axis / np.linalg.norm(tertiary_axis)

    # Generate axis line points
    axis_points = []
    axis_colors = []

    # Primary axis (Red) - cylinder main direction
    primary_start = center - axis_length * primary_axis
    primary_end = center + axis_length * primary_axis
    primary_points = np.linspace(primary_start, primary_end, 20)
    axis_points.extend(primary_points)
    axis_colors.extend([[255, 0, 0]] * len(primary_points))  # Red

    # Secondary axis (Green) - first cross-sectional direction
    secondary_start = center - axis_length * 0.7 * secondary_axis
    secondary_end = center + axis_length * 0.7 * secondary_axis
    secondary_points = np.linspace(secondary_start, secondary_end, 15)
    axis_points.extend(secondary_points)
    axis_colors.extend([[0, 255, 0]] * len(secondary_points))  # Green

    # Tertiary axis (Blue) - second cross-sectional direction
    tertiary_start = center - axis_length * 0.7 * tertiary_axis
    tertiary_end = center + axis_length * 0.7 * tertiary_axis
    tertiary_points = np.linspace(tertiary_start, tertiary_end, 15)
    axis_points.extend(tertiary_points)
    axis_colors.extend([[0, 0, 255]] * len(tertiary_points))  # Blue

    return np.array(axis_points), np.array(axis_colors, dtype=np.uint8)


def generate_cylinder_sample_points(center: np.ndarray, axis: np.ndarray, radius: float,
                                    height: float, num_points: int = 100) -> np.ndarray:
    """
    Generate sample points for cylinder visualization.

    Args:
        center: Cylinder center point
        axis: Cylinder axis vector
        radius: Cylinder radius
        height: Cylinder height
        num_points: Number of sample points to generate

    Returns:
        numpy array of sample points for visualization
    """
    # Normalize axis
    axis_norm = axis / np.linalg.norm(axis)

    # Create orthogonal vectors for cylinder circumference
    # Find a vector orthogonal to axis
    if abs(axis_norm[2]) < 0.9:
        ortho1 = np.cross(axis_norm, [0, 0, 1])
    else:
        ortho1 = np.cross(axis_norm, [1, 0, 0])
    ortho1 = ortho1 / np.linalg.norm(ortho1)
    ortho2 = np.cross(axis_norm, ortho1)

    sample_points = []

    # Generate circular cross-sections at different heights
    num_circles = max(5, int(height * CYLINDER_SAMPLE_DENSITY))
    num_points_per_circle = max(8, num_points // num_circles)

    for h in np.linspace(-height/2, height/2, num_circles):
        circle_center = center + h * axis_norm

        for theta in np.linspace(0, 2*np.pi, num_points_per_circle, endpoint=False):
            point = (circle_center +
                     radius * (np.cos(theta) * ortho1 + np.sin(theta) * ortho2))
            sample_points.append(point)

    return np.array(sample_points)


def create_visualization_output(points: np.ndarray, colors: np.ndarray,
                                detected_pillars: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create visualization with color-coded results including PCA-based coordinate axes.

    Args:
        points: Original point cloud coordinates
        colors: Original point cloud colors
        detected_pillars: List of detected pillar dictionaries

    Returns:
        Tuple of (output_points, output_colors) for visualization
    """
    print("Creating visualization output with PCA axes...")
    start_time = time.time()

    # Start with all points in gray
    output_points = points.copy()
    output_colors = np.full_like(colors, GRAY_COLOR)

    # Collect all pillar inlier points
    all_pillar_points = []
    all_axes_points = []
    all_axes_colors = []

    for pillar in detected_pillars:
        if len(pillar['inlier_points']) > 0:
            all_pillar_points.append(pillar['inlier_points'])

            # Generate PCA-based coordinate axes for each pillar
            axes_points, axes_colors = generate_pca_axes_points(pillar)
            all_axes_points.append(axes_points)
            all_axes_colors.append(axes_colors)

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

            print(f"Added PCA axes for {len(detected_pillars)} pillars ({len(combined_axes_points):,} axis points)")

        # Optional: Add cylinder sample geometry for reference
        for i, pillar in enumerate(detected_pillars):
            if len(pillar['inlier_points']) > 0:
                # Estimate height from inlier points
                axis_norm = pillar['axis'] / np.linalg.norm(pillar['axis'])
                projections = np.dot(
                    pillar['inlier_points'] - pillar['center'], axis_norm)
                height = np.max(projections) - np.min(projections)

                # Generate cylinder sample points
                cylinder_samples = generate_cylinder_sample_points(
                    pillar['center'], pillar['axis'], pillar['radius'], height, 100
                )

                # Add cylinder samples to output (lighter red to distinguish from axes)
                output_points = np.vstack([output_points, cylinder_samples])
                cylinder_colors = np.full(
                    (len(cylinder_samples), 3), (180, 0, 0), dtype=np.uint8)  # Darker red
                output_colors = np.vstack([output_colors, cylinder_colors])

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
