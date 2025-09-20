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
    Create visualization with color-coded results.

    Args:
        points: Original point cloud coordinates
        colors: Original point cloud colors
        detected_pillars: List of detected pillar dictionaries

    Returns:
        Tuple of (output_points, output_colors) for visualization
    """
    print("Creating visualization output...")
    start_time = time.time()

    # Start with all points in gray
    output_points = points.copy()
    output_colors = np.full_like(colors, GRAY_COLOR)

    # Collect all pillar inlier points
    all_pillar_points = []

    for pillar in detected_pillars:
        if len(pillar['inlier_points']) > 0:
            all_pillar_points.append(pillar['inlier_points'])

    if all_pillar_points:
        # Combine all pillar points
        pillar_points = np.vstack(all_pillar_points)

        # Color pillar points in red
        output_points = np.vstack([output_points, pillar_points])
        pillar_colors = np.full((len(pillar_points), 3),
                                RED_COLOR, dtype=np.uint8)
        output_colors = np.vstack([output_colors, pillar_colors])

        # Optional: Add cylinder sample geometry
        for i, pillar in enumerate(detected_pillars):
            if len(pillar['inlier_points']) > 0:
                # Estimate height from inlier points
                axis_norm = pillar['axis'] / np.linalg.norm(pillar['axis'])
                projections = np.dot(
                    pillar['inlier_points'] - pillar['center'], axis_norm)
                height = np.max(projections) - np.min(projections)

                # Generate cylinder sample points
                cylinder_samples = generate_cylinder_sample_points(
                    pillar['center'], pillar['axis'], pillar['radius'], height, 200
                )

                # Add cylinder samples to output
                output_points = np.vstack([output_points, cylinder_samples])
                cylinder_colors = np.full(
                    (len(cylinder_samples), 3), RED_COLOR, dtype=np.uint8)
                output_colors = np.vstack([output_colors, cylinder_colors])

    viz_time = time.time() - start_time
    print(
        f"Created visualization with {len(output_points):,} points in {viz_time:.2f} seconds")

    return output_points, output_colors
