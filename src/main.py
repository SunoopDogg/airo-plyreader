#!/usr/bin/env python3
"""
Red Pillar Detection from PLY Point Cloud Files

This script detects red cylindrical pillars in PLY point cloud files using:
1. Point cloud downsampling for performance optimization
2. HSV color space segmentation for red color detection
3. DBSCAN clustering for grouping red points
4. RANSAC cylinder fitting using pyransac3d
5. Geometric constraints validation for pillar characteristics
6. Visualization output with color-coded results

The pipeline is modularized into separate components for maintainability:
- config: Global configuration parameters
- ply_io: PLY file input/output operations
- downsampling: Point cloud downsampling for performance
- color_segmentation: HSV color filtering
- clustering: DBSCAN clustering
- cylinder_fitting: RANSAC cylinder fitting and pillar detection
- visualization: Output visualization generation
"""

import os
import time
import numpy as np
from typing import List, Dict, Any

# Import modularized components
from config import INPUT_PLY_PATH, OUTPUT_PLY_PATH
from ply_io import load_ply_file_open3d, save_ply_file_open3d
from downsampling import downsample_points
from color_segmentation import filter_red_points_hsv
from clustering import cluster_red_points
from cylinder_fitting import detect_pillars_in_clusters
from visualization import create_visualization_output


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def print_detection_summary(detected_pillars: List[Dict[str, Any]]) -> None:
    """Print summary of detected pillars."""
    print("\n" + "="*60)
    print("PILLAR DETECTION SUMMARY")
    print("="*60)

    if not detected_pillars:
        print("No pillars detected.")
        return

    print(f"Total pillars detected: {len(detected_pillars)}")
    print()

    for i, pillar in enumerate(detected_pillars):
        center = pillar['center']
        axis = pillar['axis']
        radius = pillar['radius']
        confidence = pillar['confidence']

        # Calculate height
        if len(pillar['inlier_points']) > 0:
            axis_norm = axis / np.linalg.norm(axis)
            projections = np.dot(pillar['inlier_points'] - center, axis_norm)
            height = np.max(projections) - np.min(projections)
        else:
            height = 0

        print(f"Pillar {i+1}:")
        print(f"  Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
        print(f"  Radius: {radius:.3f} m")
        print(f"  Height: {height:.3f} m")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Inlier points: {len(pillar['inlier_points']):,}")
        print()


def main() -> None:
    """Main pipeline execution."""
    print("Red Pillar Detection Pipeline")
    print("="*60)
    print(f"Input file: {INPUT_PLY_PATH}")
    print(f"Output file: {OUTPUT_PLY_PATH}")
    print()

    # Check input file exists
    if not os.path.exists(INPUT_PLY_PATH):
        raise FileNotFoundError(f"Input PLY file not found: {INPUT_PLY_PATH}")

    overall_start_time = time.time()

    try:
        # Step 1: Load PLY file
        points, colors = load_ply_file_open3d(INPUT_PLY_PATH)

        # Step 2: Downsample point cloud
        points, colors = downsample_points(points, colors)

        # Step 3: Filter red points
        red_points, red_colors = filter_red_points_hsv(points, colors)

        if len(red_points) == 0:
            print("No red points found. Exiting.")
            return

        # Step 4: Cluster red points
        clusters, cluster_ids = cluster_red_points(red_points)

        if len(clusters) == 0:
            print("No valid clusters found. Exiting.")
            return

        # Step 5: Detect pillars
        detected_pillars = detect_pillars_in_clusters(clusters, cluster_ids)

        if len(detected_pillars) == 0:
            print("No pillars detected after fitting. Exiting.")
            return

        # Step 6: Create visualization
        output_points, output_colors = create_visualization_output(
            points, colors, detected_pillars
        )

        # Step 7: Save output
        save_ply_file_open3d(OUTPUT_PLY_PATH, output_points, output_colors)

        # Step 8: Print summary
        print_detection_summary(detected_pillars)

        total_time = time.time() - overall_start_time
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Output saved to: {OUTPUT_PLY_PATH}")

    except Exception as e:
        print(f"Error in pipeline execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
