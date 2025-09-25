#!/usr/bin/env python3
"""
Red Pillar Detection from PLY Point Cloud Files

This script detects red cylindrical pillars in PLY point cloud files using:
1. Point cloud downsampling for performance optimization
2. HSV color space segmentation for red color detection
3. DBSCAN clustering for grouping red points
4. PCA (Principal Component Analysis) for cylindrical structure detection
5. Geometric constraints validation for pillar characteristics
6. Visualization output with color-coded results

The pipeline is modularized into separate components for maintainability:
- config: Global configuration parameters
- ply_io: PLY file input/output operations
- downsampling: Point cloud downsampling for performance
- color_segmentation: HSV color filtering
- clustering: DBSCAN clustering
- pca_analysis: PCA-based cylindrical structure detection and pillar identification
- visualization: Output visualization generation
"""

import os
import time
import numpy as np
from typing import List, Dict, Any

from config import (
    INPUT_PLY_PATH, OUTPUT_PLY_PATH, ENABLE_INTERMEDIATE_SAVES,
    DOWNSAMPLED_PLY_PATH, CLUSTERED_PLY_PATH, RED_POINTS_ONLY_PLY_PATH
)
from ply_io import load_ply_file_open3d, save_ply_file_open3d
from downsampling import downsample_points
from color_segmentation import filter_red_points_hsv
from clustering import cluster_red_points
from pca_analysis import detect_pillars_with_pca
from visualization import create_visualization_output, create_clustering_visualization


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

    if ENABLE_INTERMEDIATE_SAVES:
        print(f"Intermediate saves: ENABLED")
        print(f"  Downsampled points: {DOWNSAMPLED_PLY_PATH}")
        print(f"  Red points only: {RED_POINTS_ONLY_PLY_PATH}")
        print(f"  Clustering results: {CLUSTERED_PLY_PATH}")
    else:
        print(f"Intermediate saves: DISABLED")
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

        # Save downsampled results if enabled
        if ENABLE_INTERMEDIATE_SAVES:
            try:
                print(
                    f"Saving downsampled point cloud to: {DOWNSAMPLED_PLY_PATH}")
                save_ply_file_open3d(DOWNSAMPLED_PLY_PATH, points, colors)
                print(f"Downsampled point cloud saved successfully")
            except Exception as e:
                print(
                    f"Warning: Failed to save downsampled point cloud: {str(e)}")

        # Step 3: Filter red points (with HSV conversion)
        red_points, red_colors, red_indices, hsv_colors = filter_red_points_hsv(
            points, colors, return_hsv=True)

        # Save HSV converted and red-only results if enabled
        if ENABLE_INTERMEDIATE_SAVES:
            try:
                # Save red-only point cloud if red points were found
                if len(red_points) > 0:
                    print(
                        f"Saving red points only to: {RED_POINTS_ONLY_PLY_PATH}")
                    save_ply_file_open3d(
                        RED_POINTS_ONLY_PLY_PATH, red_points, red_colors)
                    print(
                        f"Red points only PLY saved successfully ({len(red_points):,} points)")

            except Exception as e:
                print(
                    f"Warning: Failed to save HSV/red region results: {str(e)}")

        if len(red_points) == 0:
            print("No red points found. Exiting.")
            return

        # Step 4: Cluster red points
        clusters, cluster_ids, cluster_indices = cluster_red_points(
            red_points, red_indices)

        if len(clusters) == 0:
            print("No valid clusters found. Exiting.")
            return

        # Save clustering results if enabled
        if ENABLE_INTERMEDIATE_SAVES and len(clusters) > 0:
            try:
                print(
                    f"Saving clustering visualization to: {CLUSTERED_PLY_PATH}")
                cluster_viz_points, cluster_viz_colors = create_clustering_visualization(
                    points, colors, clusters, cluster_ids, cluster_indices
                )
                save_ply_file_open3d(CLUSTERED_PLY_PATH,
                                     cluster_viz_points, cluster_viz_colors)
                print(f"Clustering visualization saved successfully")
            except Exception as e:
                print(
                    f"Warning: Failed to save clustering visualization: {str(e)}")

        # Step 5: Detect pillars using PCA
        detected_pillars = detect_pillars_with_pca(clusters, cluster_ids)

        if len(detected_pillars) == 0:
            print("No pillars detected after PCA analysis. Exiting.")
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
