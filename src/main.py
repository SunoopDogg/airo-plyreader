#!/usr/bin/env python3
"""
Multi-Color Pillar Detection from PLY Point Cloud Files

This script detects cylindrical pillars of any color in PLY point cloud files using:
1. Point cloud downsampling for performance optimization
2. HSV color space segmentation for configurable color detection
3. DBSCAN clustering for grouping colored points
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
import cupy as cp
from typing import List, Dict, Any

from pathlib import Path
from .config import (
    PLY_DIR, OUTPUT_PLY_PATH, ENABLE_INTERMEDIATE_SAVES,
    DOWNSAMPLED_PLY_PATH, CLUSTERED_PLY_PATH, COLOR_DETECTION_MODE,
    get_colored_points_path, GPU_DEVICE_ID, ENABLE_VISUALIZATION,
)
from .file_io.ply_io import load_ply_file_open3d, save_ply_file_open3d
from .preprocessing.downsampling import downsample_points
from .preprocessing.color_segmentation import filter_colored_points_hsv
from .analysis.clustering import cluster_colored_points
from .analysis.pca_analysis import detect_pillars_with_pca
from .visualization.visualization import create_visualization_output, create_clustering_visualization, launch_all_viewers


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def select_ply_file() -> str:
    """Scan PLY_DIR and let the user pick a file interactively."""
    ply_dir = Path(PLY_DIR)
    if not ply_dir.is_dir():
        raise FileNotFoundError(f"PLY directory not found: {PLY_DIR}")

    ply_files = sorted(ply_dir.glob("*.ply"))
    if not ply_files:
        raise FileNotFoundError(f"No .ply files found in {PLY_DIR}")

    if len(ply_files) == 1:
        selected = ply_files[0]
        print(f"Auto-selected (only file): {selected.name}")
        return str(selected)

    print("\nAvailable PLY files:")
    for i, f in enumerate(ply_files, 1):
        print(f"  {i}) {f.name}")
    print()

    while True:
        try:
            choice = int(input(f"Select file number (1-{len(ply_files)}): "))
            if 1 <= choice <= len(ply_files):
                selected = ply_files[choice - 1]
                print(f"Selected: {selected.name}\n")
                return str(selected)
        except (ValueError, EOFError):
            pass
        print(f"Please enter a number between 1 and {len(ply_files)}.")


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
    input_ply_path = select_ply_file()

    print(f"{COLOR_DETECTION_MODE.title()} Pillar Detection Pipeline")
    print("="*60)
    print(f"Color detection mode: {COLOR_DETECTION_MODE.upper()}")
    print(f"Input file: {input_ply_path}")
    print(f"Output file: {OUTPUT_PLY_PATH}")

    if ENABLE_INTERMEDIATE_SAVES:
        colored_points_path = get_colored_points_path()
        print(f"Intermediate saves: ENABLED")
        print(f"  Downsampled points: {DOWNSAMPLED_PLY_PATH}")
        print(f"  {COLOR_DETECTION_MODE.title()} points only: {colored_points_path}")
        print(f"  Clustering results: {CLUSTERED_PLY_PATH}")
    else:
        print(f"Intermediate saves: DISABLED")
    print()

    overall_start_time = time.time()

    try:
        # GPU initialization
        cp.cuda.Device(GPU_DEVICE_ID).use()
        print(f"Using GPU device: {GPU_DEVICE_ID}")

        # Step 1: Load PLY file (CPU)
        points, colors = load_ply_file_open3d(input_ply_path)

        # CPU → GPU transfer (once)
        points_gpu = cp.asarray(points, dtype=cp.float32)
        colors_gpu = cp.asarray(colors, dtype=cp.uint8)
        print(f"Transferred {len(points):,} points to GPU")

        # Step 2: Downsample point cloud (GPU)
        points_gpu, colors_gpu = downsample_points(points_gpu, colors_gpu)

        # Save downsampled results if enabled (requires GPU→CPU)
        if ENABLE_INTERMEDIATE_SAVES:
            try:
                print(f"Saving downsampled point cloud to: {DOWNSAMPLED_PLY_PATH}")
                save_ply_file_open3d(
                    DOWNSAMPLED_PLY_PATH,
                    cp.asnumpy(points_gpu),
                    cp.asnumpy(colors_gpu),
                )
                print(f"Downsampled point cloud saved successfully")
            except Exception as e:
                print(f"Warning: Failed to save downsampled point cloud: {str(e)}")

        # Step 3: Filter colored points (GPU)
        colored_points, colored_colors, colored_indices = (
            filter_colored_points_hsv(points_gpu, colors_gpu)
        )

        # Save color-filtered results if enabled
        if ENABLE_INTERMEDIATE_SAVES:
            try:
                if len(colored_points) > 0:
                    colored_points_path = get_colored_points_path()
                    print(f"Saving {COLOR_DETECTION_MODE.lower()} points only to: {colored_points_path}")
                    save_ply_file_open3d(
                        colored_points_path,
                        cp.asnumpy(colored_points),
                        cp.asnumpy(colored_colors),
                    )
                    print(f"{COLOR_DETECTION_MODE.title()} points only PLY saved successfully ({len(colored_points):,} points)")
            except Exception as e:
                print(f"Warning: Failed to save color-filtered results: {str(e)}")

        if len(colored_points) == 0:
            print(f"No {COLOR_DETECTION_MODE.lower()} points found. Exiting.")
            return

        # Step 4: Cluster colored points (GPU)
        clusters, cluster_ids, cluster_indices = cluster_colored_points(
            colored_points, colored_indices
        )

        if len(clusters) == 0:
            print("No valid clusters found. Exiting.")
            return

        # GPU → CPU transfer for visualization (once, reused below)
        points_np = cp.asnumpy(points_gpu)
        colors_np = cp.asnumpy(colors_gpu)

        # Save clustering results if enabled
        if ENABLE_INTERMEDIATE_SAVES and len(clusters) > 0:
            try:
                print(f"Saving clustering visualization to: {CLUSTERED_PLY_PATH}")
                clusters_np = [cp.asnumpy(c) for c in clusters]
                indices_np = [cp.asnumpy(idx) for idx in cluster_indices]
                cluster_viz_points, cluster_viz_colors = create_clustering_visualization(
                    points_np, colors_np, clusters_np, cluster_ids, indices_np
                )
                save_ply_file_open3d(CLUSTERED_PLY_PATH, cluster_viz_points, cluster_viz_colors)
                print(f"Clustering visualization saved successfully")
            except Exception as e:
                print(f"Warning: Failed to save clustering visualization: {str(e)}")

        # Step 5: Detect pillars using PCA (GPU, returns NumPy results)
        detected_pillars = detect_pillars_with_pca(clusters, cluster_ids)

        if len(detected_pillars) == 0:
            print("No pillars detected after PCA analysis. Exiting.")
            return

        # Step 6: Create visualization (CPU — pillar results already NumPy)
        output_points, output_colors = create_visualization_output(
            points_np, colors_np, detected_pillars
        )

        # Step 7: Save output
        save_ply_file_open3d(OUTPUT_PLY_PATH, output_points, output_colors)

        # Step 8: Print summary
        print_detection_summary(detected_pillars)

        total_time = time.time() - overall_start_time
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Output saved to: {OUTPUT_PLY_PATH}")

        # Step 9: Launch interactive viewers
        if ENABLE_VISUALIZATION:
            launch_all_viewers()

    except Exception as e:
        print(f"Error in pipeline execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
