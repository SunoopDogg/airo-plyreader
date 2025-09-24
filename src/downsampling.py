"""
Point cloud downsampling module for PLY processing pipeline.

This module provides native Open3D downsampling functionality to reduce point cloud
density while preserving spatial distribution and color information. Four downsampling
methods are supported:
1. Voxel grid downsampling - spatially uniform reduction (Open3D native)
2. Random sampling - statistical reduction (Open3D native)
3. Uniform sampling - every k-th point selection (Open3D native)
4. Farthest point sampling - maximizes spatial coverage (Open3D native)

The downsampling step helps improve processing speed for subsequent pipeline
stages while maintaining representative point distribution using optimized
Open3D algorithms.
"""

import time
import numpy as np
import open3d as o3d
from typing import Tuple

from config import (
    DOWNSAMPLING_ENABLED,
    DOWNSAMPLING_METHOD,
    DOWNSAMPLING_VOXEL_SIZE,
    DOWNSAMPLING_TARGET_RATIO,
    DOWNSAMPLING_UNIFORM_K,
    DOWNSAMPLING_FARTHEST_POINTS,
)


def numpy_to_o3d_pointcloud(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Convert numpy arrays to Open3D PointCloud object.

    Args:
        points: Point cloud array of shape (N, 3)
        colors: Color array of shape (N, 3) with values 0-255

    Returns:
        Open3D PointCloud object
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    # Convert colors from [0-255] to [0-1] range for Open3D
    if colors is not None and len(colors) > 0:
        colors_normalized = colors.astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors_normalized)

    return pcd


def o3d_pointcloud_to_numpy(pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Open3D PointCloud object to numpy arrays.

    Args:
        pcd: Open3D PointCloud object

    Returns:
        Tuple of (points, colors) as numpy arrays
    """
    points = np.asarray(pcd.points, dtype=np.float32)

    # Convert colors from [0-1] back to [0-255] range
    if pcd.has_colors():
        colors = (np.asarray(pcd.colors) * 255.0).astype(np.uint8)
    else:
        colors = np.zeros((len(points), 3), dtype=np.uint8)

    return points, colors


def downsample_voxel_grid_o3d(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample point cloud using Open3D's native voxel grid method.

    Args:
        points: Point cloud array of shape (N, 3)
        colors: Color array of shape (N, 3) with values 0-255
        voxel_size: Size of voxel grid in meters

    Returns:
        Tuple of (downsampled_points, downsampled_colors)
    """
    if len(points) == 0:
        return points, colors

    print(f"    Open3D voxel grid downsampling with voxel size: {voxel_size}m")

    # Convert to Open3D format
    pcd = numpy_to_o3d_pointcloud(points, colors)

    # Apply Open3D voxel downsampling
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)

    # Convert back to numpy
    return o3d_pointcloud_to_numpy(downsampled_pcd)


def downsample_random_o3d(
    points: np.ndarray,
    colors: np.ndarray,
    target_ratio: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample point cloud using Open3D's native random sampling.

    Args:
        points: Point cloud array of shape (N, 3)
        colors: Color array of shape (N, 3) with values 0-255
        target_ratio: Target ratio for reduction (0.1 = keep 10% of points)

    Returns:
        Tuple of (downsampled_points, downsampled_colors)
    """
    if len(points) == 0 or target_ratio >= 1.0:
        return points, colors

    print(
        f"    Open3D random downsampling with target ratio: {target_ratio:.2%}")

    # Convert to Open3D format
    pcd = numpy_to_o3d_pointcloud(points, colors)

    # Apply Open3D random downsampling
    downsampled_pcd = pcd.random_down_sample(sampling_ratio=target_ratio)

    # Convert back to numpy
    return o3d_pointcloud_to_numpy(downsampled_pcd)


def downsample_uniform_o3d(
    points: np.ndarray,
    colors: np.ndarray,
    every_k_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample point cloud using Open3D's uniform sampling (every k-th point).

    Args:
        points: Point cloud array of shape (N, 3)
        colors: Color array of shape (N, 3) with values 0-255
        every_k_points: Keep every k-th point (1 = keep all, 10 = keep every 10th)

    Returns:
        Tuple of (downsampled_points, downsampled_colors)
    """
    if len(points) == 0 or every_k_points <= 1:
        return points, colors

    print(
        f"    Open3D uniform downsampling, keeping every {every_k_points} points")

    # Convert to Open3D format
    pcd = numpy_to_o3d_pointcloud(points, colors)

    # Apply Open3D uniform downsampling
    downsampled_pcd = pcd.uniform_down_sample(every_k_points=every_k_points)

    # Convert back to numpy
    return o3d_pointcloud_to_numpy(downsampled_pcd)


def downsample_farthest_point_o3d(
    points: np.ndarray,
    colors: np.ndarray,
    num_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample point cloud using Open3D's farthest point sampling.

    This method provides better spatial coverage than random sampling by
    iteratively selecting points that are farthest from existing selection.

    Args:
        points: Point cloud array of shape (N, 3)
        colors: Color array of shape (N, 3) with values 0-255
        num_samples: Number of points to keep

    Returns:
        Tuple of (downsampled_points, downsampled_colors)
    """
    if len(points) == 0 or num_samples >= len(points):
        return points, colors

    print(f"    Open3D farthest point downsampling to {num_samples:,} points")

    # Convert to Open3D format
    pcd = numpy_to_o3d_pointcloud(points, colors)

    # Apply Open3D farthest point downsampling
    downsampled_pcd = pcd.farthest_point_down_sample(num_samples=num_samples)

    # Convert back to numpy
    return o3d_pointcloud_to_numpy(downsampled_pcd)


def downsample_points(
    points: np.ndarray,
    colors: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Main downsampling function that routes to appropriate Open3D method.

    This function applies downsampling based on configuration settings using
    native Open3D algorithms for optimal performance. Supports four methods:
    - 'voxel': Voxel grid downsampling
    - 'random': Random sampling
    - 'uniform': Uniform sampling (every k-th point)
    - 'farthest_point': Farthest point sampling

    Args:
        points: Original point cloud array of shape (N, 3)
        colors: Original color array of shape (N, 3) with values 0-255

    Returns:
        Tuple of (processed_points, processed_colors)
    """
    if not DOWNSAMPLING_ENABLED:
        print("  Downsampling: DISABLED - using original point cloud")
        return points, colors

    print(f"  Downsampling: ENABLED - Method: {DOWNSAMPLING_METHOD}")
    print(f"    Original points: {len(points):,}")

    start_time = time.time()

    try:
        if DOWNSAMPLING_METHOD == 'voxel':
            downsampled_points, downsampled_colors = downsample_voxel_grid_o3d(
                points, colors, DOWNSAMPLING_VOXEL_SIZE
            )
        elif DOWNSAMPLING_METHOD == 'random':
            downsampled_points, downsampled_colors = downsample_random_o3d(
                points, colors, DOWNSAMPLING_TARGET_RATIO
            )
        elif DOWNSAMPLING_METHOD == 'uniform':
            downsampled_points, downsampled_colors = downsample_uniform_o3d(
                points, colors, DOWNSAMPLING_UNIFORM_K
            )
        elif DOWNSAMPLING_METHOD == 'farthest_point':
            downsampled_points, downsampled_colors = downsample_farthest_point_o3d(
                points, colors, DOWNSAMPLING_FARTHEST_POINTS
            )
        else:
            raise ValueError(
                f"Unknown downsampling method: {DOWNSAMPLING_METHOD}. "
                f"Supported methods: 'voxel', 'random', 'uniform', 'farthest_point'"
            )

        # Calculate reduction statistics
        reduction_ratio = len(downsampled_points) / \
            len(points) if len(points) > 0 else 0
        points_removed = len(points) - len(downsampled_points)

        processing_time = time.time() - start_time

        print(f"    Downsampled points: {len(downsampled_points):,}")
        print(f"    Points removed: {points_removed:,}")
        print(f"    Reduction ratio: {reduction_ratio:.2%}")
        print(f"    Processing time: {processing_time:.2f} seconds")

        return downsampled_points, downsampled_colors

    except Exception as e:
        print(f"    ERROR in downsampling: {str(e)}")
        print("    Falling back to original point cloud")
        return points, colors
