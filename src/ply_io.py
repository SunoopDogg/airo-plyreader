"""
PLY file input/output operations for the red pillar detection pipeline.

This module handles all PLY file reading and writing operations, including
coordinate and color extraction, data type conversions, and file format
management.
"""

import numpy as np
import time
from typing import Tuple, Optional, Union, Dict
from plyfile import PlyData, PlyElement
import open3d as o3d
from pathlib import Path


# =============================================================================
# PLYFILE-BASED PLY I/O FUNCTIONS
# =============================================================================

def load_ply_file_plyfile(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load PLY file and extract coordinates and colors.

    Args:
        file_path: Path to PLY file

    Returns:
        Tuple of (points, colors) where:
        - points: numpy array of shape (N, 3) with [x, y, z] coordinates
        - colors: numpy array of shape (N, 3) with [r, g, b] colors (0-255)
    """
    print(f"Loading PLY file: {file_path}")
    start_time = time.time()

    try:
        plydata = PlyData.read(file_path)
        vertices = plydata['vertex']

        # Extract coordinates (handle both float32 and double precision)
        points = np.column_stack((
            vertices['x'].astype(np.float64),
            vertices['y'].astype(np.float64),
            vertices['z'].astype(np.float64)
        ))

        # Extract colors (convert to uint8 if needed)
        colors = np.column_stack((
            vertices['red'].astype(np.uint8),
            vertices['green'].astype(np.uint8),
            vertices['blue'].astype(np.uint8)
        ))

        load_time = time.time() - start_time
        print(f"Loaded {len(points):,} points in {load_time:.2f} seconds")
        print(f"Point cloud bounds: X[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
              f"Y[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
              f"Z[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

        return points, colors

    except Exception as e:
        raise RuntimeError(f"Failed to load PLY file {file_path}: {str(e)}")


def save_ply_file_plyfile(file_path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """
    Save points and colors to PLY file.

    Args:
        file_path: Output PLY file path
        points: numpy array of shape (N, 3) with [x, y, z] coordinates
        colors: numpy array of shape (N, 3) with [r, g, b] colors (0-255)
    """
    print(f"Saving output PLY file: {file_path}")
    start_time = time.time()

    try:
        # Create vertex array with proper data types
        vertex_data = np.array([
            (points[i, 0], points[i, 1], points[i, 2],
             colors[i, 0], colors[i, 1], colors[i, 2])
            for i in range(len(points))
        ], dtype=[
            ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ])

        # Create PLY element and save
        el = PlyElement.describe(vertex_data, 'vertex')
        PlyData([el], text=True).write(file_path)

        save_time = time.time() - start_time
        print(f"Saved {len(points):,} points in {save_time:.2f} seconds")

    except Exception as e:
        raise RuntimeError(f"Failed to save PLY file {file_path}: {str(e)}")


# =============================================================================
# OPEN3D-BASED PLY I/O FUNCTIONS
# =============================================================================

def load_ply_file_open3d(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load PLY file using Open3D and extract coordinates and colors.

    Open3D provides optimized I/O with better memory management for large point clouds.

    Args:
        file_path: Path to PLY file

    Returns:
        Tuple of (points, colors) where:
        - points: numpy array of shape (N, 3) with [x, y, z] coordinates
        - colors: numpy array of shape (N, 3) with [r, g, b] colors (0-255)
    """
    print(f"Loading PLY file with Open3D: {file_path}")
    start_time = time.time()

    try:
        # Load point cloud using Open3D
        pcd = o3d.io.read_point_cloud(file_path)

        if len(pcd.points) == 0:
            raise ValueError(f"No points found in PLY file: {file_path}")

        # Extract points
        points = np.asarray(pcd.points, dtype=np.float64)

        # Extract colors (Open3D uses [0,1] range, convert to [0,255])
        if pcd.has_colors():
            colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        else:
            # If no colors, create default white colors
            colors = np.full((len(points), 3), 255, dtype=np.uint8)
            print("Warning: No colors found in PLY file, using default white colors")

        load_time = time.time() - start_time
        print(
            f"Loaded {len(points):,} points with Open3D in {load_time:.2f} seconds")
        print(f"Point cloud bounds: X[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
              f"Y[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
              f"Z[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

        return points, colors

    except Exception as e:
        raise RuntimeError(
            f"Failed to load PLY file with Open3D {file_path}: {str(e)}")


def save_ply_file_open3d(file_path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """
    Save points and colors to PLY file using Open3D.

    Open3D provides optimized I/O with better compression and format options.

    Args:
        file_path: Output PLY file path
        points: numpy array of shape (N, 3) with [x, y, z] coordinates
        colors: numpy array of shape (N, 3) with [r, g, b] colors (0-255)
    """
    print(f"Saving output PLY file with Open3D: {file_path}")
    start_time = time.time()

    try:
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

        # Convert colors from [0,255] to [0,1] range for Open3D
        colors_normalized = colors.astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors_normalized)

        # Ensure output directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Save point cloud
        success = o3d.io.write_point_cloud(file_path, pcd, write_ascii=True)

        if not success:
            raise RuntimeError("Open3D failed to write point cloud")

        save_time = time.time() - start_time
        print(
            f"Saved {len(points):,} points with Open3D in {save_time:.2f} seconds")

    except Exception as e:
        raise RuntimeError(
            f"Failed to save PLY file with Open3D {file_path}: {str(e)}")


def load_ply_as_o3d(file_path: str) -> o3d.geometry.PointCloud:
    """
    Load PLY file directly as Open3D PointCloud object.

    Args:
        file_path: Path to PLY file

    Returns:
        Open3D PointCloud object
    """
    print(f"Loading PLY as Open3D PointCloud: {file_path}")
    start_time = time.time()

    try:
        pcd = o3d.io.read_point_cloud(file_path)

        if len(pcd.points) == 0:
            raise ValueError(f"No points found in PLY file: {file_path}")

        load_time = time.time() - start_time
        print(
            f"Loaded Open3D PointCloud with {len(pcd.points):,} points in {load_time:.2f} seconds")

        return pcd

    except Exception as e:
        raise RuntimeError(
            f"Failed to load PLY as Open3D PointCloud {file_path}: {str(e)}")


def save_o3d_as_ply(file_path: str, pcd: o3d.geometry.PointCloud) -> None:
    """
    Save Open3D PointCloud directly to PLY file.

    Args:
        file_path: Output PLY file path
        pcd: Open3D PointCloud object
    """
    print(f"Saving Open3D PointCloud as PLY: {file_path}")
    start_time = time.time()

    try:
        # Ensure output directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Save point cloud
        success = o3d.io.write_point_cloud(file_path, pcd, write_ascii=True)

        if not success:
            raise RuntimeError("Open3D failed to write point cloud")

        save_time = time.time() - start_time
        print(
            f"Saved Open3D PointCloud with {len(pcd.points):,} points in {save_time:.2f} seconds")

    except Exception as e:
        raise RuntimeError(
            f"Failed to save Open3D PointCloud as PLY {file_path}: {str(e)}")


# =============================================================================
# OPEN3D UTILITY FUNCTIONS
# =============================================================================

def numpy_to_o3d(points: np.ndarray, colors: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
    """
    Convert numpy arrays to Open3D PointCloud object.

    Args:
        points: numpy array of shape (N, 3) with [x, y, z] coordinates
        colors: optional numpy array of shape (N, 3) with [r, g, b] colors (0-255)

    Returns:
        Open3D PointCloud object
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    if colors is not None:
        # Convert colors from [0,255] to [0,1] range for Open3D
        colors_normalized = colors.astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors_normalized)

    return pcd


def o3d_to_numpy(pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Open3D PointCloud object to numpy arrays.

    Args:
        pcd: Open3D PointCloud object

    Returns:
        Tuple of (points, colors) where:
        - points: numpy array of shape (N, 3) with [x, y, z] coordinates
        - colors: numpy array of shape (N, 3) with [r, g, b] colors (0-255)
    """
    points = np.asarray(pcd.points, dtype=np.float64)

    if pcd.has_colors():
        # Convert colors from [0,1] to [0,255] range
        colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    else:
        # If no colors, create default white colors
        colors = np.full((len(points), 3), 255, dtype=np.uint8)

    return points, colors


def validate_point_cloud(points: np.ndarray, colors: np.ndarray) -> bool:
    """
    Validate point cloud data consistency and basic properties.

    Args:
        points: numpy array of shape (N, 3) with [x, y, z] coordinates
        colors: numpy array of shape (N, 3) with [r, g, b] colors (0-255)

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails
    """
    if points.shape[0] != colors.shape[0]:
        raise ValueError(
            f"Points and colors array length mismatch: {points.shape[0]} vs {colors.shape[0]}")

    if points.shape[1] != 3:
        raise ValueError(
            f"Points array must have 3 columns (x,y,z), got {points.shape[1]}")

    if colors.shape[1] != 3:
        raise ValueError(
            f"Colors array must have 3 columns (r,g,b), got {colors.shape[1]}")

    # Check for valid numeric values
    if not np.all(np.isfinite(points)):
        raise ValueError("Points array contains invalid values (NaN or Inf)")

    # Check color range
    if np.any(colors < 0) or np.any(colors > 255):
        raise ValueError("Colors must be in range [0, 255]")

    if len(points) == 0:
        raise ValueError("Empty point cloud")

    print(f"Point cloud validation passed: {len(points):,} points")
    return True


def get_point_cloud_info(file_path: str, use_open3d: bool = True) -> Dict[str, Union[int, float, str]]:
    """
    Get basic information about a PLY file without fully loading it.

    Args:
        file_path: Path to PLY file
        use_open3d: Whether to use Open3D for loading (faster for large files)

    Returns:
        Dictionary with point cloud information
    """
    try:
        if use_open3d:
            pcd = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd.points)
            has_colors = pcd.has_colors()
            if has_colors:
                colors = np.asarray(pcd.colors)
        else:
            points, colors = load_ply_file_plyfile(file_path)
            has_colors = colors is not None

        info = {
            'file_path': file_path,
            'file_size_mb': Path(file_path).stat().st_size / (1024 * 1024),
            'num_points': len(points),
            'has_colors': has_colors,
            'x_range': [float(points[:, 0].min()), float(points[:, 0].max())],
            'y_range': [float(points[:, 1].min()), float(points[:, 1].max())],
            'z_range': [float(points[:, 2].min()), float(points[:, 2].max())],
            'centroid': [float(points[:, 0].mean()), float(points[:, 1].mean()), float(points[:, 2].mean())]
        }

        if has_colors and use_open3d:
            # Colors in Open3D are [0,1], convert to [0,255] for reporting
            colors_uint8 = (colors * 255).astype(np.uint8)
            info.update({
                'r_range': [int(colors_uint8[:, 0].min()), int(colors_uint8[:, 0].max())],
                'g_range': [int(colors_uint8[:, 1].min()), int(colors_uint8[:, 1].max())],
                'b_range': [int(colors_uint8[:, 2].min()), int(colors_uint8[:, 2].max())]
            })
        elif has_colors:
            info.update({
                'r_range': [int(colors[:, 0].min()), int(colors[:, 0].max())],
                'g_range': [int(colors[:, 1].min()), int(colors[:, 1].max())],
                'b_range': [int(colors[:, 2].min()), int(colors[:, 2].max())]
            })

        return info

    except Exception as e:
        raise RuntimeError(
            f"Failed to get point cloud info for {file_path}: {str(e)}")


# =============================================================================
# BACKEND SELECTION FUNCTIONS
# =============================================================================

def load_ply_with_backend(file_path: str, backend: str = "plyfile") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load PLY file using specified backend.

    Args:
        file_path: Path to PLY file
        backend: Backend to use ("plyfile" or "open3d")

    Returns:
        Tuple of (points, colors) where:
        - points: numpy array of shape (N, 3) with [x, y, z] coordinates
        - colors: numpy array of shape (N, 3) with [r, g, b] colors (0-255)
    """
    if backend.lower() == "open3d":
        return load_ply_file_open3d(file_path)
    elif backend.lower() == "plyfile":
        return load_ply_file_plyfile(file_path)
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Choose 'plyfile' or 'open3d'")


def save_ply_with_backend(file_path: str, points: np.ndarray, colors: np.ndarray, backend: str = "plyfile") -> None:
    """
    Save PLY file using specified backend.

    Args:
        file_path: Output PLY file path
        points: numpy array of shape (N, 3) with [x, y, z] coordinates
        colors: numpy array of shape (N, 3) with [r, g, b] colors (0-255)
        backend: Backend to use ("plyfile" or "open3d")
    """
    if backend.lower() == "open3d":
        save_ply_file_open3d(file_path, points, colors)
    elif backend.lower() == "plyfile":
        save_ply_file_plyfile(file_path, points, colors)
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Choose 'plyfile' or 'open3d'")
