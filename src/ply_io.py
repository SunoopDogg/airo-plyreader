"""
PLY file input/output operations for the red pillar detection pipeline.

This module handles all PLY file reading and writing operations, including
coordinate and color extraction, data type conversions, and file format
management.
"""

import numpy as np
import time
from typing import Tuple
from plyfile import PlyData, PlyElement


def load_ply_file(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
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


def save_ply_file(file_path: str, points: np.ndarray, colors: np.ndarray) -> None:
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
