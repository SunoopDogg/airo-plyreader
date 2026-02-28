"""
Shared utilities for the pillar detection pipeline.
"""

import numpy as np

from ..config import ENABLE_INTERMEDIATE_SAVES
from ..file_io.ply_io import save_ply_file_open3d


def save_intermediate(points: np.ndarray, colors: np.ndarray, path: str, description: str) -> None:
    """Save intermediate pipeline result if ENABLE_INTERMEDIATE_SAVES is True."""
    if not ENABLE_INTERMEDIATE_SAVES:
        return
    try:
        print(f"Saving {description} to: {path}")
        save_ply_file_open3d(path, points, colors)
        print(f"{description} saved successfully")
    except Exception as e:
        print(f"Warning: Failed to save {description}: {str(e)}")
