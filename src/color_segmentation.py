"""
Color segmentation functions for the red pillar detection pipeline.

This module handles HSV color space conversion and red color filtering
operations, including vectorized RGB to HSV conversion and HSV-based
red point filtering with configurable thresholds.
"""

import numpy as np
import cv2
import time
from typing import Tuple
from config import HSV_RED_H_RANGES, HSV_RED_S_MIN, HSV_RED_V_MIN


def rgb_to_hsv_vectorized(rgb_colors: np.ndarray) -> np.ndarray:
    """
    Convert RGB colors to HSV using OpenCV (vectorized).

    Args:
        rgb_colors: numpy array of shape (N, 3) with RGB values (0-255)

    Returns:
        numpy array of shape (N, 3) with HSV values:
        - H: 0-180 (OpenCV format)
        - S: 0-255
        - V: 0-255
    """
    # OpenCV expects BGR format, but we'll work with RGB directly
    # Convert to float32 for OpenCV
    rgb_float = rgb_colors.astype(np.float32)

    # Reshape for OpenCV processing
    rgb_image = rgb_float.reshape(1, -1, 3)

    # Convert RGB to HSV
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # Reshape back to (N, 3)
    hsv_colors = hsv_image.reshape(-1, 3)

    return hsv_colors


def filter_red_points_hsv(points: np.ndarray, colors: np.ndarray, return_hsv: bool = False):
    """
    Filter points based on HSV red color criteria.

    Args:
        points: numpy array of shape (N, 3) with [x, y, z] coordinates
        colors: numpy array of shape (N, 3) with [r, g, b] colors (0-255)
        return_hsv: Whether to return HSV colors of all points

    Returns:
        If return_hsv=False (default):
            Tuple of (red_points, red_colors, red_indices)
        If return_hsv=True:
            Tuple of (red_points, red_colors, red_indices, hsv_colors)
    """
    print("Filtering red points using HSV color space...")
    start_time = time.time()

    # Convert RGB to HSV
    hsv_colors = rgb_to_hsv_vectorized(colors)

    # Extract HSV components
    h = hsv_colors[:, 0]  # Hue (0-180)
    s = hsv_colors[:, 1]  # Saturation (0-255)
    v = hsv_colors[:, 2]  # Value (0-255)

    # Convert thresholds to OpenCV scale
    # Note: OpenCV HSV saturation is 0-1 (float), value is 0-255
    s_min = HSV_RED_S_MIN  # Already in 0-1 range
    v_min = HSV_RED_V_MIN * 255  # Convert to 0-255 range

    # Create red color mask
    red_mask = np.zeros(len(h), dtype=bool)

    # Apply HSV red hue ranges
    for h_min, h_max in HSV_RED_H_RANGES:
        hue_mask = (h >= h_min) & (h <= h_max)
        red_mask |= hue_mask

    # Apply saturation and value constraints
    red_mask &= (s >= s_min) & (v >= v_min)

    # Filter points, colors, and get indices
    red_points = points[red_mask]
    red_colors = colors[red_mask]
    red_indices = np.where(red_mask)[0]  # Get original indices of red points

    filter_time = time.time() - start_time
    print(f"Found {len(red_points):,} red points ({len(red_points)/len(points)*100:.2f}%) "
          f"in {filter_time:.2f} seconds")

    # Debug: Show HSV statistics for filtered points
    if len(red_points) > 0:
        red_hsv = rgb_to_hsv_vectorized(red_colors)
        print(f"  Red points HSV ranges: H[{red_hsv[:, 0].min():.1f}-{red_hsv[:, 0].max():.1f}], "
              f"S[{red_hsv[:, 1].min():.3f}-{red_hsv[:, 1].max():.3f}], "
              f"V[{red_hsv[:, 2].min():.1f}-{red_hsv[:, 2].max():.1f}]")

    # Return different tuples based on return_hsv flag for backwards compatibility
    if return_hsv:
        return red_points, red_colors, red_indices, hsv_colors
    else:
        return red_points, red_colors, red_indices
