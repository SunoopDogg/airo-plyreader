"""
Color segmentation functions for the multi-color pillar detection pipeline.

This module handles HSV color space conversion and configurable color filtering
operations, supporting red, blue, and green color detection modes with
HSV-based filtering and configurable thresholds.
"""

import numpy as np
import cv2
import time
from config import (
    COLOR_DETECTION_MODE,
    HSV_RED_H_RANGES, HSV_RED_S_MIN, HSV_RED_V_MIN,
    HSV_BLUE_H_RANGES, HSV_BLUE_S_MIN, HSV_BLUE_V_MIN,
    HSV_GREEN_H_RANGES, HSV_GREEN_S_MIN, HSV_GREEN_V_MIN
)


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


def filter_colored_points_hsv(points: np.ndarray, colors: np.ndarray, return_hsv: bool = False):
    """
    Filter points based on HSV color criteria for the configured color mode.

    Args:
        points: numpy array of shape (N, 3) with [x, y, z] coordinates
        colors: numpy array of shape (N, 3) with [r, g, b] colors (0-255)
        return_hsv: Whether to return HSV colors of all points

    Returns:
        If return_hsv=False (default):
            Tuple of (colored_points, colored_colors, colored_indices)
        If return_hsv=True:
            Tuple of (colored_points, colored_colors, colored_indices, hsv_colors)
    """
    color_name = COLOR_DETECTION_MODE.lower()
    print(f"Filtering {color_name} points using HSV color space...")
    start_time = time.time()

    # Get color parameters based on mode
    if color_name == 'red':
        h_ranges = HSV_RED_H_RANGES
        s_min = HSV_RED_S_MIN
        v_min = HSV_RED_V_MIN
    elif color_name == 'blue':
        h_ranges = HSV_BLUE_H_RANGES
        s_min = HSV_BLUE_S_MIN
        v_min = HSV_BLUE_V_MIN
    elif color_name == 'green':
        h_ranges = HSV_GREEN_H_RANGES
        s_min = HSV_GREEN_S_MIN
        v_min = HSV_GREEN_V_MIN
    else:
        raise ValueError(
            f"Unsupported color detection mode: {COLOR_DETECTION_MODE}")

    # Convert RGB to HSV
    hsv_colors = rgb_to_hsv_vectorized(colors)

    # Extract HSV components
    h = hsv_colors[:, 0]  # Hue (0-180)
    s = hsv_colors[:, 1]  # Saturation (0-255)
    v = hsv_colors[:, 2]  # Value (0-255)

    # Convert thresholds to OpenCV scale
    s_min_scaled = s_min  # Already in 0-1 range
    v_min_scaled = v_min * 255  # Convert to 0-255 range

    # Create color mask
    color_mask = np.zeros(len(h), dtype=bool)

    # Apply HSV hue ranges for the selected color
    for h_min, h_max in h_ranges:
        hue_mask = (h >= h_min) & (h <= h_max)
        color_mask |= hue_mask

    # Apply saturation and value constraints
    color_mask &= (s >= s_min_scaled) & (v >= v_min_scaled)

    # Filter points, colors, and get indices
    colored_points = points[color_mask]
    colored_colors = colors[color_mask]
    colored_indices = np.where(color_mask)[0]  # Get original indices

    filter_time = time.time() - start_time
    print(f"Found {len(colored_points):,} {color_name} points ({len(colored_points)/len(points)*100:.2f}%) "
          f"in {filter_time:.2f} seconds")

    # Debug: Show HSV statistics for filtered points
    if len(colored_points) > 0:
        filtered_hsv = rgb_to_hsv_vectorized(colored_colors)
        print(f"  {color_name.title()} points HSV ranges: H[{filtered_hsv[:, 0].min():.1f}-{filtered_hsv[:, 0].max():.1f}], "
              f"S[{filtered_hsv[:, 1].min():.3f}-{filtered_hsv[:, 1].max():.3f}], "
              f"V[{filtered_hsv[:, 2].min():.1f}-{filtered_hsv[:, 2].max():.1f}]")

    # Return different tuples based on return_hsv flag for backwards compatibility
    if return_hsv:
        return colored_points, colored_colors, colored_indices, hsv_colors
    else:
        return colored_points, colored_colors, colored_indices
