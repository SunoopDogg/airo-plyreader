"""
GUI-based ROI (Region of Interest) selection and GPU cropping.

Provides a Matplotlib top-down scatter plot for interactive rectangular
region selection, and GPU-accelerated cropping via CuPy boolean masking.
"""

import cupy as cp

from ..core.gpu import PointCloudGPU


def crop_to_roi(
    cloud: PointCloudGPU,
    roi: tuple[float, float, float, float],
) -> PointCloudGPU:
    """Crop point cloud to a 2D XY bounding box on GPU.

    Args:
        cloud: Input point cloud on GPU.
        roi: (x_min, x_max, y_min, y_max) bounds.

    Returns:
        New PointCloudGPU containing only points within the ROI.
    """
    x_min, x_max, y_min, y_max = roi
    x = cloud.points[:, 0]
    y = cloud.points[:, 1]

    mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)

    cropped = PointCloudGPU(cloud.points[mask], cloud.colors[mask])

    cp.get_default_memory_pool().free_all_blocks()

    print(f"ROI crop: {len(cloud):,} -> {len(cropped):,} points "
          f"(X[{x_min:.2f}, {x_max:.2f}], Y[{y_min:.2f}, {y_max:.2f}])")

    return cropped
