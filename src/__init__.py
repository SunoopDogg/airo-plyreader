"""
Red Pillar Detection Pipeline

A modular pipeline for detecting red cylindrical pillars in PLY point cloud files.

This package provides a complete modular implementation of the red pillar detection
algorithm, organized into logical components for maintainability and reusability.

Modules:
    config: Global configuration parameters
    ply_io: PLY file input/output operations
    color_segmentation: HSV color filtering functions
    clustering: DBSCAN clustering functionality
    visualization: Visualization and output generation
    main: Main pipeline execution
"""


# Import main components for easy access
from .main import main, print_detection_summary
from .config import *
from .ply_io import (
    load_ply_file_open3d, save_ply_file_open3d,
    load_ply_as_o3d, save_o3d_as_ply,
    numpy_to_o3d, o3d_to_numpy,
    validate_point_cloud, get_point_cloud_info
)
from .color_segmentation import filter_red_points_hsv
from .clustering import cluster_red_points
from .visualization import create_visualization_output
