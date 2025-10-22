"""
Multi-Color Pillar Detection Pipeline

A modular pipeline for detecting cylindrical pillars of any color in PLY point cloud files.

This package provides a complete modular implementation of the multi-color pillar detection
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
from .ply_io import load_ply_file_open3d, save_ply_file_open3d
from .clustering import cluster_red_points
from .visualization import create_visualization_output
