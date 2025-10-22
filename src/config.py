"""
Configuration parameters for the multi-color pillar detection pipeline.

This module contains all global configuration constants used throughout
the pillar detection pipeline, including file paths, color segmentation
parameters, clustering settings, geometric constraints, and visualization
parameters.
"""

# =============================================================================
# INPUT/OUTPUT CONFIGURATION
# =============================================================================

# Input/Output Configuration
INPUT_PLY_PATH = "ply/object_blue.ply"
OUTPUT_PLY_PATH = "output/output_pillars.ply"

# Intermediate Results Configuration
# Enable/disable saving intermediate results
ENABLE_INTERMEDIATE_SAVES = True
# Path for downsampled point cloud
DOWNSAMPLED_PLY_PATH = "output/downsampled_points.ply"
# Path for clustering visualization
CLUSTERED_PLY_PATH = "output/clustered_points.ply"
# Path for colored points only (filtered color regions) - dynamically named based on mode


def get_colored_points_path():
    return f"output/{COLOR_DETECTION_MODE.lower()}_points_only.ply"


# =============================================================================
# COLOR DETECTION MODE AND PARAMETERS
# =============================================================================
# Color Detection Mode: 'red', 'blue', 'green'
COLOR_DETECTION_MODE = 'blue'

# HSV Color Segmentation Parameters for different colors
# Red color ranges
HSV_RED_H_RANGES = [(0, 10), (350, 360)]  # Hue ranges for red color
HSV_RED_S_MIN = 0.55                      # Minimum saturation (0-1)
HSV_RED_V_MIN = 0.45                      # Minimum value/brightness (0-1)

# Blue color ranges (expanded for better detection)
HSV_BLUE_H_RANGES = [(220, 250)]          # Expanded hue range for blue color
HSV_BLUE_S_MIN = 0.80                     # Reduced minimum saturation (0-1)
# Reduced minimum value/brightness (0-1)
HSV_BLUE_V_MIN = 0.25

# Green color ranges
HSV_GREEN_H_RANGES = [(80, 160)]          # Hue range for green color
HSV_GREEN_S_MIN = 0.40                    # Minimum saturation (0-1)
HSV_GREEN_V_MIN = 0.35                    # Minimum value/brightness (0-1)

# =============================================================================
# DBSCAN CLUSTERING PARAMETERS
# =============================================================================

# DBSCAN Clustering Parameters (TIGHTER for more precise clusters)
# Maximum distance between points in cluster (meters)
DBSCAN_EPS = 0.3
DBSCAN_MIN_SAMPLES = 50                    # Minimum points to form a cluster

# =============================================================================
# CYLINDER GEOMETRIC CONSTRAINTS
# =============================================================================

# Cylinder Geometric Constraints (RELAXED for more pillar types)
# Relaxed minimum pillar radius (meters)
PILLAR_RADIUS_MIN = 0.01
# Relaxed maximum pillar radius (meters)
PILLAR_RADIUS_MAX = 100.0
# Relaxed minimum pillar height (meters)
PILLAR_HEIGHT_MIN = 0.1

# =============================================================================
# VISUALIZATION PARAMETERS
# =============================================================================

# Visualization Parameters
GRAY_COLOR = (128, 128, 128)              # Gray color for original points
RED_COLOR = (255, 0, 0)                   # Red color for detected pillars
# Points per unit length for cylinder sampling
CYLINDER_SAMPLE_DENSITY = 50

# =============================================================================
# DOWNSAMPLING PARAMETERS
# =============================================================================

# Downsampling Parameters (Open3D Native Implementation)
DOWNSAMPLING_ENABLED = True                # Enable/disable downsampling step
# 'voxel', 'random', 'uniform', 'farthest_point'
DOWNSAMPLING_METHOD = 'voxel'
# Voxel size for voxel grid method (meters)
DOWNSAMPLING_VOXEL_SIZE = 0.01
# Target ratio for random sampling (0.1 = 10%)
DOWNSAMPLING_TARGET_RATIO = 0.1
# Keep every k-th point for uniform method
DOWNSAMPLING_UNIFORM_K = 10
# Number of points to keep for farthest point method
DOWNSAMPLING_FARTHEST_POINTS = 100000

# =============================================================================
# PCA ANALYSIS PARAMETERS
# =============================================================================

# PCA Analysis Parameters
# Minimum eigenvalue ratio for primary axis to be considered cylindrical
PCA_CYLINDER_THRESHOLD = 0.6
# Maximum difference between 2nd and 3rd eigenvalue ratios for cylindrical detection
PCA_CROSS_SECTION_RATIO_THRESHOLD = 0.3
# Minimum variance required for secondary axes (to avoid degenerate cases)
PCA_MIN_SECONDARY_VARIANCE = 0.05
# Maximum allowed variance for tertiary axis in cylindrical structures
PCA_MAX_TERTIARY_VARIANCE = 0.4

# =============================================================================
# PROCESSING PARAMETERS
# =============================================================================

# Processing Parameters
# Limit for cylinder fitting to prevent memory issues
MAX_POINTS_PER_CLUSTER = 50000
# Number of top clusters (by point count) to analyze with PCA
TOP_CLUSTERS_TO_ANALYZE = 5
