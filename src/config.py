"""
Configuration parameters for the red pillar detection pipeline.

This module contains all global configuration constants used throughout
the pillar detection pipeline, including file paths, color segmentation
parameters, clustering settings, geometric constraints, and visualization
parameters.
"""

# =============================================================================
# INPUT/OUTPUT CONFIGURATION
# =============================================================================

# Input/Output Configuration
INPUT_PLY_PATH = "ply/Object_Cloud.ply"
OUTPUT_PLY_PATH = "output/output_pillars.ply"

# =============================================================================
# HSV COLOR SEGMENTATION PARAMETERS
# =============================================================================

# HSV Color Segmentation Parameters (RELAXED for more detection)
HSV_RED_H_RANGES = [(0, 10), (170, 180)]  # Expanded hue ranges for red color
HSV_RED_S_MIN = 0.35                       # Relaxed minimum saturation (0-1)
# Relaxed minimum value/brightness (0-1)
HSV_RED_V_MIN = 0.25

# =============================================================================
# DBSCAN CLUSTERING PARAMETERS
# =============================================================================

# DBSCAN Clustering Parameters (RELAXED for more clusters)
# Relaxed maximum distance between points in cluster (meters)
DBSCAN_EPS = 1.0
# Relaxed minimum points to form a cluster
DBSCAN_MIN_SAMPLES = 50

# =============================================================================
# CYLINDER GEOMETRIC CONSTRAINTS
# =============================================================================

# Cylinder Geometric Constraints (RELAXED for more pillar types)
# Relaxed minimum pillar radius (meters)
PILLAR_RADIUS_MIN = 0.05
# Relaxed maximum pillar radius (meters)
PILLAR_RADIUS_MAX = 5.0
# Relaxed minimum pillar height (meters)
PILLAR_HEIGHT_MIN = 0.5
# Relaxed maximum angle between pillar axis and Z-axis (degrees)
PILLAR_AXIS_Z_ANGLE_MAX = 45

# =============================================================================
# RANSAC PARAMETERS
# =============================================================================

# RANSAC Parameters (RELAXED for better cylinder fitting)
# Relaxed distance threshold for inlier classification
RANSAC_THRESHOLD = 0.2
RANSAC_MAX_ITERATIONS = 10000             # Increased maximum RANSAC iterations

# =============================================================================
# VISUALIZATION PARAMETERS
# =============================================================================

# Visualization Parameters
GRAY_COLOR = (128, 128, 128)             # Gray color for original points
RED_COLOR = (255, 0, 0)                  # Red color for detected pillars
# Points per unit length for cylinder sampling
CYLINDER_SAMPLE_DENSITY = 50

# =============================================================================
# PROCESSING PARAMETERS
# =============================================================================

# Processing Parameters
# Limit for cylinder fitting to prevent memory issues
MAX_POINTS_PER_CLUSTER = 50000
PROGRESS_REPORT_INTERVAL = 100000        # Report progress every N points
