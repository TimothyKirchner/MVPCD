# ~/Desktop/MVPCD/utils/depth_processing.py

import numpy as np
import os

def load_depth_map(depth_path):
    """
    Loads the depth map from a .npy file.
    """
    if os.path.exists(depth_path):
        depth_map = np.load(depth_path)
        return depth_map
    else:
        print(f"Depth map not found: {depth_path}")
        return None

def process_depth(depth_map, min_depth, max_depth):
    """
    Segments the depth map based on the threshold values.
    """
    mask = np.logical_and(depth_map > min_depth, depth_map < max_depth)
    return (mask.astype(np.uint8) * 255)
