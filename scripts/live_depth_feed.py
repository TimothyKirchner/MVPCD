# ~/Desktop/MVPCD/scripts/live_depth_feed.py

import sys
import os
import cv2
import numpy as np
import yaml

# Adjust sys.path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.camera_utils import initialize_camera, capture_frame
from utils.depth_processing import process_depth

def load_config(config_path='config/config.yaml'):
    config_path = os.path.join(project_root, config_path)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(config, config_path='config/config.yaml'):
    config_path = os.path.join(project_root, config_path)
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

def live_depth_feed(config):
    camera = initialize_camera(config)

    # Initial depth threshold values from config
    min_depth = config.get('depth_threshold', {}).get('min', 500)
    max_depth = config.get('depth_threshold', {}).get('max', 2000)

    cv2.namedWindow("Live Depth Feed")
    cv2.createTrackbar("Min Depth", "Live Depth Feed", min_depth, 10000, lambda x: None)
    cv2.createTrackbar("Max Depth", "Live Depth Feed", max_depth, 10000, lambda x: None)

    try:
        while True:
            _, depth_map = capture_frame(camera)
            if depth_map is None:
                continue

            # Update thresholds from trackbars
            min_depth = cv2.getTrackbarPos("Min Depth", "Live Depth Feed")
            max_depth = cv2.getTrackbarPos("Max Depth", "Live Depth Feed")
            if min_depth >= max_depth:
                max_depth = min_depth + 1

            depth_mask = process_depth(depth_map, min_depth, max_depth)

            depth_display = cv2.normalize(depth_mask, None, 0, 255, cv2.NORM_MINMAX)
            depth_display = depth_display.astype(np.uint8)
            depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

            cv2.imshow("Live Depth Feed", depth_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # Save the current thresholds to config
                config['depth_threshold'] = {'min': min_depth, 'max': max_depth}
                save_config(config)
                print("Depth thresholds saved.")
                break

    finally:
        camera.close()
        cv2.destroyAllWindows()
        print("Live Depth Viewer exited.")

if __name__ == "__main__":
    config = load_config()
    live_depth_feed(config)
