# scripts/live_depth_feed.py
import sys
import os
import time  # Added for timer functionality
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import cv2
import numpy as np
import yaml
from utils.camera_utils import initialize_camera, capture_frame
from utils.depth_processing import process_depth

def load_config(config_path='config/config.yaml'):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, config_path)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(config, config_path='config/config.yaml'):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, config_path)
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

def live_depth_feed(config, class_name, angle_index):
    camera = initialize_camera(config)
    
    if camera is None:
        print("Camera initialization failed.")
        return

    # Initialize depth thresholds if not set
    if 'depth_thresholds' not in config:
        config['depth_thresholds'] = {}
    if class_name not in config['depth_thresholds']:
        config['depth_thresholds'][class_name] = {}
    if angle_index not in config['depth_thresholds'][class_name]:
        config['depth_thresholds'][class_name][angle_index] = {'min': 500, 'max': 1000}

    min_depth = config['depth_thresholds'][class_name][angle_index]['min']
    max_depth = config['depth_thresholds'][class_name][angle_index]['max']

    cv2.namedWindow("Live Depth Feed")
    cv2.createTrackbar("Min Depth", "Live Depth Feed", min_depth, 1000, lambda x: None)
    cv2.createTrackbar("Max Depth", "Live Depth Feed", max_depth, 1000, lambda x: None)

    print(f"\n--- Adjusting Depth Thresholds for Class '{class_name}', angle {angle_index} ---")
    print(f"Initial Min Depth: {min_depth} mm, Max Depth: {max_depth} mm\n")

    last_restart_time = time.time()  # Initialize timer

    try:
        while True:
            _, depth_map = capture_frame(camera)
            if depth_map is None:
                continue

            # Display current depth thresholds
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
                # Save the updated depth thresholds for the specific class and angle
                config['depth_thresholds'][class_name][angle_index] = {'min': min_depth, 'max': max_depth}
                save_config(config)
                print(f"Depth thresholds for class '{class_name}', angle {angle_index} saved: Min={min_depth} mm, Max={max_depth} mm")
                break
            elif key == ord('r'):
                # Reinitialize the camera and viewer
                print("Reinitializing the camera and viewer...")
                try:
                    camera.close()
                    camera = initialize_camera(config)
                    if camera is None:
                        print("Camera re-initialization failed.")
                        break
                    # Reinitialize the OpenCV window and trackbars
                    cv2.destroyAllWindows()
                    cv2.namedWindow("Live Depth Feed")
                    cv2.createTrackbar("Min Depth", "Live Depth Feed", min_depth, 1000, lambda x: None)
                    cv2.createTrackbar("Max Depth", "Live Depth Feed", max_depth, 1000, lambda x: None)
                except Exception as e:
                    print(f"Error during reinitialization: {e}")
                    break
                last_restart_time = time.time()  # Reset timer
            elif key == ord('v'):
                # Close and reopen the OpenCV window
                print("Closing and reopening the OpenCV window...")
                cv2.destroyWindow("Live Depth Feed")
                cv2.namedWindow("Live Depth Feed")
                cv2.createTrackbar("Min Depth", "Live Depth Feed", min_depth, 1000, lambda x: None)
                cv2.createTrackbar("Max Depth", "Live Depth Feed", max_depth, 1000, lambda x: None)
                last_restart_time = time.time()  # Reset timer

            # Automatically restart the window every 2 minutes
            current_time = time.time()
            if current_time - last_restart_time >= 120:
                print("Automatically restarting the OpenCV window after 2 minutes...")
                cv2.destroyWindow("Live Depth Feed")
                cv2.namedWindow("Live Depth Feed")
                cv2.createTrackbar("Min Depth", "Live Depth Feed", min_depth, 1000, lambda x: None)
                cv2.createTrackbar("Max Depth", "Live Depth Feed", max_depth, 1000, lambda x: None)
                last_restart_time = current_time  # Reset timer

    finally:
        camera.close()
        cv2.destroyAllWindows()
        print("Live Depth Viewer exited.")

if __name__ == "__main__":
    config = load_config()
    if len(sys.argv) < 3:
        print("Usage: python live_depth_feed.py [classname] [angle_index]")
        sys.exit(1)
    class_name = sys.argv[1]
    angle_index = int(sys.argv[2])
    live_depth_feed(config, class_name, angle_index)