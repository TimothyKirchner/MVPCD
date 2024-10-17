# ~/Desktop/MVPCD/scripts/set_roi.py

import sys
import os
import cv2
import yaml

# Adjust sys.path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.camera_utils import initialize_camera, capture_frame

def load_config(config_path='config/config.yaml'):
    config_path = os.path.join(project_root, config_path)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(config, config_path='config/config.yaml'):
    config_path = os.path.join(project_root, config_path)
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

def set_rois(config):
    camera = initialize_camera(config)
    image, _ = capture_frame(camera)
    camera.close()

    if image is None:
        print("Could not capture image from camera.")
        return

    rois = []
    while True:
        roi = cv2.selectROI("Select ROI (Press Enter when done, Esc to cancel)", image, False, False)
        if roi == (0, 0, 0, 0):
            break
        rois.append({'x': int(roi[0]), 'y': int(roi[1]), 'width': int(roi[2]), 'height': int(roi[3])})
        print(f"ROI added: {rois[-1]}")

        print("Press 'q' to finish selecting ROIs, or any other key to select another ROI.")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    if rois:
        config['rois'] = rois
        save_config(config)
        print(f"{len(rois)} ROI(s) saved in config/config.yaml")
    else:
        print("No ROIs defined.")

if __name__ == "__main__":
    config = load_config()
    set_rois(config)
