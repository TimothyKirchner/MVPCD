# scripts/set_roi.py
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import cv2
import yaml
from utils.camera_utils import initialize_camera, capture_frame

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

def set_rois(config, class_name, angle_index):
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
        if 'rois' not in config:
            config['rois'] = {}
        if class_name not in config['rois']:
            config['rois'][class_name] = {}
        config['rois'][class_name][angle_index] = rois
        save_config(config)
        print(f"{len(rois)} ROI(s) saved in config/config.yaml for class '{class_name}', angle {angle_index}")
    else:
        print("No ROIs defined.")

if __name__ == "__main__":
    config = load_config()
    if len(sys.argv) < 3:
        print("Usage: python set_rois.py [classname] [angle_index]")
        sys.exit(1)
    class_name = sys.argv[1]
    angle_index = int(sys.argv[2])
    set_rois(config, class_name, angle_index)