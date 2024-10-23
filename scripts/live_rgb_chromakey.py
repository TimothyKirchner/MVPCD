# scripts/live_rgb_chromakey.py
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import cv2
import numpy as np
import yaml
from utils.camera_utils import initialize_camera, capture_frame
from utils.chroma_key import apply_chroma_key

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

def live_rgb_chromakey(config):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    camera = initialize_camera(config)

    lower_h, lower_s, lower_v = config.get('chroma_key', {}).get('lower_color', [0, 0, 0])
    upper_h, upper_s, upper_v = config.get('chroma_key', {}).get('upper_color', [179, 255, 255])

    cv2.namedWindow("Live RGB Chroma-Keying")
    cv2.createTrackbar('Lower H', 'Live RGB Chroma-Keying', lower_h, 179, lambda x: None)
    cv2.createTrackbar('Lower S', 'Live RGB Chroma-Keying', lower_s, 255, lambda x: None)
    cv2.createTrackbar('Lower V', 'Live RGB Chroma-Keying', lower_v, 255, lambda x: None)
    cv2.createTrackbar('Upper H', 'Live RGB Chroma-Keying', upper_h, 179, lambda x: None)
    cv2.createTrackbar('Upper S', 'Live RGB Chroma-Keying', upper_s, 255, lambda x: None)
    cv2.createTrackbar('Upper V', 'Live RGB Chroma-Keying', upper_v, 255, lambda x: None)

    try:
        while True:
            image, _ = capture_frame(camera)
            if image is None:
                continue

            lower_h = cv2.getTrackbarPos('Lower H', 'Live RGB Chroma-Keying')
            lower_s = cv2.getTrackbarPos('Lower S', 'Live RGB Chroma-Keying')
            lower_v = cv2.getTrackbarPos('Lower V', 'Live RGB Chroma-Keying')
            upper_h = cv2.getTrackbarPos('Upper H', 'Live RGB Chroma-Keying')
            upper_s = cv2.getTrackbarPos('Upper S', 'Live RGB Chroma-Keying')
            upper_v = cv2.getTrackbarPos('Upper V', 'Live RGB Chroma-Keying')

            if lower_h >= upper_h:
                upper_h = lower_h + 1
            if lower_s >= upper_s:
                upper_s = lower_s + 1
            if lower_v >= upper_v:
                upper_v = lower_v + 1

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_bound = np.array([lower_h, lower_s, lower_v])
            upper_bound = np.array([upper_h, upper_s, upper_v])

            chroma_keyed = apply_chroma_key(hsv, lower_bound, upper_bound)

            cv2.imshow("Live RGB Chroma-Keying", chroma_keyed)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                config['chroma_key']['lower_color'] = [int(lower_h), int(lower_s), int(lower_v)]
                config['chroma_key']['upper_color'] = [int(upper_h), int(upper_s), int(upper_v)]
                save_config(config)
                print("Chroma key settings saved.")
                break

    finally:
        camera.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    config = load_config()
    live_rgb_chromakey(config)
