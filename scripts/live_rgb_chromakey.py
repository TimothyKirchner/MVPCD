# scripts/live_rgb_chromakey.py
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

def live_rgb_chromakey(config, class_name, angle_index):
    camera = initialize_camera(config)
    chroma_key_settings = config.get('chroma_key_settings', {})

    # Initialize default chroma key values if not set
    if 'chroma_key_settings' not in config:
        config['chroma_key_settings'] = {}
    if class_name not in config['chroma_key_settings']:
        config['chroma_key_settings'][class_name] = {}
    if angle_index not in config['chroma_key_settings'][class_name]:
        config['chroma_key_settings'][class_name][angle_index] = {
            'lower_color': [0, 0, 0],
            'upper_color': [179, 255, 255]
        }

    lower_h, lower_s, lower_v = config['chroma_key_settings'][class_name][angle_index]['lower_color']
    upper_h, upper_s, upper_v = config['chroma_key_settings'][class_name][angle_index]['upper_color']

    cv2.namedWindow("Live RGB Chroma-Keying")
    cv2.createTrackbar('Lower H', 'Live RGB Chroma-Keying', lower_h, 179, lambda x: None)
    cv2.createTrackbar('Lower S', 'Live RGB Chroma-Keying', lower_s, 255, lambda x: None)
    cv2.createTrackbar('Lower V', 'Live RGB Chroma-Keying', lower_v, 255, lambda x: None)
    cv2.createTrackbar('Upper H', 'Live RGB Chroma-Keying', upper_h, 179, lambda x: None)
    cv2.createTrackbar('Upper S', 'Live RGB Chroma-Keying', upper_s, 255, lambda x: None)
    cv2.createTrackbar('Upper V', 'Live RGB Chroma-Keying', upper_v, 255, lambda x: None)
    print(f"Adjusting chroma key for class: {class_name}, angle {angle_index}")

    last_restart_time = time.time()  # Initialize timer

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
                # Save chroma key settings for this class and angle
                config['chroma_key_settings'][class_name][angle_index]['lower_color'] = [int(lower_h), int(lower_s), int(lower_v)]
                config['chroma_key_settings'][class_name][angle_index]['upper_color'] = [int(upper_h), int(upper_s), int(upper_v)]
                save_config(config)
                print("Chroma key settings saved.")
                break
            elif key == ord('r'):
                # Reinitialize the camera and viewer
                print("Reinitializing the camera and viewer...")
                try:
                    camera.release()
                    camera = initialize_camera(config)
                    if camera is None:
                        print("Camera re-initialization failed.")
                        break
                    # Reinitialize the OpenCV window and trackbars
                    cv2.destroyAllWindows()
                    cv2.namedWindow("Live RGB Chroma-Keying")
                    cv2.createTrackbar('Lower H', 'Live RGB Chroma-Keying', lower_h, 179, lambda x: None)
                    cv2.createTrackbar('Lower S', 'Live RGB Chroma-Keying', lower_s, 255, lambda x: None)
                    cv2.createTrackbar('Lower V', 'Live RGB Chroma-Keying', lower_v, 255, lambda x: None)
                    cv2.createTrackbar('Upper H', 'Live RGB Chroma-Keying', upper_h, 179, lambda x: None)
                    cv2.createTrackbar('Upper S', 'Live RGB Chroma-Keying', upper_s, 255, lambda x: None)
                    cv2.createTrackbar('Upper V', 'Live RGB Chroma-Keying', upper_v, 255, lambda x: None)
                except Exception as e:
                    print(f"Error during reinitialization: {e}")
                    break
                last_restart_time = time.time()  # Reset timer
            elif key == ord('v'):
                # Close and reopen the OpenCV window
                print("Closing and reopening the OpenCV window...")
                cv2.destroyWindow("Live RGB Chroma-Keying")
                cv2.namedWindow("Live RGB Chroma-Keying")
                cv2.createTrackbar('Lower H', 'Live RGB Chroma-Keying', lower_h, 179, lambda x: None)
                cv2.createTrackbar('Lower S', 'Live RGB Chroma-Keying', lower_s, 255, lambda x: None)
                cv2.createTrackbar('Lower V', 'Live RGB Chroma-Keying', lower_v, 255, lambda x: None)
                cv2.createTrackbar('Upper H', 'Live RGB Chroma-Keying', upper_h, 179, lambda x: None)
                cv2.createTrackbar('Upper S', 'Live RGB Chroma-Keying', upper_s, 255, lambda x: None)
                cv2.createTrackbar('Upper V', 'Live RGB Chroma-Keying', upper_v, 255, lambda x: None)
                last_restart_time = time.time()  # Reset timer

            # Automatically restart the window every 2 minutes
            current_time = time.time()
            if current_time - last_restart_time >= 120:
                print("Automatically restarting the OpenCV window after 2 minutes...")
                cv2.destroyWindow("Live RGB Chroma-Keying")
                cv2.namedWindow("Live RGB Chroma-Keying")
                cv2.createTrackbar('Lower H', 'Live RGB Chroma-Keying', lower_h, 179, lambda x: None)
                cv2.createTrackbar('Lower S', 'Live RGB Chroma-Keying', lower_s, 255, lambda x: None)
                cv2.createTrackbar('Lower V', 'Live RGB Chroma-Keying', lower_v, 255, lambda x: None)
                cv2.createTrackbar('Upper H', 'Live RGB Chroma-Keying', upper_h, 179, lambda x: None)
                cv2.createTrackbar('Upper S', 'Live RGB Chroma-Keying', upper_s, 255, lambda x: None)
                cv2.createTrackbar('Upper V', 'Live RGB Chroma-Keying', upper_v, 255, lambda x: None)
                last_restart_time = current_time  # Reset timer

    finally:
        camera.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    config = load_config()
    if len(sys.argv) < 3:
        print("Usage: python live_rgb_chromakey.py [classname] [angle_index]")
        sys.exit(1)
    class_name = sys.argv[1]
    angle_index = int(sys.argv[2])
    live_rgb_chromakey(config, class_name, angle_index)