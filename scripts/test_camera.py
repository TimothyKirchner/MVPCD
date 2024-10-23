# scripts/test_camera.py
import cv2
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.camera_utils import initialize_camera, capture_frame

def test_camera(config):
    camera = initialize_camera(config)
    if camera is None:
        print("Failed to initialize the camera.")
        return

    try:
        print("Testing camera. Press 'q' to exit.")
        while True:
            image, _ = capture_frame(camera)
            if image is None:
                continue

            cv2.imshow("Camera Test", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting camera test.")
                break

    finally:
        camera.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import yaml
    config_path = 'config/config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    test_camera(config)
