# ~/Desktop/MVPCD/scripts/capture.py

import sys
import os
import cv2
import yaml
import numpy as np
from datetime import datetime

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

def capture_images(config):
    image_dir = os.path.join(project_root, config['output']['image_dir'])
    depth_dir = os.path.join(project_root, config['output']['depth_dir'])

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(depth_dir):
        os.makedirs(depth_dir)

    camera = initialize_camera(config)
    num_images = config['capture']['num_images']
    interval = config['capture']['interval']

    for i in range(num_images):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image, depth = capture_frame(camera)

        if image is None or depth is None:
            print("Failed to capture image or depth map.")
            continue

        image_filename = os.path.join(image_dir, f"image_{timestamp}.png")
        depth_filename = os.path.join(depth_dir, f"depth_{timestamp}.npy")

        cv2.imwrite(image_filename, image)
        np.save(depth_filename, depth)
        np.save("processed_" + depth_filename, depth)

        print(f"Captured image {i+1}/{num_images}: {image_filename} and depth map {depth_filename}")

        cv2.imshow("Captured Image", image)
        cv2.waitKey(int(interval * 1000))

    camera.close()
    cv2.destroyAllWindows()
    print("Image capture completed.")

if __name__ == "__main__":
    config = load_config()
    capture_images(config)
