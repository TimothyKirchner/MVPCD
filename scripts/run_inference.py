# ~/Desktop/MVPCD/scripts/run_inference.py

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
from ultralytics import YOLO

def load_config(config_path='config/config.yaml'):
    config_path = os.path.join(project_root, config_path)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def run_inference(config):
    # Load the trained YOLOv8 model
    model_path = os.path.join(project_root, 'runs', 'detect', 'mvpcd_yolov86', 'weights', 'best.pt')
    if not os.path.exists(model_path):
        print(f"Trained model not found at {model_path}. Please ensure the model has been trained.")
        return

    model = YOLO(model_path)

    # Initialize the ZED camera
    camera = initialize_camera(config)
    if camera is None:
        print("Failed to initialize the camera.")
        return

    try:
        print("Starting live inference. Press 'q' to exit.")
        while True:
            image, _ = capture_frame(camera)
            if image is None:
                continue

            # Run inference
            results = model(image, verbose=False)

            # Annotate the image with detections
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting inference.")
                break

    finally:
        camera.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    config = load_config()
    run_inference(config)
