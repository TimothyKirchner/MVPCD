# scripts/run_inference.py
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import cv2
import numpy as np
import yaml
from utils.camera_utils import initialize_camera, capture_frame
from ultralytics import YOLO

def load_config(config_path='config/config.yaml'):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, config_path)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def run_inference(config):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Corrected model path
    model_path = os.path.join(project_root, 'runs', 'detect', 'mvpcd_yolov82', 'weights', 'best.pt')
    if not os.path.exists(model_path):
        print(f"Trained model not found at {model_path}. Please ensure the model has been trained.")
        return

    model = YOLO(model_path)

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

            # Adjust confidence and IoU thresholds as needed
            results = model(image, conf=0.25, iou=0.45, verbose=False)
            annotated_frame = results[0].plot()

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
