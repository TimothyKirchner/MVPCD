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
from pathlib import Path

def find_most_recent_folder(directory):
    # Use list comprehension to get folders in the directory with creation times
    folders = [(folder, os.path.getctime(folder)) for folder in Path(directory).iterdir() if folder.is_dir()]
    
    # Sort folders by creation time in descending order and get the first one
    most_recent_folder = max(folders, key=lambda x: x[1])[0] if folders else None
    
    return most_recent_folder

def load_config(config_path='config/config.yaml'):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, config_path)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def test():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Corrected model path
    modeldir = os.path.join(project_root, 'runs', 'detect')
    model_path = os.path.join(project_root, "runs", "detect", find_most_recent_folder(modeldir), "weights", "best.pt")
    print(model_path)

def run_inference(config):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Corrected model path
    modeldir = os.path.join(project_root, 'runs', 'detect')
    # model_path = os.path.join(project_root, "runs", "detect", find_most_recent_folder(modeldir), "weights", "best.pt")
    model_path = os.path.join(project_root, "runs", "detect", "mvpcd_yolov8", "weights", "best.pt")

    print("model path: ", model_path)
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
            results = model(image, conf=0.15, iou=0.65, verbose=False)
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
