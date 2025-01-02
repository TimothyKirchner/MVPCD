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
    """
    Finds the most recent folder among the two most recent folders in the 'detect' and 'segment' subdirectories.

    Parameters:
        directory (str or Path): The main directory containing 'detect' and 'segment' subfolders.

    Returns:
        Path or None: The Path object of the most recent folder or None if no folders are found.
    """
    subfolders = ['detect', 'segment']
    recent_folders = []

    for sub in subfolders:
        sub_dir = Path(directory) / sub
        if not sub_dir.exists() or not sub_dir.is_dir():
            print(f"Subdirectory '{sub_dir}' does not exist or is not a directory. Skipping.")
            continue

        # List all subdirectories with their creation times
        folders = [(folder, os.path.getctime(folder)) for folder in sub_dir.iterdir() if folder.is_dir()]

        if not folders:
            print(f"No subfolders found in '{sub_dir}'.")
            continue

        # Sort folders by creation time in descending order
        sorted_folders = sorted(folders, key=lambda x: x[1], reverse=True)

        # Get the two most recent folders
        top_two = sorted_folders[:2]
        recent_folders.extend([folder for folder, _ in top_two])

    if not recent_folders:
        print("No recent folders found in 'detect' or 'segment' subdirectories.")
        return None

    # Determine the most recent folder among the collected folders
    most_recent_folder = max(recent_folders, key=lambda x: os.path.getctime(x))

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
    # model_path = os.path.join(project_root, "runs", "segment", "mvpcd_yolov8_seg7", "weights", "best.pt")
    model_path = os.path.join(project_root, "runs", "detect", "mvpcd_yolov8_detect2", "weights", "best.pt")

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
            # image = cv2.resize(image, (480,480))

            # Adjust confidence and IoU thresholds as needed
            results = model(image, conf=0.45, verbose=False)
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
