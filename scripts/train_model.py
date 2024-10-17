# ~/Desktop/MVPCD/scripts/train_model.py

import sys
import os
import yaml
from ultralytics import YOLO

# Adjust sys.path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_config(config_path='config/config.yaml'):
    config_path = os.path.join(project_root, config_path)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_yolo_model(config, epochs=50, learning_rate=0.001, batch_size=16):
    data_yaml_path = os.path.join(project_root, 'data', 'mvpcd.yaml')

    model = YOLO('yolov8n.pt')
    project_path = os.path.join(project_root, 'runs', 'detect')
    print("Model will be saved to:", project_path)
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        lr0=learning_rate,
        name='mvpcd_yolov8',
        project=project_path
    )

    print("YOLOv8 model training completed.")

if __name__ == "__main__":
    config = load_config()
    train_yolo_model(config)
