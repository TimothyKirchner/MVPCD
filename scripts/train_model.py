# scripts/train_model.py
import sys
import os
import yaml
from ultralytics import YOLO

def load_config(config_path='config/config.yaml'):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, config_path)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_yolo_model(config, epochs=50, learning_rate=0.001, batch_size=16):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_yaml_path = os.path.join(project_root, 'data', 'mvpcd.yaml')

    # Initialize YOLOv8 model (e.g., yolov8n.pt for nano)
    model = YOLO('yolov8s.pt')

    print(f"Using data configuration: {data_yaml_path}")
    project_path = os.path.join(project_root, 'runs', 'detect')
    print("Model will be saved to:", project_path)

    # Start training with built-in augmentations
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        lr0=learning_rate,
        name='mvpcd_yolov8',
        project=project_path,
        augment=True  # Enable built-in augmentations
    )

    print("YOLOv8 model training completed.")

if __name__ == "__main__":
    config = load_config()
    if len(sys.argv) != 4:
        print("Usage: python train_model.py [epochs] [learning_rate] [batch_size]")
        sys.exit(1)
    try:
        epochs = int(sys.argv[1])
        learning_rate = float(sys.argv[2])
        batch_size = int(sys.argv[3])
    except ValueError:
        print("Invalid training parameters. Please ensure epochs and batch_size are integers and learning_rate is a float.")
        sys.exit(1)
    train_yolo_model(config, epochs, learning_rate, batch_size)
