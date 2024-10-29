# scripts/train_model.py
import sys
import os
import yaml
from ultralytics import YOLO

# Adjust sys.path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_config(config_path='config/config.yaml'):
    config_full_path = os.path.join(project_root, config_path)
    with open(config_full_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_yolo_model(config, epochs=100, learning_rate=0.0001, batch_size=8):
    """
    Train the YOLOv8 model with specified parameters.
    """
    # Ensure that mvpcd.yaml exists
    mvpcd_yaml_path = os.path.join(project_root, 'data', 'mvpcd.yaml')
    if not os.path.exists(mvpcd_yaml_path):
        print(f"Configuration file {mvpcd_yaml_path} does not exist.")
        return

    # Initialize the YOLO model
    model = YOLO('yolov8s.pt')  # You can choose other variants

    # Train the model
    model.train(
        data=mvpcd_yaml_path,
        epochs=epochs,
        lr0=learning_rate,
        batch=batch_size,
        imgsz=640,  # Reduced image size for faster training
        name='mvpcd_yolov8',
        project=os.path.join(project_root, 'runs', 'detect'),
        verbose=True,
        augment=True,    # Enable data augmentation
        pretrained=True, # Ensure transfer learning is utilized
        weight_decay=0.00005,  # Add regularization
        visualize=True,
)
    latest_model_path = model.ckpt_path  # Or use model.last
    print("Path to the last trained model:", latest_model_path)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python train_model.py [epochs] [learning_rate] [batch_size]")
    else:
        epochs = int(sys.argv[1])
        learning_rate = float(sys.argv[2])
        batch_size = int(sys.argv[3])
        config = load_config()
        train_yolo_model(config, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)
