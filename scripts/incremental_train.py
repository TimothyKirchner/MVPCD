# scripts/incremental_train.py
import sys
import os
import yaml
from ultralytics import YOLO
import torch
from torch import nn
from copy import deepcopy

# Adjust sys.path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_config(config_path='config/config.yaml'):
    """Load the YAML configuration file."""
    config_full_path = os.path.join(project_root, config_path)
    with open(config_full_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def incremental_train_yolo_model(
    config,
    base_model_path,
    epochs=50,
    learning_rate=0.0001,
    batch_size=8,
    classes_to_remove=[]
):
    """
    Incrementally train the YOLOv8 model with new classes and remove specified classes,
    using a simplified knowledge distillation approach by freezing existing layers.
    """
    # Ensure that mvpcd.yaml exists
    mvpcd_yaml_path = os.path.join(project_root, 'data', 'mvpcd.yaml')
    if not os.path.exists(mvpcd_yaml_path):
        print(f"Configuration file '{mvpcd_yaml_path}' does not exist.")
        return

    # Load the existing model (teacher model)
    teacher_model = YOLO(base_model_path)
    teacher_model.eval()

    # Get existing class names from the teacher model
    existing_class_names = teacher_model.names.copy()

    print("The classes in the Old model are: ",existing_class_names)

    # Remove specified classes
    for class_name in classes_to_remove:
        if class_name in existing_class_names:
            idx = existing_class_names.index(class_name)
            existing_class_names.pop(idx)
            print(f"Class '{class_name}' removed from the model.")
        else:
            print(f"Class '{class_name}' not found in the model. Skipping removal.")

    # Add new classes from config
    new_class_names = config.get('class_names', [])
    for class_name in new_class_names:
        if class_name not in existing_class_names:
            existing_class_names.append(class_name)
            print(f"Class '{class_name}' added to the model.")

    # Update mvpcd.yaml with the new class names
    update_mvpcd_yaml(existing_class_names)

    # Initialize a new YOLO model with the updated number of classes
    student_model = YOLO('yolov8s.pt')  # You can choose other variants like 'yolov8m.pt', etc.
    student_model.overrides['nc'] = len(existing_class_names)
    student_model.overrides['names'] = existing_class_names

    # Load weights from the teacher model
    student_model.load_weights(base_model_path)

    # Freeze all layers except the detection head
    for name, param in student_model.model.named_parameters():
        if 'model.24' not in name:  # Adjust based on actual layer names; 'model.24' typically refers to the detection head
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Adjust the detection head to match the new number of classes
    num_classes = len(existing_class_names)
    student_model.model.model[-1] = nn.Conv2d(
        in_channels=student_model.model.model[-1].in_channels,
        out_channels=num_classes * 5,  # 5 = 4 bbox coords + 1 objectness score
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0)
    )
    student_model.model.model[-1].to(student_model.device)

    # Define optimizer
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, student_model.model.parameters()),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=0.0005
    )

    # Define loss functions
    ce_loss_fn = nn.CrossEntropyLoss()

    # Load training data
    print("Loading training data...")
    student_model.overrides['batch'] = batch_size
    student_model.overrides['epochs'] = epochs
    student_model.overrides['lr0'] = learning_rate
    student_model.overrides['data'] = mvpcd_yaml_path
    student_model.overrides['imgsz'] = 640

    # Start training
    print("Starting incremental training...")
    student_model.train()

    # Save the trained model
    save_dir = os.path.join(project_root, 'runs', 'detect', 'mvpcd_incremental_yolov8')
    os.makedirs(save_dir, exist_ok=True)
    weights_path = os.path.join(save_dir, 'best.pt')
    student_model.save(weights_path)
    print(f"Incrementally trained model saved to '{weights_path}'")

def update_mvpcd_yaml(class_names):
    """Update the mvpcd.yaml file with current class names."""
    mvpcd_yaml_path = os.path.join(project_root, 'data', 'mvpcd.yaml')
    with open(mvpcd_yaml_path, 'w') as file:
        yaml.dump({
            'train': {
                'images': './data/images/train',
                'labels': './data/labels/train'
            },
            'val': {
                'images': './data/images/val',
                'labels': './data/labels/val'
            },
            'nc': len(class_names),
            'names': class_names
        }, file)
    print("Updated 'mvpcd.yaml' with current class names.")

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python incremental_train.py [base_model_path] [epochs] [learning_rate] [batch_size] [classes_to_remove(comma-separated)]")
    else:
        base_model_path = sys.argv[1]
        epochs = int(sys.argv[2]) if sys.argv[2].isdigit() else 50
        learning_rate = float(sys.argv[3]) if sys.argv[3].replace('.', '', 1).isdigit() else 0.0001
        batch_size = int(sys.argv[4]) if sys.argv[4].isdigit() else 8
        classes_to_remove = sys.argv[5].split(',') if sys.argv[5] else []
        config = load_config()
        incremental_train_yolo_model(
            config,
            base_model_path=base_model_path,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            classes_to_remove=classes_to_remove
        )
