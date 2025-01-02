# scripts/train_model.py

import sys
import os
import argparse
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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train YOLOv8 model for Detection or Segmentation.")
    parser.add_argument(
        '--task',
        type=str,
        choices=['detection', 'segmentation'],
        default='segmentation',  # Default to segmentation
        help="Choose the training task: 'detection' for bounding boxes or 'segmentation' for bounding boxes and masks."
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help="Number of training epochs."
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0001,
        help="Initial learning rate."
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help="Training batch size."
    )
    return parser.parse_args()

def train_yolo_model_masks(config, task, weight_decay, epochs, learning_rate, batch_size):
    """
    Train the YOLOv8 model (detection or segmentation) with specified parameters.
    """
    # Ensure that mvpcd.yaml exists
    mvpcd_yaml_path = os.path.join(project_root, 'data', 'mvpcd.yaml')
    if not os.path.exists(mvpcd_yaml_path):
        print(f"Configuration file {mvpcd_yaml_path} does not exist.")
        return
    
    # task_label = task

    # print("The detection model is currently: ", task_label, ". Is this coorrect?")
    # check = input("Is this correct? (y/n): ")
    # if check == "n":
    #     while task_label != "detection" and task_label != "segment":
    #         print("ERROR: Task Label must be either \"detection\" or \"segment\".")
    #         task = input("Input \"detection\" for bbox or \"segment\" for masks: ")
    #         task_label = task

    # print("Task: ", task)
    # print("Task_Label: ", task_label)

    # # Select the appropriate model based on the task
    # if task == 'detection':
    #     model = YOLO('yolov8s.pt')  # YOLOv8 Small detection model
    #     project_name = 'mvpcd_yolov8_detect'
    #     task_label = 'detect'
    # else:
    #     model = YOLO('yolov8s-seg.pt')  # YOLOv8 Small segmentation model
    #     project_name = 'mvpcd_yolov8_seg'
    #     task_label = 'segment'
    
    project_name = 'mvpcd_yolov8_detect'
    model = YOLO('yolov8m.pt')  # YOLOv8 Small detection model
    task = "detection"
    task_label = 'detect'
    print("Task: ", task)
    print("Task_label: ", task_label)

    # Set up training parameters
    model.train(
        data=mvpcd_yaml_path,
        epochs=epochs,
        lr0=learning_rate,
        batch=batch_size,
        imgsz=640,  # Adjust image size as needed
        name=project_name,
        project=os.path.join(project_root, 'runs', task_label),
        verbose=True,
        augment=True,    # Enable data augmentation
        pretrained=True, # Use pre-trained weights
        weight_decay=weight_decay,  # Adjust regularization if needed
        task=task_label, # Specify the task
        optimizer='AdamW',  # Use AdamW optimizer
        # cls=1.0,                   # Increased classification loss weight
        # box=0.5,                   # Decreased box loss weight
        # conf=0.001,
        workers=8,
        # scale=0.5,
        # translate=0.1,
        # hsv_v=0.4,
        # degrees=0.05,
        # hsv_h=0.015,
        # hsv_s=0.7,
        # shear=0.05,
        # flipud=0.05,
        # fliplr=0.05,vvv
        # perspective=0.0,
        # mosaic=0.1,
        # copy_paste=0.025,
        # mixup=0.05,
        # bgr=0.025,
        # erasing=0.4,
        # crop_fraction=1.0,
    )
    model.train(
        data=mvpcd_yaml_path,
        epochs=epochs,
        lr0=learning_rate,
        batch=batch_size,
        imgsz=640,  # Adjust image size as needed
        name=project_name,
        project=os.path.join(project_root, 'runs', task_label),
        verbose=True,
        augment=True,             # Enable data augmentation
        pretrained=True,          # Use pre-trained weights
        weight_decay=weight_decay,# Adjust regularization if needed
        task=task_label,          # Specify the task
        optimizer='AdamW',        # Use AdamW optimizer
        workers=8,
        # Loss Function Weights
        cls=1.0,                   # Increased classification loss weight
        box=0.5,                   # Decreased box loss weight
        conf=0.001,                # Adjust confidence loss weight
        # Geometric Augmentations
        scale=0.5,                 # Random scaling
        translate=0.1,             # Random translation
        shear=0.05,                # Random shear
        flipud=0.05,               # Random vertical flip
        fliplr=0.05,               # Random horizontal flip
        # Color Space Augmentations
        hsv_h=0.015,               # Hue adjustment
        hsv_s=0.7,                 # Saturation adjustment
        hsv_v=0.4,                 # Value (brightness) adjustment
        bgr=0.025,                 # BGR color augmentation
        # Advanced Augmentations
        mosaic=0.1,                # Mosaic augmentation
        mixup=0.05,                # MixUp augmentation
        erasing=0.4,               # Random erasing
        # Optional: Uncomment if needed
        # copy_paste=0.025,        # Copy-Paste augmentation
        # perspective=0.0,         # Perspective transformation (no change)
        # crop_fraction=1.0,        # No cropping
    )

    # model = YOLO('yolov8s.pt')  # YOLOv8 Small detection model

    # model.train(
    #     data=mvpcd_yaml_path,
    #     epochs=epochs,
    #     lr0=learning_rate,
    #     batch=batch_size,
    #     imgsz=640,  # Adjust image size as needed
    #     name=project_name,
    #     project=os.path.join(project_root, 'runs', task_label),
    #     verbose=True,
    #     augment=True,    # Enable data augmentation
    #     pretrained=True, # Use pre-trained weights
    #     weight_decay=weight_decay,  # Adjust regularization if needed
    #     task=task_label, # Specify the task
    #     optimizer='AdamW',  # Use AdamW optimizer
    #     # cls=1.0,                   # Increased classification loss weight
    #     # box=0.5,                   # Decreased box loss weight
    #     # conf=0.001,
    #     workers=8,
    #     # scale=0.5,
    #     # translate=0.1,
    #     # hsv_v=0.4,
    #     # degrees=0.05,
    #     # hsv_h=0.015,
    #     # hsv_s=0.7,
    #     # shear=0.05,
    #     # flipud=0.05,
    #     # fliplr=0.05,vvv
    #     # perspective=0.0,
    #     # mosaic=0.1,
    #     # copy_paste=0.025,
    #     # mixup=0.05,
    #     # bgr=0.025,
    #     # erasing=0.4,
    #     # crop_fraction=1.0,
    # )
    # model.train(
    #     data=mvpcd_yaml_path,
    #     epochs=epochs,
    #     lr0=learning_rate,
    #     batch=batch_size,
    #     imgsz=640,  # Adjust image size as needed
    #     name=project_name,
    #     project=os.path.join(project_root, 'runs', task_label),
    #     verbose=True,
    #     augment=True,             # Enable data augmentation
    #     pretrained=True,          # Use pre-trained weights
    #     weight_decay=weight_decay,# Adjust regularization if needed
    #     task=task_label,          # Specify the task
    #     optimizer='AdamW',        # Use AdamW optimizer
    #     workers=8,
    #     # Loss Function Weights
    #     cls=1.0,                   # Increased classification loss weight
    #     box=0.5,                   # Decreased box loss weight
    #     conf=0.001,                # Adjust confidence loss weight
    #     # Geometric Augmentations
    #     scale=0.5,                 # Random scaling
    #     translate=0.1,             # Random translation
    #     shear=0.05,                # Random shear
    #     flipud=0.05,               # Random vertical flip
    #     fliplr=0.05,               # Random horizontal flip
    #     # Color Space Augmentations
    #     hsv_h=0.015,               # Hue adjustment
    #     hsv_s=0.7,                 # Saturation adjustment
    #     hsv_v=0.4,                 # Value (brightness) adjustment
    #     bgr=0.025,                 # BGR color augmentation
    #     # Advanced Augmentations
    #     mosaic=0.1,                # Mosaic augmentation
    #     mixup=0.05,                # MixUp augmentation
    #     erasing=0.4,               # Random erasing
    #     # Optional: Uncomment if needed
    #     # copy_paste=0.025,        # Copy-Paste augmentation
    #     # perspective=0.0,         # Perspective transformation (no change)
    #     # crop_fraction=1.0,        # No cropping
    # )

    # model = YOLO('yolov8l.pt')  # YOLOv8 Small detection model

    # model.train(
    #     data=mvpcd_yaml_path,
    #     epochs=epochs,
    #     lr0=learning_rate,
    #     batch=batch_size,
    #     imgsz=640,  # Adjust image size as needed
    #     name=project_name,
    #     project=os.path.join(project_root, 'runs', task_label),
    #     verbose=True,
    #     augment=True,    # Enable data augmentation
    #     pretrained=True, # Use pre-trained weights
    #     weight_decay=weight_decay,  # Adjust regularization if needed
    #     task=task_label, # Specify the task
    #     optimizer='AdamW',  # Use AdamW optimizer
    #     # cls=1.0,                   # Increased classification loss weight
    #     # box=0.5,                   # Decreased box loss weight
    #     # conf=0.001,
    #     workers=8,
    #     # scale=0.5,
    #     # translate=0.1,
    #     # hsv_v=0.4,
    #     # degrees=0.05,
    #     # hsv_h=0.015,
    #     # hsv_s=0.7,
    #     # shear=0.05,
    #     # flipud=0.05,
    #     # fliplr=0.05,vvv
    #     # perspective=0.0,
    #     # mosaic=0.1,
    #     # copy_paste=0.025,
    #     # mixup=0.05,
    #     # bgr=0.025,
    #     # erasing=0.4,
    #     # crop_fraction=1.0,
    # )
    # model.train(
    #     data=mvpcd_yaml_path,
    #     epochs=epochs,
    #     lr0=learning_rate,
    #     batch=batch_size,
    #     imgsz=640,  # Adjust image size as needed
    #     name=project_name,
    #     project=os.path.join(project_root, 'runs', task_label),
    #     verbose=True,
    #     augment=True,             # Enable data augmentation
    #     pretrained=True,          # Use pre-trained weights
    #     weight_decay=weight_decay,# Adjust regularization if needed
    #     task=task_label,          # Specify the task
    #     optimizer='AdamW',        # Use AdamW optimizer
    #     workers=8,
    #     # Loss Function Weights
    #     cls=1.0,                   # Increased classification loss weight
    #     box=0.5,                   # Decreased box loss weight
    #     conf=0.001,                # Adjust confidence loss weight
    #     # Geometric Augmentations
    #     scale=0.5,                 # Random scaling
    #     translate=0.1,             # Random translation
    #     shear=0.05,                # Random shear
    #     flipud=0.05,               # Random vertical flip
    #     fliplr=0.05,               # Random horizontal flip
    #     # Color Space Augmentations
    #     hsv_h=0.015,               # Hue adjustment
    #     hsv_s=0.7,                 # Saturation adjustment
    #     hsv_v=0.4,                 # Value (brightness) adjustment
    #     bgr=0.025,                 # BGR color augmentation
    #     # Advanced Augmentations
    #     mosaic=0.1,                # Mosaic augmentation
    #     mixup=0.05,                # MixUp augmentation
    #     erasing=0.4,               # Random erasing
    #     # Optional: Uncomment if needed
    #     # copy_paste=0.025,        # Copy-Paste augmentation
    #     # perspective=0.0,         # Perspective transformation (no change)
    #     # crop_fraction=1.0,        # No cropping
    # )

    # model = YOLO('yolov8n.pt')  # YOLOv8 Small detection model

    # model.train(
    #     data=mvpcd_yaml_path,
    #     epochs=epochs,
    #     lr0=learning_rate,
    #     batch=batch_size,
    #     imgsz=640,  # Adjust image size as needed
    #     name=project_name,
    #     project=os.path.join(project_root, 'runs', task_label),
    #     verbose=True,
    #     augment=True,    # Enable data augmentation
    #     pretrained=True, # Use pre-trained weights
    #     weight_decay=weight_decay,  # Adjust regularization if needed
    #     task=task_label, # Specify the task
    #     optimizer='AdamW',  # Use AdamW optimizer
    #     # cls=1.0,                   # Increased classification loss weight
    #     # box=0.5,                   # Decreased box loss weight
    #     # conf=0.001,
    #     workers=8,
    #     # scale=0.5,
    #     # translate=0.1,
    #     # hsv_v=0.4,
    #     # degrees=0.05,
    #     # hsv_h=0.015,
    #     # hsv_s=0.7,
    #     # shear=0.05,
    #     # flipud=0.05,
    #     # fliplr=0.05,vvv
    #     # perspective=0.0,
    #     # mosaic=0.1,
    #     # copy_paste=0.025,
    #     # mixup=0.05,
    #     # bgr=0.025,
    #     # erasing=0.4,
    #     # crop_fraction=1.0,
    # )
    # model.train(
    #     data=mvpcd_yaml_path,
    #     epochs=epochs,
    #     lr0=learning_rate,
    #     batch=batch_size,
    #     imgsz=640,  # Adjust image size as needed
    #     name=project_name,
    #     project=os.path.join(project_root, 'runs', task_label),
    #     verbose=True,
    #     augment=True,             # Enable data augmentation
    #     pretrained=True,          # Use pre-trained weights
    #     weight_decay=weight_decay,# Adjust regularization if needed
    #     task=task_label,          # Specify the task
    #     optimizer='AdamW',        # Use AdamW optimizer
    #     workers=8,
    #     # Loss Function Weights
    #     cls=1.0,                   # Increased classification loss weight
    #     box=0.5,                   # Decreased box loss weight
    #     conf=0.001,                # Adjust confidence loss weight
    #     # Geometric Augmentations
    #     scale=0.5,                 # Random scaling
    #     translate=0.1,             # Random translation
    #     shear=0.05,                # Random shear
    #     flipud=0.05,               # Random vertical flip
    #     fliplr=0.05,               # Random horizontal flip
    #     # Color Space Augmentations
    #     hsv_h=0.015,               # Hue adjustment
    #     hsv_s=0.7,                 # Saturation adjustment
    #     hsv_v=0.4,                 # Value (brightness) adjustment
    #     bgr=0.025,                 # BGR color augmentation
    #     # Advanced Augmentations
    #     mosaic=0.1,                # Mosaic augmentation
    #     mixup=0.05,                # MixUp augmentation
    #     erasing=0.4,               # Random erasing
    #     # Optional: Uncomment if needed
    #     # copy_paste=0.025,        # Copy-Paste augmentation
    #     # perspective=0.0,         # Perspective transformation (no change)
    #     # crop_fraction=1.0,        # No cropping
    # )

    # model.train(
    #     data=mvpcd_yaml_path,
    #     epochs=epochs,
    #     lr0=learning_rate,
    #     batch=batch_size,
    #     imgsz=640,  # Adjust image size as needed
    #     name=project_name,
    #     project=os.path.join(project_root, 'runs', task_label),
    #     verbose=True,
    #     augment=True,    # Enable data augmentation
    #     pretrained=True, # Use pre-trained weights
    #     weight_decay=0.0005,      # Weight decay for regularization
    #     task=task_label, # Specify the task
    #     optimizer='SGD',          # Optimizer (SGD or Adam)
    #     lrf=0.1,                  # Final learning rate (lr0 * lrf)
    #     momentum=0.937,           # Momentum
    #     warmup_epochs=3.0,        # Warmup epochs
    #     warmup_momentum=0.8,      # Warmup momentum
    #     warmup_bias_lr=0.1,       # Warmup bias learning rate
    #     box=7.5,                  # Box loss gain
    #     cls=0.5,                  # Class loss gain
    #     dfl=1.5,                  # Distribution Focal Loss gain
    #     label_smoothing=0.0,      # Label smoothing
    #     save_period=10,           # Save checkpoints every 10 epochs
    #     resume=False,             # Resume training from the last checkpoint
    #     device=0,                 # GPU device (set to -1 for CPU)
    #     exist_ok=True,            # Overwrite existing runs
    #     cache=True,               # Cache images for faster training
    #     multi_scale=True,         # Enable multi-scale training
    #     single_cls=False,         # False for multi-class training
    #     workers=8,                # Number of dataloader workers
    #     patience=50,              # Early stopping patience
    #     iou_thres=0.45            # IoU threshold for NMS
    # )
    latest_model_path = model.ckpt_path  # Or use model.last
    print("Path to the last trained model:", latest_model_path)

if __name__ == "__main__":
    # args = parse_arguments()
    config = load_config()
    train_yolo_model_masks(
        config,
        task="detection",
        epochs=150,
        learning_rate=0.000125,
        batch_size=6,
        weight_decay=0.000075
    )