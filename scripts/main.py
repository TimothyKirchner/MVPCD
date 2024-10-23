# scripts/main.py
import sys
import os
# Adjust sys.path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import yaml
import argparse
import cv2
import numpy as np
import time
from capture import capture_images
from set_roi import set_rois
from preprocess import preprocess_images
from annotate import annotate_images
from verify_annotations import verify_annotations
from train_model import train_yolo_model
from live_depth_feed import live_depth_feed
from live_rgb_chromakey import live_rgb_chromakey
from run_inference import run_inference
from split_dataset import split_dataset

def load_config(config_path='config/config.yaml'):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, config_path)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(config, config_path='config/config.yaml'):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, config_path)
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

def prompt_for_class_names(config):
    class_names = []
    while True:
        class_name = input("Please enter the class name for the object: ").strip()
        if class_name:
            if class_name in config.get('class_names', []):
                print(f"Class '{class_name}' already exists.")
                continue
            class_names.append(class_name)
            config['class_names'].append(class_name)
            config['image_counters'][class_name] = 1
            # Initialize depth_threshold for the new class
            if 'depth_thresholds' not in config:
                config['depth_thresholds'] = {}
            config['depth_thresholds'][class_name] = {'min': 500, 'max': 2000}  # Default values
        else:
            print("Class name cannot be empty.")
            continue
        more = input("Do you want to add another class? (y/n): ").strip().lower()
        if more != 'y':
            break
    save_config(config)
    return class_names

def prompt_for_training_params():
    print("Enter training parameters (leave blank for defaults):")
    epochs = input("Number of epochs (default 50): ").strip()
    epochs = int(epochs) if epochs.isdigit() else 50

    learning_rate = input("Learning rate (default 0.001): ").strip()
    try:
        learning_rate = float(learning_rate)
    except:
        learning_rate = 0.001

    batch_size = input("Batch size (default 16): ").strip()
    batch_size = int(batch_size) if batch_size.isdigit() else 16

    return epochs, learning_rate, batch_size

def update_mvpcd_yaml(config):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
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
            'nc': len(config.get('class_names', [])),
            'names': config.get('class_names', [])
        }, file)

def main():
    parser = argparse.ArgumentParser(description='MVPCD Pipeline')
    parser.add_argument('--configure', action='store_true', help='Configure settings')
    args = parser.parse_args()

    config = load_config()

    if args.configure or not config.get('class_names'):
        class_names = prompt_for_class_names(config)
        update_mvpcd_yaml(config)
    else:
        class_names = config.get('class_names', [])

    for class_name in class_names:
        print(f"\n--- Processing Class: {class_name} ---")
        
        print("Starting Live Depth Viewer to adjust depth cutoff values...")
        live_depth_feed_script = os.path.join(os.path.dirname(__file__), 'live_depth_feed.py')
        os.system(f"python {live_depth_feed_script} {class_name}")
        
        print("Starting Live RGB Viewer to adjust chroma keying colors...")
        live_rgb_chromakey_script = os.path.join(os.path.dirname(__file__), 'live_rgb_chromakey.py')
        os.system(f"python {live_rgb_chromakey_script} {class_name}")
        
        print(f"Starting image and depth map capture for class '{class_name}'...")
        capture_images(config, class_name)

    print("\nDefining the Region of Interest (ROI)...")
    set_roi_script = os.path.join(os.path.dirname(__file__), 'set_roi.py')
    os.system(f"python {set_roi_script}")

    # **Reordered Execution: Split Dataset Before Preprocessing**
    print("Splitting dataset into training and validation sets...")
    split_dataset_script = os.path.join(os.path.dirname(__file__), 'split_dataset.py')
    os.system(f"python {split_dataset_script}")

    print("Starting preprocessing of images...")
    preprocess_script = os.path.join(os.path.dirname(__file__), 'preprocess.py')
    os.system(f"python {preprocess_script}")

    while True:
        print("\n--- Training YOLOv8 Model ---")
        epochs, learning_rate, batch_size = prompt_for_training_params()
        train_model_script = os.path.join(os.path.dirname(__file__), 'train_model.py')
        os.system(f"python {train_model_script} {epochs} {learning_rate} {batch_size}")

        try_inference = input("Do you want to add another class and record more objects? (y/n): ").strip().lower()
        if try_inference == 'y':
            new_class_names = prompt_for_class_names(config)
            update_mvpcd_yaml(config)
            for class_name in new_class_names:
                print(f"\n--- Processing Class: {class_name} ---")
                
                print("Starting Live Depth Viewer to adjust depth cutoff values...")
                live_depth_feed_script = os.path.join(os.path.dirname(__file__), 'live_depth_feed.py')
                os.system(f"python {live_depth_feed_script} {class_name}")
                
                print("Starting Live RGB Viewer to adjust chroma keying colors...")
                live_rgb_chromakey_script = os.path.join(os.path.dirname(__file__), 'live_rgb_chromakey.py')
                os.system(f"python {live_rgb_chromakey_script} {class_name}")
                
                print(f"Starting image and depth map capture for class '{class_name}'...")
                capture_images(config, class_name)
            print("Splitting dataset into training and validation sets...")
            split_dataset_script = os.path.join(os.path.dirname(__file__), 'split_dataset.py')
            os.system(f"python {split_dataset_script}")
            print("Starting preprocessing of images for new classes...")
            preprocess_script = os.path.join(os.path.dirname(__file__), 'preprocess.py')
            os.system(f"python {preprocess_script}")
        else:
            break

    print("\nModel training completed.")

    try_inference = input("Do you want to run the trained model for live inference? (y/n): ").strip().lower()
    if try_inference == 'y':
        run_inference_script = os.path.join(os.path.dirname(__file__), 'run_inference.py')
        os.system(f"python {run_inference_script}")
    else:
        print("Inference skipped.")

    print("Pipeline completed!")

if __name__ == "__main__":
    main()
