# ~/Desktop/MVPCD/scripts/main.py

import sys
import os
import yaml
import argparse
import cv2
import numpy as np
import time

# Adjust sys.path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import modules directly since they are in the same directory
from capture import capture_images
from set_roi import set_rois
from preprocess import preprocess_images
from annotate import annotate_images
from verify_annotations import verify_annotations
from train_model import train_yolo_model
from live_depth_feed import live_depth_feed
from live_rgb_chromakey import live_rgb_chromakey
from run_inference import run_inference
from split_dataset import split_dataset  # Import the dataset splitting function

def load_config(config_path='config/config.yaml'):
    config_path = os.path.join(project_root, config_path)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(config, config_path='config/config.yaml'):
    config_path = os.path.join(project_root, config_path)
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

def configure_settings(config):
    # Placeholder for any configuration settings you might want to adjust
    pass

def prompt_for_class_name(config):
    if 'class_name' not in config:
        class_name = input("Please enter the class name for the object: ")
        config['class_name'] = class_name
        save_config(config)
    else:
        class_name = config['class_name']
    return class_name

def prompt_for_training_params():
    print("Enter training parameters (leave blank for defaults):")
    epochs = input("Number of epochs (default 50): ")
    epochs = int(epochs) if epochs.strip() != '' else 50

    learning_rate = input("Learning rate (default 0.001): ")
    learning_rate = float(learning_rate) if learning_rate.strip() != '' else 0.001

    batch_size = input("Batch size (default 16): ")
    batch_size = int(batch_size) if batch_size.strip() != '' else 16

    return epochs, learning_rate, batch_size

def main():
    parser = argparse.ArgumentParser(description='MVPCD Pipeline')
    parser.add_argument('--configure', action='store_true', help='Configure settings')
    args = parser.parse_args()

    config = load_config()

    if args.configure:
        configure_settings(config)

    print("Starting Live Depth Viewer to adjust depth cutoff values...")
    live_depth_feed(config)

    print("Starting Live RGB Viewer to adjust chroma keying colors...")
    live_rgb_chromakey(config)

    print("Defining the Region of Interest (ROI)...")
    set_rois(config)

    print("Starting image and depth map capture...")
    capture_images(config)

    print("Starting preprocessing of images...")
    preprocess_images(config)

    print("Splitting dataset into training and validation sets...")
    split_dataset()

    print("Starting YOLOv8 model training...")
    epochs, learning_rate, batch_size = prompt_for_training_params()
    train_yolo_model(config, epochs, learning_rate, batch_size)

    print("Model training completed.")

    # Prompt the user to run inference
    try_inference = input("Do you want to run the trained model for live inference? (y/n): ").strip().lower()
    if try_inference == 'y':
        run_inference(config)
    else:
        print("Inference skipped.")

    print("Pipeline completed!")

if __name__ == "__main__":
    main()
