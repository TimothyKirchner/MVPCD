# scripts/main.py
import sys
import os
import yaml
import argparse
import shutil

# Adjust sys.path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.camera_utils import initialize_camera, capture_frame
from capture import capture_images
from set_roi import set_rois
from preprocess import preprocess_images
from train_model import train_yolo_model
from run_inference import run_inference
from split_dataset import split_dataset

def load_config(config_path='config/config.yaml'):
    """Load the YAML configuration file."""
    config_full_path = os.path.join(project_root, config_path)
    if not os.path.exists(config_full_path):
        # Initialize with default structure if config doesn't exist
        config = {
            'class_names': [],
            'image_counters': {},
            'depth_thresholds': {}
        }
        save_config(config, config_path)
    else:
        with open(config_full_path, 'r') as file:
            config = yaml.safe_load(file)
    return config

def save_config(config, config_path='config/config.yaml'):
    """Save the YAML configuration file."""
    config_full_path = os.path.join(project_root, config_path)
    with open(config_full_path, 'w') as file:
        yaml.dump(config, file)

def prompt_for_class_name(config):
    """Prompt the user to enter a single class name."""
    while True:
        class_name = input("Please enter the class name for the object: ").strip()
        if not class_name:
            print("Class name cannot be empty.")
            continue
        if class_name in config.get('class_names', []):
            print(f"Class '{class_name}' already exists.")
            continue
        return class_name

def add_class(config, class_name):
    """Add a new class to the configuration."""
    config['class_names'].append(class_name)
    config['image_counters'][class_name] = 1
    # Initialize depth_threshold for the new class
    config['depth_thresholds'][class_name] = {'min': 500, 'max': 2000}  # Default values
    save_config(config)

def update_mvpcd_yaml(config):
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
            'nc': len(config.get('class_names', [])),
            'names': config.get('class_names', [])
        }, file)

def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")

def delete_all_data():
    
    configpath = os.path.join(project_root, "config/config.yaml")
    
    with open(configpath, "r") as file:
        data = yaml.safe_load(file)

    # Extract the 'output' section and store values in a list
    dirs = list(data['output'].values())

    for datadir in dirs:
        print("deleting files in ", os.path.join(project_root, datadir))
        delete_files_in_directory(os.path.join(project_root, datadir))

    # Extract the 'output' section and store values in a list
    debugdirs = list(data['debug'].values())

    for debugdir in debugdirs:
        print("deleting files in ", os.path.join(project_root, debugdir))
        delete_files_in_directory(os.path.join(project_root, debugdir))

    # Do not delete the config file, just reset class_names and related entries
    config = load_config()
    config['class_names'] = []
    config['image_counters'] = {}
    # Reset depth thresholds for existing classes
    for class_name in config.get('class_names', []):
        config['depth_thresholds'][class_name] = {'min': 500, 'max': 2000}
    config['depth_thresholds'] = {}
    save_config(config)
    print("Reset class names and configurations in 'config.yaml'.")

def main():
    """Main function to run the MVPCD pipeline."""
    parser = argparse.ArgumentParser(description='MVPCD Pipeline')
    parser.add_argument('--configure', action='store_true', help='Configure settings')
    counter = 0
    processedimages = []
    args = parser.parse_args()

    # Check if user wants to delete all data
    delete_data = input("Do you want to delete all existing data (images, labels) and reset configurations? (y/n): ").strip().lower()
    if delete_data == 'y':
        delete_all_data()

    config = load_config()

    while True:
        add_object = input("Do you want to add an object? (y/n): ").strip().lower()
        if add_object == 'y':
            # Prompt for class name
            class_name = prompt_for_class_name(config)
            add_class(config, class_name)
            update_mvpcd_yaml(config)

            # Start Live Depth Viewer
            print("\nStarting Live Depth Viewer to adjust depth cutoff values...")
            print("\nPress r to restart viewer, press v to restart opencv window, press q to quit and save values")
            from live_depth_feed import live_depth_feed  # Import here to ensure updated path
            live_depth_feed(config, class_name)

            # Start Live RGB Viewer
            print("\nStarting Live RGB Viewer to adjust chroma keying colors...")
            from live_rgb_chromakey import live_rgb_chromakey  # Import here to ensure updated path
            live_rgb_chromakey(config, class_name)

            # Capture images
            print(f"\nStarting image capture for class '{class_name}'...")
            capture_images(config, class_name)

            # Set ROI
            print("\nSetting Region of Interest (ROI)...")
            set_rois(config)

            # Split dataset
            print("\nSplitting dataset into training and validation sets...")
            split_dataset(config)

            # Preprocess images
            print("\nPreprocessing images...")
            preprocess_images(config, processedimages=processedimages, counter=counter)

        elif add_object == 'n':
            if not config.get('class_names'):
                print("No classes available for training. Exiting.")
                return
            break
        else:
            print("Please enter 'y' or 'n'.")

    # Start Training
    print("\n--- Training YOLOv8 Model ---")
    epochs = input("Enter number of epochs (default 100): ").strip()
    epochs = int(epochs) if epochs.isdigit() else 100

    learning_rate = input("Enter learning rate (default 0.0001): ").strip()
    try:
        learning_rate = float(learning_rate)
    except ValueError:
        learning_rate = 0.0001

    batch_size = input("Enter batch size (default 8): ").strip()
    batch_size = int(batch_size) if batch_size.isdigit() else 8

    # Pass 'config' when calling 'train_yolo_model'
    train_yolo_model(config, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)

    # Clear class names after training
    config['class_names'] = []
    config['image_counters'] = {}
    config['depth_thresholds'] = {}
    save_config(config)
    print("\nClass names and configurations have been cleared after training.")

    # Ask for Inference
    while True:
        run_inf = input("\nDo you want to run the trained model for live inference? (y/n): ").strip().lower()
        if run_inf == 'y':
            print("\n--- Running Live Inference ---")
            run_inference()
            break
        elif run_inf == 'n':
            print("Inference skipped.")
            break
        else:
            print("Please enter 'y' or 'n'.")

    print("\nPipeline completed!")

if __name__ == "__main__":
    main()
