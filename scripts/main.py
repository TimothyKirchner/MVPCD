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
from remove_greenscreen import adjust_green_thresholds, remove_green_background  # Import the new functions
from capture import capture_images
from set_roi import set_rois
from preprocess import preprocess_images
from train_model import train_yolo_model
from incremental_train import incremental_train_yolo_model  # New script for incremental learning
from run_inference import run_inference
from split_dataset import split_dataset

def load_config(config_path='config/config.yaml'):
    """Load the YAML configuration file."""
    config_full_path = os.path.join(project_root, config_path)
    if not os.path.exists(config_full_path):
        # Initialize with default structure if config doesn't exist
        config = {
            'image_counters': {},
            'depth_thresholds': {},
            'output': {
                'image_dir': 'data/images',
                'label_dir': 'data/labels',
                'depth_dir': 'data/depth_maps'
            },
            'debug': {
                'rgbmask': 'data/debug/rgbmask',
                'depthmask': 'data/debug/depthmask',
                'combined_mask': 'data/debug/combined_mask',
                'contours': 'data/debug/contours',
                'bboxes': 'data/debug/bboxes'
            }
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

def prompt_for_class_name(existing_classes):
    """Prompt the user to enter a single class name."""
    while True:
        class_name = input("Please enter the class name for the object: ").strip()
        if not class_name:
            print("Class name cannot be empty.")
            continue
        if class_name in existing_classes:
            print(f"Class '{class_name}' already exists.")
            continue
        return class_name

def delete_files_in_directory(directory_path):
    """Delete all files in the specified directory."""
    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"All files in '{directory_path}' have been deleted successfully.")
    except OSError:
        print(f"Error occurred while deleting files in '{directory_path}'.")

def delete_all_data(config):
    """Delete all existing data (images, labels) and reset configurations."""
    # Extract the 'output' section and store values in a list
    dirs = list(config['output'].values())

    for datadir in dirs:
        full_path = os.path.join(project_root, datadir)
        if os.path.exists(full_path):
            print(f"Deleting files in '{full_path}'")
            delete_files_in_directory(full_path)
        else:
            print(f"Directory '{full_path}' does not exist. Skipping.")

    # Extract the 'debug' section and store values in a list
    debugdirs = list(config['debug'].values())

    for debugdir in debugdirs:
        full_path = os.path.join(project_root, debugdir)
        if os.path.exists(full_path):
            print(f"Deleting files in '{full_path}'")
            delete_files_in_directory(full_path)
        else:
            print(f"Directory '{full_path}' does not exist. Skipping.")

    # Reset configurations
    config['image_counters'] = {}
    config['depth_thresholds'] = {}
    save_config(config)
    print("Reset configurations in 'config.yaml'.")

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

def main():
    """Main function to run the MVPCD pipeline."""
    parser = argparse.ArgumentParser(description='MVPCD Pipeline')
    parser.add_argument('--configure', action='store_true', help='Configure settings')
    args = parser.parse_args()
    processedimages=[] 
    counter=0

    config = load_config()

    # Ask the user if they want to train a new model or modify an existing one
    while True:
        choice = input("Do you want to (1) train a new model or (2) add/remove classes from an existing model? Enter 1 or 2: ").strip()
        if choice == '1':
            train_new_model = True
            break
        elif choice == '2':
            train_new_model = False
            break
        else:
            print("Invalid input. Please enter 1 or 2.")

    if train_new_model:
        # Proceed with existing pipeline for training a new model
        # Optionally delete all existing data
        delete_data = input("Do you want to delete all existing data (images, labels) and reset configurations? (y/n): ").strip().lower()
        if delete_data == 'y':
            delete_all_data(config)

        classes_to_add = []
        while True:
            add_object = input("Do you want to add an object? (y/n): ").strip().lower()
            if add_object == 'y':
                # Prompt for class name
                existing_classes = []  # No classes yet since config is reset
                class_name = prompt_for_class_name(existing_classes)
                classes_to_add.append(class_name)
                print(f"Class '{class_name}' added for training.")
            elif add_object == 'n':
                if not classes_to_add:
                    print("No classes added. Exiting.")
                    return
                break
            else:
                print("Please enter 'y' or 'n'.")

        # Start Live Depth Viewer and RGB Chroma Key Adjusters for each new class
        for class_name in classes_to_add:
            # Start Live Depth Viewer
            print(f"\nStarting Live Depth Viewer to adjust depth cutoff values for class '{class_name}'...")
            print("Press 'r' to restart viewer, 'v' to restart OpenCV window, 'q' to quit and save values.")
            from live_depth_feed import live_depth_feed  # Import here to ensure updated path
            live_depth_feed(config, class_name)

            # Start Live RGB Viewer
            print(f"\nStarting Live RGB Viewer to adjust chroma keying colors for class '{class_name}'...")
            from live_rgb_chromakey import live_rgb_chromakey  # Import here to ensure updated path
            live_rgb_chromakey(config, class_name)

            # Capture images
            print(f"\nStarting image capture for class '{class_name}'...")
            capture_images(config, class_name)

            # Set ROI
            print(f"\nSetting Region of Interest (ROI) for class '{class_name}'...")
            set_rois(config)

            # Split dataset
            print("\nSplitting dataset into training and validation sets...")
            split_dataset(config)

            # Preprocess images
            print("\nPreprocessing images...")
            preprocess_images(config, processedimages=processedimages, counter=counter)

        # Update mvpcd.yaml with new classes
        update_mvpcd_yaml(classes_to_add)

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

        print("\nAdjusting green screen thresholds and removing background from images...")
        adjust_green_thresholds(config)
        remove_green_background(config)
        print("Green screen removal completed.")

        # Train the model
        train_yolo_model(config, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)

        # Clear configurations after training
        config['image_counters'] = {}
        config['depth_thresholds'] = {}
        save_config(config)
        print("\nConfigurations have been cleared after training.")

    else:
        # Modify an existing model by adding/removing classes
        # Get the name/path of the existing model
        delete_data = input("Do you want to delete all existing data (images, labels) and reset configurations? (y/n): ").strip().lower()
        if delete_data == 'y':
            delete_all_data(config)
        model_name = input("Enter the name of the existing model to modify (e.g., 'mvpcd_yolov8'): ").strip()
        model_path = os.path.join(project_root, 'runs', 'detect', model_name, 'weights', 'best.pt')
        if not os.path.exists(model_path):
            # Try checking in the weights subdirectory
            model_path = os.path.join(project_root, 'runs', 'detect', model_name, 'weights', 'best.pt')
            if not os.path.exists(model_path):
                print(f"Model not found at '{model_path}'. Please check the model name.")
                return

        classes_to_add = []
        classes_to_remove = []

        # Ask for classes to add
        while True:
            add_new_class = input("Do you want to add a new class? (y/n): ").strip().lower()
            if add_new_class == 'y':
                # Prompt for class name
                existing_classes = []  # Existing classes are managed by the model, not config
                class_name = prompt_for_class_name(existing_classes)
                classes_to_add.append(class_name)
                print(f"Class '{class_name}' added for incremental training.")
            elif add_new_class == 'n':
                break
            else:
                print("Please enter 'y' or 'n'.")

        # Ask for classes to remove
        while True:
            remove_class_prompt = input("Do you want to remove an existing class from the model? (y/n): ").strip().lower()
            if remove_class_prompt == 'y':
                class_name = input("Enter the class name to remove: ").strip()
                classes_to_remove.append(class_name)
                print(f"Class '{class_name}' added for removal.")
            elif remove_class_prompt == 'n':
                break
            else:
                print("Please enter 'y' or 'n'.")

        if not classes_to_add and not classes_to_remove:
            print("No classes to add or remove. Exiting.")
            return

        # Collect data for new classes
        for class_name in classes_to_add:
            # Start Live Depth Viewer
            print(f"\nStarting Live Depth Viewer to adjust depth cutoff values for class '{class_name}'...")
            print("Press 'r' to restart viewer, 'v' to restart OpenCV window, 'q' to quit and save values.")
            from live_depth_feed import live_depth_feed  # Import here to ensure updated path
            live_depth_feed(config, class_name)

            # Start Live RGB Viewer
            print(f"\nStarting Live RGB Viewer to adjust chroma keying colors for class '{class_name}'...")
            from live_rgb_chromakey import live_rgb_chromakey  # Import here to ensure updated path
            live_rgb_chromakey(config, class_name)

            # Capture images
            print(f"\nStarting image capture for class '{class_name}'...")
            capture_images(config, class_name)

            # Set ROI
            print(f"\nSetting Region of Interest (ROI) for class '{class_name}'...")
            set_rois(config)

            # Split dataset
            print("\nSplitting dataset into training and validation sets...")
            split_dataset(config)

            # Preprocess images
            print("\nPreprocessing images...")
            preprocess_images(config, processedimages=processedimages, counter=counter)

        # Perform incremental training with knowledge distillation
        print("\n--- Incremental Training of YOLOv8 Model with Knowledge Distillation ---")
        epochs = input("Enter number of epochs for incremental training (default 50): ").strip()
        epochs = int(epochs) if epochs.isdigit() else 50

        learning_rate = input("Enter learning rate for incremental training (default 0.0001): ").strip()
        try:
            learning_rate = float(learning_rate)
        except ValueError:
            learning_rate = 0.0001

        batch_size = input("Enter batch size for incremental training (default 8): ").strip()
        batch_size = int(batch_size) if batch_size.isdigit() else 8

        # Perform incremental training
        incremental_train_yolo_model(
            config,
            base_model_path=model_path,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            classes_to_remove=classes_to_remove
        )

        # Clear configurations after training
        config['image_counters'] = {}
        config['depth_thresholds'] = {}
        save_config(config)
        print("\nConfigurations have been cleared after incremental training.")

    # Ask for Inference
    while True:
        run_inf = input("\nDo you want to run the trained model for live inference? (y/n): ").strip().lower()
        if run_inf == 'y':
            print("\n--- Running Live Inference ---")
            run_inference(config)
            break
        elif run_inf == 'n':
            print("Inference skipped.")
            break
        else:
            print("Please enter 'y' or 'n'.")

    print("\nPipeline completed!")

if __name__ == "__main__":
    main()
