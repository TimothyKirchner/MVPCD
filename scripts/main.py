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
from remove_greenscreen import replace_background_with_preset_color_using_contours
from capture import capture_images
from set_roi import set_rois
from preprocess import preprocess_images
from train_model import train_yolo_model
from train_model_masks import train_yolo_model_masks
from incremental_train import incremental_train_yolo_model  # New script for incremental learning
from run_inference import run_inference
from split_dataset import split_dataset
from archive_dataset import archive_dataset
from capture_backgrounds import capture_backgrounds
from replace_background_with_random_insert import replace_images_with_mosaic
from restore_archive import restore_archive

def load_config(config_path='config/config.yaml'):
    """Load the YAML configuration file."""
    config_full_path = os.path.join(project_root, config_path)
    print(f"Configuration file '{config_full_path}' not found. Creating a new one.")
    num_images = input("How many images should be taken?: ")
    try:
        num_images = int(num_images)
    except ValueError:
        print("Invalid input for number of images. Defaulting to 100.")
        num_images = 100
    interval = input("What should be the interval between images captured (in seconds)?: ")
    try:
        interval = float(interval)
    except ValueError:
        print("Invalid input for interval. Defaulting to 1.0 seconds.")
        interval = 1.0
    config = {
        'camera': {
            'fps': 30,
            'resolution': [1280, 720]
        },
        'capture': {
            'interval': interval,
            'num_images': num_images
        },
        'chroma_key_settings': {},  # Initialize as empty dict for per-class per-angle settings
        'debug': {
            'bboxes': 'data/debug/bboxes',
            'combined_mask': 'data/debug/combined_mask',
            'contours': 'data/debug/contours',
            'depthmask': 'data/debug/depthmask',
            'rgbmask': 'data/debug/rgbmask',
            "maskinyolo": "data/debug/maskinyolo",
            "coloringmask_random_bg": "data/debug/coloringmask_random_bg",
            "coloringmask": "data/debug/coloringmask",
            "placement": "data/debug/placement",
            "backgrounds": "data/backgrounds"
        },
        'depth_thresholds': {},
        'image_counters': {},
        'output': {
            'depth_dir': 'data/depth_maps',
            'image_dir': 'data/images',
            'label_dir': 'data/labels',
            'train_image_dir': 'data/images/train',
            'train_label_dir': 'data/labels/train',
            'val_image_dir': 'data/images/val',
            'val_label_dir': 'data/labels/val',
            'test_image_dir': 'data/images/test',        # Added Test Image Directory
            'test_label_dir': 'data/labels/test',        # Added Test Label Directory
            "background_image_dir": "data/backgrounds"
        },
        'rois': {},
        'class_names': []
    }
    save_config(config, config_path)
    print(f"Created default configuration at '{config_full_path}'.")
    return config

def save_config(config, config_path='config/config.yaml'):
    """Save the YAML configuration file."""
    config_full_path = os.path.join(project_root, config_path)
    config_dir = os.path.dirname(config_full_path)
    os.makedirs(config_dir, exist_ok=True)  # Ensure the config directory exists
    try:
        with open(config_full_path, 'w') as file:
            yaml.dump(config, file, sort_keys=False)
        print(f"Configuration saved to '{config_full_path}'.")
    except Exception as e:
        print(f"Failed to save configuration to '{config_full_path}': {e}")

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
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return
    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"All files in '{directory_path}' have been deleted successfully.")
    except OSError as e:
        print(f"Error occurred while deleting files in '{directory_path}': {e}")

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
    config['rois'] = {}
    config['chroma_key_settings'] = {}
    save_config(config)
    print("Reset configurations in 'config.yaml'.")

def update_mvpcd_yaml(class_names):
    """Update the mvpcd.yaml file with current class names."""
    mvpcd_yaml_path = os.path.join(project_root, 'data', 'mvpcd.yaml')
    try:
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
                'test': {                                          # Added Test Set Paths
                    'images': './data/images/test',
                    'labels': './data/labels/test'
                },
                'nc': len(class_names),
                'names': class_names
            }, file)
        print("Updated 'mvpcd.yaml' with current class names.")
    except Exception as e:
        print(f"Failed to update 'mvpcd.yaml': {e}")

def main():
    """Main function to run the MVPCD pipeline."""
    parser = argparse.ArgumentParser(description='MVPCD Pipeline')
    parser.add_argument('--configure', action='store_true', help='Configure settings')
    args = parser.parse_args()
    processedimages=[] 
    counter=0

    config = load_config()
    masktoinsert_dir = os.path.join(project_root, "data/debug/masktoinsert")
    delete_files_in_directory(masktoinsert_dir)

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

    delete_data = input("Do you want to delete all existing data (images, labels) and reset configurations? (y/n): ").strip().lower()
    if delete_data == 'y':
        delete_all_data(config)

    while True:
        choice = input("Do you want to (1) manually take pictures of your workspace or (2) have the model train on pictures with virtually generated backgrounds? 1 generally leads to better model performance in your specific workspace. Enter 1 or 2: ").strip()
        if choice == '1':
            capture_backgrounds(config, max_retries=5)
            take_background = True
            break
        elif choice == '2':
            take_background = False
            break
        else:
            print("Invalid input. Please enter 1 or 2.")

    # boxormask = ""
    # while boxormask != "1" and boxormask != "2":
    #     boxormask = input("Do you want to train with masks or bboxes? Input 1 for bbox, or 2 for masks: ")
    #     if boxormask == "1":
    #         task = "detection"
    #         mode = task
    #         break
    #     elif boxormask == "2":
    #         task = "segmentation"
    #         mode = task
    #         break

    task = mode = boxormask = "detection"

    classes_to_add = []
    class_angles = {}  # Dictionary to store number of angles per class
    while True:
        add_object = input("Do you want to add an object? (y/n): ").strip().lower()
        if add_object == 'y':
            # Prompt for class name
            existing_classes = config.get('class_names', [])
            class_name = prompt_for_class_name(existing_classes)
            classes_to_add.append(class_name)
            print(f"Class '{class_name}' added for training.")

            # Ask if the user wants to capture images from multiple angles for this class
            multi_angle_input = input(f"Do you want to capture images from multiple angles for class '{class_name}'? (y/n): ").strip().lower()
            if multi_angle_input == 'y':
                num_angles = input(f"How many angles do you want to capture for class '{class_name}'?: ").strip()
                try:
                    num_angles = int(num_angles)
                    if num_angles < 1:
                        print("Number of angles must be at least 1.")
                        num_angles = 1
                except ValueError:
                    print("Invalid input. Defaulting to 1 angle.")
                    num_angles = 1
            else:
                num_angles = 1
            class_angles[class_name] = num_angles
        elif add_object == 'n':
            if not classes_to_add:
                print("No classes added. Exiting.")
                return
            break
        else:
            print("Please enter 'y' or 'n'.")

    # **Corrected Line: Extend the list instead of appending**
    config["class_names"].extend(classes_to_add)
    save_config(config)  # Save the updated class names

    # Start processing each class
    for class_name in classes_to_add:
        num_angles = class_angles.get(class_name, 1)
        total_images = config['capture']['num_images']

        # Calculate images per angle
        images_per_angle = total_images // num_angles
        remainder = total_images % num_angles
        images_per_angle_list = [images_per_angle] * num_angles
        for i in range(remainder):
            images_per_angle_list[i] += 1  # Add extra images to the first angles

        for angle_index in range(num_angles):
            num_images_to_capture = images_per_angle_list[angle_index]
            print(f"\n--- Processing angle {angle_index + 1} of {num_angles} for class '{class_name}' ---")

            # Start Live Depth Viewer
            print(f"\nStarting Live Depth Viewer to adjust depth cutoff values for class '{class_name}', angle {angle_index}...")
            print("Press 'r' to restart viewer, 'v' to restart OpenCV window, 'q' to quit and save values.")
            from live_depth_feed import live_depth_feed  # Import here to ensure updated path
            live_depth_feed(config, class_name, angle_index)

            # Start Live RGB Viewer
            print(f"\nStarting Live RGB Viewer to adjust chroma keying colors for class '{class_name}', angle {angle_index}...")
            from live_rgb_chromakey import live_rgb_chromakey  # Import here to ensure updated path
            live_rgb_chromakey(config, class_name, angle_index)

            # Set ROI
            print(f"\nSetting Region of Interest (ROI) for class '{class_name}', angle {angle_index}...")
            set_rois(config, class_name, angle_index)

            # Capture images
            print(f"\nStarting image capture for class '{class_name}', angle {angle_index}...")
            capture_images(config, class_name, num_images_to_capture, angle_index)

    for class_name in classes_to_add:
        # Split dataset
        print("\nSplitting dataset into training, validation, and test sets...")
        split_dataset(config, class_name=class_name, test_size=0.1)  # Updated to include test split

        # Preprocess images
        print("\nPreprocessing images...")
        preprocess_images(config, processedimages=processedimages, counter=counter, mode=mode, class_name=class_name)
        if take_background:
            replace_images_with_mosaic(config, class_name=class_name)
            print("Replacing Background with Mosaic of Workspace.")
        else:
            replace_background_with_preset_color_using_contours(config, class_name=class_name)
            print("Replaced Background as solid color.")

    # Update mvpcd.yaml with new classes
    update_mvpcd_yaml(config["class_names"])

    image_dir = os.path.join(project_root, config['output']['image_dir'])
    background_dirs = [
        os.path.join(project_root, "data/debug/placement"),
        os.path.join(project_root, "data/placement")
    ]
    delete_files_in_directory(image_dir)
    for background_dir in background_dirs:
        delete_files_in_directory(background_dir)

    model_name = "mvpcd_yolov8"
    directory = os.path.join(project_root, 'runs', 'detect')

    while os.path.exists(os.path.join(directory, model_name)):
        # Append the counter to the base name, separated by an underscore
        model_name = f"{model_name}_{counter}"
        counter += 1

    archiving = input("Do you want to archive the current dataset? (y/n): ").strip().lower()
    print("config: ", config)
    print("model_name: ", model_name)
    if archiving == "y":
        archive_dataset(config, model_name)
        print("Archived dataset")

    # Start Training
    print("\n--- Training YOLOv8 Model ---")
    epochs = input("Enter number of epochs (default 100): ").strip()
    epochs = int(epochs) if epochs.isdigit() else 100

    learning_rate = input("Enter learning rate (default 0.0001): ").strip()
    try:
        learning_rate = float(learning_rate)
    except ValueError:
        learning_rate = 0.0001

    weight_decay = input("Enter weight decay (default 0.0001): ").strip()
    try:
        weight_decay = float(weight_decay)
    except ValueError:
        weight_decay = 0.0001

    batch_size = input("Enter batch size (default 16): ").strip()
    batch_size = int(batch_size) if batch_size.isdigit() else 16  
    
    if not train_new_model:
        restore_archive()

    train_yolo_model_masks(config, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, task=boxormask, weight_decay=weight_decay)

    # Clear configurations after training
    config['image_counters'] = {}
    config['depth_thresholds'] = {}
    config['rois'] = {}
    config['chroma_key_settings'] = {}
    save_config(config)
    print("\nConfigurations have been cleared after training.")

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