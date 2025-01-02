# scripts/archive_dataset.py

import os
import shutil
import yaml
import datetime
import logging
import sys

# Adjust sys.path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_config(config_path='config/config.yaml'):
    """
    Load the YAML configuration file.
    """
    config_full_path = os.path.join(project_root, config_path)
    if not os.path.exists(config_full_path):
        print(f"Configuration file not found at {config_full_path}.")
        sys.exit(1)
    with open(config_full_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def archive_dataset(config, model_name):
    """
    Archive the current dataset and related configurations.

    Parameters:
        config (dict): Configuration dictionary.
        model_name (str): Name of the model being archived.
    """
    print("in archive_dataset. config: ", config, "; model_name: ", model_name)

    # Set up logging
    log_file = os.path.join(project_root, 'scripts', 'archive_dataset.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )

    logging.info(f"Starting archiving process for model: {model_name}")
    print(f"Starting archiving process for model: {model_name}")

    # Ask the user if they want to specify a custom folder name
    user_choice = input("Do you want to specify a custom archive folder name? (y/n): ").strip().lower()
    if user_choice == 'y':
        while True:
            custom_name = input("Enter the custom folder name for the archive: ").strip()
            # Validate folder name (no invalid characters)
            if not custom_name:
                print("Folder name cannot be empty. Please try again.")
                continue
            # Define a list of invalid characters
            invalid_chars = ['/', '\\', '?', '<', '>', ':', '*', '|', '"']
            if any(char in custom_name for char in invalid_chars):
                print("Folder name contains invalid characters. Please avoid / \\ ? < > : * | \"")
                continue
            else:
                archive_folder_name = custom_name
                break
    else:
        # Use default naming method
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_folder_name = f"{model_name}_{timestamp}"

    # Define the archive directory path
    archive_dir = os.path.join(project_root, 'archive', archive_folder_name)

    try:
        os.makedirs(archive_dir, exist_ok=True)
        logging.info(f"Created archive directory at '{archive_dir}'.")
        print(f"Created archive directory at '{archive_dir}'.")
    except Exception as e:
        logging.error(f"Failed to create archive directory '{archive_dir}': {e}")
        print(f"Failed to create archive directory '{archive_dir}': {e}")
        sys.exit(1)

    # Define source directories to archive
    source_dirs = [
        config['output']['train_image_dir'],
        config['output']['train_label_dir'],
        config['output']['val_image_dir'],
        config['output']['val_label_dir'],
        config['output']['test_image_dir'],
        config['output']['test_label_dir'],
        config['debug']['bboxes'],
        config['debug']['combined_mask'],
        config['debug']['contours'],
        config['debug']['depthmask'],
        config['debug']['rgbmask'],
        config['output']['background_image_dir']  # Added background images directory
    ]

    # Copy each source directory to the archive directory
    for src in source_dirs:
        src_full_path = os.path.join(project_root, src)
        if os.path.exists(src_full_path):
            dest_path = os.path.join(archive_dir, src)
            try:
                shutil.copytree(src_full_path, dest_path, dirs_exist_ok=True)
                logging.info(f"Archived '{src_full_path}' to '{dest_path}'.")
                print(f"Archived '{src_full_path}' to '{dest_path}'.")
            except Exception as e:
                logging.error(f"Failed to archive '{src_full_path}' to '{dest_path}': {e}")
                print(f"Failed to archive '{src_full_path}' to '{dest_path}': {e}")
        else:
            logging.warning(f"Source directory '{src_full_path}' does not exist. Skipping.")
            print(f"Source directory '{src_full_path}' does not exist. Skipping.")

    # Save the class_names to a separate file in the archive
    class_names = config.get('class_names', [])
    print("class_names: ", class_names)
    if class_names:
        class_names_file = os.path.join(archive_dir, 'class_names.yaml')
        print("class_names_file: ", class_names_file)
        try:
            with open(class_names_file, 'w') as f:
                yaml.dump({'class_names': class_names}, f)
            logging.info(f"Saved class names to '{class_names_file}'.")
            print(f"Saved class names to '{class_names_file}'.")
        except Exception as e:
            logging.error(f"Failed to save class names to '{class_names_file}': {e}")
            print(f"Failed to save class names to '{class_names_file}': {e}")
    else:
        logging.warning("No class names found in configuration. Skipping saving class names.")
        print("No class names found in configuration. Skipping saving class names.")

    # Save the archive folder name and class_names to a .txt file
    archive_info_file = os.path.join(archive_dir, 'archive_info.txt')
    try:
        with open(archive_info_file, 'w') as f:
            f.write(f"Archive Folder: {archive_folder_name}\n")
            f.write("Class Names:\n")
            for class_name in class_names:
                f.write(f"- {class_name}\n")
        logging.info(f"Saved archive information to '{archive_info_file}'.")
        print(f"Saved archive information to '{archive_info_file}'.")
    except Exception as e:
        logging.error(f"Failed to save archive information to '{archive_info_file}': {e}")
        print(f"Failed to save archive information to '{archive_info_file}': {e}")

    # Optionally, save the entire config to the archive for reference
    config_file = os.path.join(archive_dir, 'config_backup.yaml')
    print("config_file: ", config_file)
    try:
        with open(config_file, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
        logging.info(f"Saved configuration to '{config_file}'.")
        print(f"Saved configuration to '{config_file}'.")
    except Exception as e:
        logging.error(f"Failed to save configuration to '{config_file}': {e}")
        print(f"Failed to save configuration to '{config_file}': {e}")

    logging.info(f"Dataset and configurations have been successfully archived to '{archive_dir}'.")
    print(f"Dataset and configurations have been successfully archived to '{archive_dir}'.")

if __name__ == "__main__":
    config = load_config()

    # Example: Specify the model name here or retrieve it from arguments/config
    model_name = "your_model_name"  # Replace with your actual model name

    archive_dataset(config, model_name)