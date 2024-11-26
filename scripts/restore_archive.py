# scripts/restore_archive.py

import os
import shutil
import yaml
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

def setup_logging():
    """
    Set up logging for the script.
    """
    log_file = os.path.join(project_root, 'scripts', 'restore_archive.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    logging.info("=== Starting Restore Archive Process ===")
    print(f"Logging initialized. Logs will be saved to '{log_file}'.")

def list_archive_folders(archive_dir):
    """
    List all directories in the archive directory.
    """
    if not os.path.exists(archive_dir):
        print(f"Archive directory '{archive_dir}' does not exist.")
        sys.exit(1)
    
    folders = [f for f in os.listdir(archive_dir) if os.path.isdir(os.path.join(archive_dir, f))]
    
    if not folders:
        print(f"No archive folders found in '{archive_dir}'.")
        sys.exit(1)
    
    return folders

def display_folders(folders):
    """
    Display archived folders with corresponding numbers.
    """
    print("\nAvailable Archive Folders:")
    for idx, folder in enumerate(folders, 1):
        print(f"{idx}. {folder}")

def get_user_choice(num_folders):
    """
    Prompt the user to select an archive by number.
    """
    while True:
        try:
            choice = int(input(f"\nEnter the number of the archive to restore (1-{num_folders}): ").strip())
            if 1 <= choice <= num_folders:
                return choice - 1  # Zero-based index
            else:
                print(f"Please enter a number between 1 and {num_folders}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def copy_data_from_archive(archive_path, config):
    """
    Copy data from the selected archive to current data directories.
    """
    source_output_dir = os.path.join(archive_path, 'output')
    if not os.path.exists(source_output_dir):
        logging.error(f"'output' directory not found in archive '{archive_path}'.")
        print(f"'output' directory not found in archive '{archive_path}'. Aborting.")
        sys.exit(1)
    
    # Mapping of source to destination directories
    mappings = {
        'train_image_dir': config['output']['train_image_dir'],
        'train_label_dir': config['output']['train_label_dir'],
        'val_image_dir': config['output']['val_image_dir'],
        'val_label_dir': config['output']['val_label_dir'],
        # Add more mappings if necessary
    }
    
    for key, dest_rel_path in mappings.items():
        source_dir = os.path.join(source_output_dir, key)
        dest_dir = os.path.join(project_root, dest_rel_path)
        
        if not os.path.exists(source_dir):
            logging.warning(f"Source directory '{source_dir}' does not exist in archive. Skipping.")
            print(f"Source directory '{source_dir}' does not exist in archive. Skipping.")
            continue
        
        if not os.path.exists(dest_dir):
            try:
                os.makedirs(dest_dir, exist_ok=True)
                logging.info(f"Created destination directory '{dest_dir}'.")
                print(f"Created destination directory '{dest_dir}'.")
            except Exception as e:
                logging.error(f"Failed to create destination directory '{dest_dir}': {e}")
                print(f"Failed to create destination directory '{dest_dir}': {e}")
                continue
        
        # Copy contents
        try:
            for item in os.listdir(source_dir):
                s = os.path.join(source_dir, item)
                d = os.path.join(dest_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            logging.info(f"Copied data from '{source_dir}' to '{dest_dir}'.")
            print(f"Copied data from '{source_dir}' to '{dest_dir}'.")
        except Exception as e:
            logging.error(f"Failed to copy from '{source_dir}' to '{dest_dir}': {e}")
            print(f"Failed to copy from '{source_dir}' to '{dest_dir}': {e}")

def update_class_names(archive_path, config):
    """
    Update the config.yaml with class_names from the archive.
    """
    class_names_path = os.path.join(archive_path, 'class_names.yaml')
    if not os.path.exists(class_names_path):
        logging.warning(f"'class_names.yaml' not found in archive '{archive_path}'. Skipping class names update.")
        print(f"'class_names.yaml' not found in archive '{archive_path}'. Skipping class names update.")
        return
    
    with open(class_names_path, 'r') as f:
        archive_class_names = yaml.safe_load(f).get('class_names', [])
    
    if not archive_class_names:
        logging.warning(f"No class names found in '{class_names_path}'. Skipping class names update.")
        print(f"No class names found in '{class_names_path}'. Skipping class names update.")
        return
    
    # Update config
    config['class_names'] = archive_class_names
    
    # Save updated config.yaml
    config_yaml_path = os.path.join(project_root, 'config', 'config.yaml')
    try:
        with open(config_yaml_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
        logging.info(f"Updated 'config.yaml' with class names from archive '{archive_path}'.")
        print(f"Updated 'config.yaml' with class names from archive '{archive_path}'.")
    except Exception as e:
        logging.error(f"Failed to update 'config.yaml': {e}")
        print(f"Failed to update 'config.yaml': {e}")

def restore_archive():
    """
    Function that performs the entire process from user selection to importing archive data.
    """
    setup_logging()
    config = load_config()
    
    # Define archive directory
    archive_dir = os.path.join(project_root, 'archive')
    folders = list_archive_folders(archive_dir)
    
    display_folders(folders)
    
    choice_idx = get_user_choice(len(folders))
    selected_folder = folders[choice_idx]
    print(f"\nSelected archive: '{selected_folder}'")
    logging.info(f"User selected archive '{selected_folder}'.")
    
    selected_archive_path = os.path.join(archive_dir, selected_folder)
    
    # Copy data
    copy_data_from_archive(selected_archive_path, config)
    
    # Update class_names
    update_class_names(selected_archive_path, config)
    
    logging.info("Archive restoration completed successfully.")
    print("\nArchive restoration completed successfully.")

if __name__ == "__main__":
    restore_archive()