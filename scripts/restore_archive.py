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

def update_class_names(archive_class_names, project_class_names, add_list, remove_list):
    """
    Update the project class names based on the add and remove lists.
    """
    # Remove classes
    project_class_names = [cls for cls in project_class_names if cls not in remove_list]

    # Add new classes from the archive
    for cls in add_list:
        if cls not in project_class_names:
            project_class_names.append(cls)

    return project_class_names

def save_class_names(config, class_names):
    """
    Save the updated class names to the config.yaml file.
    """
    config['class_names'] = class_names
    config_yaml_path = os.path.join(project_root, 'config', 'config.yaml')
    try:
        with open(config_yaml_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
        logging.info(f"Updated 'config.yaml' with new class names.")
        print(f"Updated 'config.yaml' with new class names.")
    except Exception as e:
        logging.error(f"Failed to update 'config.yaml': {e}")
        print(f"Failed to update 'config.yaml': {e}")

def remove_class_data(class_name, config):
    """
    Remove images and labels related to the specified class.
    """
    output_paths = config.get('output', {})
    # Folders to search: train, val, test images and labels
    folders = ['train_image_dir', 'val_image_dir', 'test_image_dir',
               'train_label_dir', 'val_label_dir', 'test_label_dir']
    
    for folder_key in folders:
        folder_path = os.path.join(project_root, output_paths.get(folder_key, ''))
        if not os.path.exists(folder_path):
            continue
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                if filename.startswith(class_name):
                    os.remove(file_path)
                    logging.info(f"Removed file '{file_path}' for class '{class_name}'.")
    print(f"Removed all data for class '{class_name}'.")

def remap_class_ids_in_label_file(label_file_path, class_name, project_class_names, archive_class_names):
    """
    Remap class IDs in the label file to match the project class indices.
    """
    with open(label_file_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        class_id = int(parts[0])
        archive_class = archive_class_names[class_id]
        if archive_class == class_name:
            # Get the new class ID from project class names
            new_class_id = project_class_names.index(class_name)
            parts[0] = str(new_class_id)
            new_line = ' '.join(parts)
            new_lines.append(new_line)
    # Overwrite the label file with updated class IDs
    with open(label_file_path, 'w') as f:
        for line in new_lines:
            f.write(line + '\n')

def copy_class_data(archive_path, config, add_list, project_class_names, archive_class_names):
    """
    Copy images and labels for specified classes from the archive to the project.
    """
    output_paths = config.get('output', {})
    # Folders to process: train, val, test images and labels
    folders = ['train_image_dir', 'val_image_dir', 'test_image_dir',
               'train_label_dir', 'val_label_dir', 'test_label_dir']
    
    for folder_key in folders:
        rel_path = output_paths.get(folder_key, '')
        source_dir = os.path.join(archive_path, rel_path)
        dest_dir = os.path.join(project_root, rel_path)
        
        if not os.path.exists(source_dir):
            continue
        
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
        
        for filename in os.listdir(source_dir):
            # Check if the file corresponds to the classes to add
            for class_name in add_list:
                if filename.startswith(class_name):
                    s = os.path.join(source_dir, filename)
                    d = os.path.join(dest_dir, filename)
                    if os.path.isfile(s):
                        # If it's a label file, remap class IDs if necessary
                        if folder_key.endswith('label_dir') and filename.endswith('.txt'):
                            shutil.copy2(s, d)
                            # Remap class IDs
                            remap_class_ids_in_label_file(d, class_name, project_class_names, archive_class_names)
                        else:
                            shutil.copy2(s, d)
                    break  # No need to check other class names for this file

def restore_archive(add_list, remove_list):
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
    
    # Load class names from project and archive
    project_class_names = config.get('class_names', [])
    
    archive_class_names_path = os.path.join(selected_archive_path, 'config', 'class_names.yaml')
    if not os.path.exists(archive_class_names_path):
        logging.error(f"'class_names.yaml' not found in archive '{selected_archive_path}'.")
        print(f"'class_names.yaml' not found in archive '{selected_archive_path}'. Aborting.")
        sys.exit(1)
    
    with open(archive_class_names_path, 'r') as f:
        archive_class_names = yaml.safe_load(f).get('class_names', [])
    
    # Update class names
    updated_class_names = update_class_names(archive_class_names, project_class_names, add_list, remove_list)
    save_class_names(config, updated_class_names)
    
    # Remove classes
    for class_name in remove_list:
        if class_name in project_class_names:
            remove_class_data(class_name, config)
        else:
            print(f"Class '{class_name}' not found in project. Skipping removal.")
            logging.warning(f"Class '{class_name}' not found in project. Skipping removal.")
    
    # Add classes
    for class_name in add_list:
        if class_name in archive_class_names:
            copy_class_data(selected_archive_path, config, [class_name], updated_class_names, archive_class_names)
            print(f"Added data for class '{class_name}'.")
            logging.info(f"Added data for class '{class_name}'.")
        else:
            print(f"Class '{class_name}' not found in archive. Skipping addition.")
            logging.warning(f"Class '{class_name}' not found in archive. Skipping addition.")
    
    logging.info("Archive restoration completed successfully.")
    print("\nArchive restoration completed successfully.")

if __name__ == "__main__":
    # Example usage: you can remove this part if you don't need it
    # and will call restore_archive(add_list, remove_list) from main.py
    add_classes = []
    remove_classes = []
    restore_archive(add_classes, remove_classes)