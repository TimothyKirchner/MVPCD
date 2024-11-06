# scripts/archive_dataset.py

import sys
import os
import shutil
import datetime

# Adjust sys.path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def archive_dataset(config, model_name):
    # Get the paths from the config
    train_image_dir = os.path.join(project_root, config['output']['train_image_dir'])
    val_image_dir = os.path.join(project_root, config['output']['val_image_dir'])
    train_label_dir = os.path.join(project_root, config['output']['train_label_dir'])
    val_label_dir = os.path.join(project_root, config['output']['val_label_dir'])
    
    # Define the archive directory
    archive_root = os.path.join(project_root, 'archive')
    if not os.path.exists(archive_root):
        os.makedirs(archive_root)
    
    # Create a timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    
    # Create the archive folder name
    archive_folder_name = f"{model_name}_{timestamp}"
    archive_folder_path = os.path.join(archive_root, archive_folder_name)
    
    # Create the archive directory structure
    os.makedirs(os.path.join(archive_folder_path, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(archive_folder_path, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(archive_folder_path, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(archive_folder_path, 'labels', 'val'), exist_ok=True)
    
    # Copy train images
    copy_files(train_image_dir, os.path.join(archive_folder_path, 'images', 'train'))
    # Copy val images
    copy_files(val_image_dir, os.path.join(archive_folder_path, 'images', 'val'))
    # Copy train labels
    copy_files(train_label_dir, os.path.join(archive_folder_path, 'labels', 'train'))
    # Copy val labels
    copy_files(val_label_dir, os.path.join(archive_folder_path, 'labels', 'val'))
    
    print(f"Dataset archived at: {archive_folder_path}")

def copy_files(src_dir, dst_dir):
    if not os.path.exists(src_dir):
        print(f"Source directory does not exist: {src_dir}")
        return
    files = os.listdir(src_dir)
    for file in files:
        src_file = os.path.join(src_dir, file)
        dst_file = os.path.join(dst_dir, file)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)
            print(f"Copied {src_file} to {dst_file}")

def main():
    import yaml
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Prompt the user for the model name
    model_name = input("Enter the name of the model to be trained: ").strip()
    if not model_name:
        print("Model name cannot be empty.")
        return
    
    archive_dataset(config, model_name)

if __name__ == "__main__":
    main()
