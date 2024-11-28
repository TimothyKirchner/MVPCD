# scripts/delete_every_second_file.py

import os
import sys
import yaml
import logging
import argparse

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
    log_file = os.path.join(project_root, 'scripts', 'delete_every_second_file.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    logging.info("=== Starting Delete Every Second File Process ===")
    print(f"Logging initialized. Logs will be saved to '{log_file}'.")

def get_image_label_pairs(image_dir, label_dir, image_extensions):
    """
    Get a sorted list of image files and their corresponding label files.
    """
    images = [f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in image_extensions]
    images.sort()  # Sort to ensure consistent pairing

    pairs = []
    for image in images:
        base_name = os.path.splitext(image)[0]
        # Assume label files have the same base name with .txt extension
        label = base_name + '.txt'
        image_path = os.path.join(image_dir, image)
        label_path = os.path.join(label_dir, label)
        if os.path.exists(label_path):
            pairs.append((image_path, label_path))
        else:
            logging.warning(f"Label file '{label_path}' for image '{image_path}' does not exist.")
            print(f"Warning: Label file '{label_path}' for image '{image_path}' does not exist.")
    return pairs

def delete_files(pairs, start_index=1):
    """
    Delete every second file starting from start_index.
    """
    deleted_images = []
    deleted_labels = []

    # Iterate over the list with step 2, starting from start_index
    for i in range(start_index, len(pairs), 2):
        image_path, label_path = pairs[i]
        try:
            os.remove(image_path)
            deleted_images.append(image_path)
            logging.info(f"Deleted image file: {image_path}")
            print(f"Deleted image file: {image_path}")
        except Exception as e:
            logging.error(f"Failed to delete image file '{image_path}': {e}")
            print(f"Error: Failed to delete image file '{image_path}': {e}")

        try:
            os.remove(label_path)
            deleted_labels.append(label_path)
            logging.info(f"Deleted label file: {label_path}")
            print(f"Deleted label file: {label_path}")
        except Exception as e:
            logging.error(f"Failed to delete label file '{label_path}': {e}")
            print(f"Error: Failed to delete label file '{label_path}': {e}")

    return deleted_images, deleted_labels

def process_split(split, image_dir, label_dir, image_extensions):
    """
    Process a single data split (train, val, test).
    """
    print(f"\nProcessing '{split}' split:")
    logging.info(f"Processing '{split}' split.")
    
    if not os.path.exists(image_dir):
        logging.warning(f"Image directory '{image_dir}' does not exist. Skipping '{split}' split.")
        print(f"Warning: Image directory '{image_dir}' does not exist. Skipping '{split}' split.")
        return
    
    if not os.path.exists(label_dir):
        logging.warning(f"Label directory '{label_dir}' does not exist. Skipping '{split}' split.")
        print(f"Warning: Label directory '{label_dir}' does not exist. Skipping '{split}' split.")
        return

    pairs = get_image_label_pairs(image_dir, label_dir, image_extensions)
    total_pairs = len(pairs)
    if total_pairs < 2:
        logging.info(f"Not enough files to delete in '{split}' split.")
        print(f"Not enough files to delete in '{split}' split.")
        return
    
    # Delete every second file starting from index 1 (i.e., the second file)
    deleted_images, deleted_labels = delete_files(pairs, start_index=1)
    
    print(f"Deleted {len(deleted_images)} images and {len(deleted_labels)} labels from '{split}' split.")
    logging.info(f"Deleted {len(deleted_images)} images and {len(deleted_labels)} labels from '{split}' split.")

def main(add_confirmation=True):
    """
    Main function to execute the script.
    """
    parser = argparse.ArgumentParser(description="Delete every second image and corresponding label in train, val, and test folders.")
    parser.add_argument('--dry-run', action='store_true', help="Perform a dry run without deleting any files.")
    args = parser.parse_args()
    
    if args.dry_run:
        print("Performing a dry run. No files will be deleted.")
        logging.info("Performing a dry run. No files will be deleted.")
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    # Load configuration
    config = load_config()
    
    # Set up logging
    setup_logging()
    
    # Define splits
    splits = ['train', 'val', 'test']
    
    for split in splits:
        image_key = f"{split}_image_dir"
        label_key = f"{split}_label_dir"
        
        image_dir = os.path.join(project_root, config['output'].get(image_key, ''))
        label_dir = os.path.join(project_root, config['output'].get(label_key, ''))
        
        process_split(split, image_dir, label_dir, image_extensions)
    
    print("\nDeletion process completed.")
    logging.info("Deletion process completed.")

if __name__ == "__main__":
    main()