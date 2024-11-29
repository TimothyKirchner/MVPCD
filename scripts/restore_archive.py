# scripts/restore_archive.py
import os
import shutil

# Adjust sys.path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import sys
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def restore_archive(add_list, remove_list, archive_path):
    """
    Restore an archived dataset by copying its contents into the project's data directories.

    Parameters:
    - add_list (list): List of classes to add from the archive.
    - remove_list (list): List of classes to remove from the current dataset.
    - archive_path (str): Path to the archived dataset folder.
    """
    if not os.path.exists(archive_path):
        print(f"Archive path '{archive_path}' does not exist.")
        return

    # Define the target directories
    target_dirs = {
        'images_train': 'data/images/train',
        'labels_train': 'data/labels/train',
        'images_val': 'data/images/val',
        'labels_val': 'data/labels/val',
        'images_test': 'data/images/test',
        'labels_test': 'data/labels/test',
        'backgrounds': 'data/backgrounds'  # Added background images directory
    }

    # Ensure target directories exist
    for dir_key, dir_path in target_dirs.items():
        full_path = os.path.join(project_root, dir_path)
        os.makedirs(full_path, exist_ok=True)

    # Restore background images
    src_backgrounds = os.path.join(archive_path, 'data', 'backgrounds')
    dest_backgrounds = os.path.join(project_root, 'data', 'backgrounds')

    if os.path.exists(src_backgrounds):
        shutil.copytree(src_backgrounds, dest_backgrounds, dirs_exist_ok=True)
        print("Background images restored.")
    else:
        print(f"Background images not found in archive at '{src_backgrounds}'.")

    # Function to copy files for a specific class
    def copy_files(class_name, split):
        src_images = os.path.join(archive_path, 'data', 'images', split, class_name)
        src_labels = os.path.join(archive_path, 'data', 'labels', split, class_name)
        dest_images = os.path.join(project_root, 'data', 'images', split, class_name)
        dest_labels = os.path.join(project_root, 'data', 'labels', split, class_name)

        # Create destination class directories
        os.makedirs(dest_images, exist_ok=True)
        os.makedirs(dest_labels, exist_ok=True)

        # Copy image files
        if os.path.exists(src_images):
            for file in os.listdir(src_images):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    shutil.copy2(os.path.join(src_images, file), os.path.join(dest_images, file))
        else:
            print(f"Source images directory '{src_images}' does not exist.")

        # Copy label files
        if os.path.exists(src_labels):
            for file in os.listdir(src_labels):
                if file.lower().endswith('.txt'):
                    shutil.copy2(os.path.join(src_labels, file), os.path.join(dest_labels, file))
        else:
            print(f"Source labels directory '{src_labels}' does not exist.")

    # Add classes from the archive
    for class_name in add_list:
        print(f"Restoring class '{class_name}' from archive...")
        for split in ['train', 'val', 'test']:
            copy_files(class_name, split)
        print(f"Class '{class_name}' restored successfully.")

    # Remove classes if needed
    for class_name in remove_list:
        print(f"Removing class '{class_name}' from current dataset...")
        for split in ['train', 'val', 'test']:
            images_dir = os.path.join(project_root, 'data', 'images', split, class_name)
            labels_dir = os.path.join(project_root, 'data', 'labels', split, class_name)
            if os.path.exists(images_dir):
                shutil.rmtree(images_dir)
                print(f"Deleted images for class '{class_name}' in '{split}'.")
            if os.path.exists(labels_dir):
                shutil.rmtree(labels_dir)
                print(f"Deleted labels for class '{class_name}' in '{split}'.")
        print(f"Class '{class_name}' removed successfully.")

    print("Archived dataset has been successfully restored.")