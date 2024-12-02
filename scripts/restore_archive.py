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

    # Define the source and destination directories
    src_data_dir = os.path.join(archive_path, 'data')
    dest_data_dir = os.path.join(project_root, 'data')

    # Ensure the destination data directory exists
    os.makedirs(dest_data_dir, exist_ok=True)

    # Copy the entire data directory from the archive to the project
    try:
        # Copy images
        src_images_dir = os.path.join(src_data_dir, 'images')
        dest_images_dir = os.path.join(dest_data_dir, 'images')
        if os.path.exists(src_images_dir):
            shutil.copytree(src_images_dir, dest_images_dir, dirs_exist_ok=True)
            print("Images restored from archive.")
        else:
            print(f"No images found in archive at '{src_images_dir}'.")

        # Copy labels
        src_labels_dir = os.path.join(src_data_dir, 'labels')
        dest_labels_dir = os.path.join(dest_data_dir, 'labels')
        if os.path.exists(src_labels_dir):
            shutil.copytree(src_labels_dir, dest_labels_dir, dirs_exist_ok=True)
            print("Labels restored from archive.")
        else:
            print(f"No labels found in archive at '{src_labels_dir}'.")

        # Copy backgrounds
        src_backgrounds_dir = os.path.join(src_data_dir, 'backgrounds')
        dest_backgrounds_dir = os.path.join(dest_data_dir, 'backgrounds')
        if os.path.exists(src_backgrounds_dir):
            shutil.copytree(src_backgrounds_dir, dest_backgrounds_dir, dirs_exist_ok=True)
            print("Background images restored.")
            background_images_restored = True
        else:
            print(f"Background images not found in archive at '{src_backgrounds_dir}'.")
            background_images_restored = False

    except Exception as e:
        print(f"Error while copying data from archive: {e}")
        return

    # Remove classes if needed
    for class_name in remove_list:
        print(f"Removing class '{class_name}' from current dataset...")
        for split in ['train', 'val', 'test']:
            images_dir = os.path.join(dest_data_dir, 'images', split)
            labels_dir = os.path.join(dest_data_dir, 'labels', split)
            # Delete images and labels for the class
            if os.path.exists(images_dir):
                for file in os.listdir(images_dir):
                    if file.startswith(class_name + '_'):
                        os.remove(os.path.join(images_dir, file))
            if os.path.exists(labels_dir):
                for file in os.listdir(labels_dir):
                    if file.startswith(class_name + '_'):
                        os.remove(os.path.join(labels_dir, file))
        print(f"Class '{class_name}' removed successfully.")

    # If add_list is empty, assume all classes are to be added
    if not add_list:
        # Get all class names from the restored data
        classes_in_data = set()
        for split in ['train', 'val', 'test']:
            images_dir = os.path.join(dest_data_dir, 'images', split)
            if os.path.exists(images_dir):
                for file in os.listdir(images_dir):
                    if '_' in file:
                        class_name = file.split('_')[0]
                        classes_in_data.add(class_name)
        add_list = list(classes_in_data)

    # Update mvpcd.yaml
    mvpcd_yaml_path = os.path.join(project_root, 'data', 'mvpcd.yaml')
    try:
        with open(mvpcd_yaml_path, 'w') as file:
            import yaml
            yaml.dump({
                'train': {
                    'images': './data/images/train',
                    'labels': './data/labels/train'
                },
                'val': {
                    'images': './data/images/val',
                    'labels': './data/labels/val'
                },
                'test': {
                    'images': './data/images/test',
                    'labels': './data/labels/test'
                },
                'nc': len(add_list),
                'names': add_list
            }, file)
        print("Updated 'mvpcd.yaml' with restored classes.")
    except Exception as e:
        print(f"Failed to update 'mvpcd.yaml': {e}")

    print("Archived dataset has been successfully restored.")

    return background_images_restored  # Return whether background images were restored