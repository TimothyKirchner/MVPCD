# scripts/restore_archive.py
import os
import shutil
import yaml  # For updating mvpcd.yaml

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
        return False

    # Define the source and destination directories
    src_data_dir = os.path.join(archive_path, 'data')
    dest_data_dir = os.path.join(project_root, 'data')

    # Ensure the destination data directory exists
    os.makedirs(dest_data_dir, exist_ok=True)

    # Copy background images
    src_backgrounds_dir = os.path.join(src_data_dir, 'backgrounds')
    dest_backgrounds_dir = os.path.join(dest_data_dir, 'backgrounds')
    background_images_restored = False
    if os.path.exists(src_backgrounds_dir):
        if os.path.exists(dest_backgrounds_dir):
            shutil.rmtree(dest_backgrounds_dir)
        shutil.copytree(src_backgrounds_dir, dest_backgrounds_dir)
        print("Background images restored.")
        background_images_restored = True
    else:
        print(f"Background images not found in archive at '{src_backgrounds_dir}'.")

    # Function to copy images and labels for classes
    def copy_class_files(split):
        src_images_split_dir = os.path.join(src_data_dir, 'images', split)
        src_labels_split_dir = os.path.join(src_data_dir, 'labels', split)
        dest_images_split_dir = os.path.join(dest_data_dir, 'images', split)
        dest_labels_split_dir = os.path.join(dest_data_dir, 'labels', split)
        os.makedirs(dest_images_split_dir, exist_ok=True)
        os.makedirs(dest_labels_split_dir, exist_ok=True)

        # Get list of classes in the archive split
        if not os.path.exists(src_images_split_dir):
            return
        classes_in_split = set()
        for file in os.listdir(src_images_split_dir):
            if '_' in file:
                class_name = file.split('_')[0]
                classes_in_split.add(class_name)

        # If add_list is empty, we copy all classes
        classes_to_add = add_list if add_list else classes_in_split

        for file in os.listdir(src_images_split_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                class_name = file.split('_')[0]
                if class_name in classes_to_add:
                    # Copy image
                    shutil.copy2(os.path.join(src_images_split_dir, file), dest_images_split_dir)
                    # Copy corresponding label if exists
                    label_file = os.path.splitext(file)[0] + '.txt'
                    src_label_file = os.path.join(src_labels_split_dir, label_file)
                    if os.path.exists(src_label_file):
                        shutil.copy2(src_label_file, dest_labels_split_dir)

    # Copy files for each split
    for split in ['train', 'val', 'test']:
        copy_class_files(split)

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

    # Get the final list of classes
    all_classes = set()
    for split in ['train', 'val', 'test']:
        images_dir = os.path.join(dest_data_dir, 'images', split)
        if os.path.exists(images_dir):
            for file in os.listdir(images_dir):
                if '_' in file:
                    class_name = file.split('_')[0]
                    all_classes.add(class_name)
    all_classes = sorted(all_classes)

    # Update mvpcd.yaml
    mvpcd_yaml_path = os.path.join(dest_data_dir, 'mvpcd.yaml')
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
                'test': {
                    'images': './data/images/test',
                    'labels': './data/labels/test'
                },
                'nc': len(all_classes),
                'names': list(all_classes)
            }, file)
        print("Updated 'mvpcd.yaml' with restored classes.")
    except Exception as e:
        print(f"Failed to update 'mvpcd.yaml': {e}")

    print("Archived dataset has been successfully restored.")

    return background_images_restored  # Return whether background images were restored