# scripts/split_dataset.py

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import shutil
import yaml

def load_config(config_path='config/config.yaml'):
    with open(os.path.join(project_root, config_path), 'r') as file:
        return yaml.safe_load(file)

def split_dataset(config, class_name, test_size=0.2):
    """
    Split dataset for a specific class, ensuring each class has proper representation in train/val sets.
    """
    image_dir = os.path.join(project_root, 'data', 'images')
    train_image_dir = os.path.join(image_dir, 'train')
    val_image_dir = os.path.join(image_dir, 'val')

    # Create directories if they don't exist
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)

    # Get images for the specific class
    images = [
        f for f in os.listdir(image_dir)
        if f.startswith(f"{class_name}_") and f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    if not images:
        print(f"Warning: No images found for class {class_name}")
        return

    # Sort to ensure consistent sequence numbering
    images.sort()

    # Split while maintaining sequence
    num_val = max(1, int(len(images) * test_size))
    # Take every nth image for validation to ensure even distribution
    stride = len(images) // num_val
    val_indices = list(range(0, len(images), stride))[:num_val]

    class_val_files = [images[i] for i in val_indices]
    class_train_files = [img for i, img in enumerate(images) if i not in val_indices]

    print(f"\nClass {class_name}:")
    print(f"Total images: {len(images)}")
    print(f"Training images: {len(class_train_files)}")
    print(f"Validation images: {len(class_val_files)}")

    def copy_files(file_list, src_dir, dst_dir):
        for filename in file_list:
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, filename)
            shutil.copy2(src_path, dst_path)

    # Copy files to their respective directories
    copy_files(class_train_files, image_dir, train_image_dir)
    copy_files(class_val_files, image_dir, val_image_dir)

    print("\nDataset splitting completed for class '{}':".format(class_name))
    print(f"Training images: {len(class_train_files)}")
    print(f"Validation images: {len(class_val_files)}")

if __name__ == "__main__":
    config = load_config()
    # You need to provide class_name when calling the function
    # For standalone testing, replace 'your_class_name' with an actual class name
    class_name = "blackthing"  # Replace with actual class name
    split_dataset(config, class_name)