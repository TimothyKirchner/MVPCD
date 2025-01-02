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

def split_dataset(config, class_name, test_size=0.1):
    """
    Split dataset for a specific class, ensuring each class has proper representation in train/val/test sets.
    Split ratio: 80% train, 10% val, 10% test
    """
    image_dir = os.path.join(project_root, config['output']['image_dir'])
    train_image_dir = os.path.join(project_root, config['output']['train_image_dir'])
    val_image_dir = os.path.join(project_root, config['output']['val_image_dir'])
    test_image_dir = os.path.join(project_root, config['output']['test_image_dir'])  # Added Test Image Directory

    train_label_dir = os.path.join(project_root, config['output']['train_label_dir'])
    val_label_dir = os.path.join(project_root, config['output']['val_label_dir'])
    test_label_dir = os.path.join(project_root, config['output']['test_label_dir'])    # Added Test Label Directory

    # Create directories if they don't exist
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(test_image_dir, exist_ok=True)    # Create Test Image Directory
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)    # Create Test Label Directory

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

    total_images = len(images)
    num_train = int(total_images * 0.8)
    num_val = int(total_images * 0.1)
    num_test = total_images - num_train - num_val  # Ensure all images are used

    train_files = images[:num_train]
    val_files = images[num_train:num_train + num_val]
    test_files = images[num_train + num_val:]

    print(f"\nClass {class_name}:")
    print(f"Total images: {total_images}")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")
    print(f"Test images: {len(test_files)}")

    def copy_files(file_list, src_dir, dst_dir):
        for filename in file_list:
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, filename)
            shutil.copy2(src_path, dst_path)

    # Copy files to their respective directories
    copy_files(train_files, image_dir, train_image_dir)
    copy_files(val_files, image_dir, val_image_dir)
    copy_files(test_files, image_dir, test_image_dir)    # Copy to Test Image Directory

    # Similarly, copy label files
    def copy_label_files(file_list, src_label_dir, dst_label_dir):
        for filename in file_list:
            base_filename = os.path.splitext(filename)[0]
            label_filename = base_filename + '.txt'
            src_label_path = os.path.join(src_label_dir, label_filename)
            dst_label_path = os.path.join(dst_label_dir, label_filename)
            if os.path.exists(src_label_path):
                shutil.copy2(src_label_path, dst_label_path)

    copy_label_files(train_files, image_dir, train_label_dir)
    copy_label_files(val_files, image_dir, val_label_dir)
    copy_label_files(test_files, image_dir, test_label_dir)    # Copy to Test Label Directory

    print("\nDataset splitting completed for class '{}':".format(class_name))
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")
    print(f"Test images: {len(test_files)}")

if __name__ == "__main__":
    config = load_config()
    # You need to provide class_name when calling the function
    # For standalone testing, replace 'your_class_name' with an actual class name
    class_name = "blackthing"  # Replace with actual class name
    split_dataset(config, class_name)