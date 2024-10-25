# scripts/split_dataset.py
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import shutil
import yaml
from sklearn.model_selection import train_test_split

def load_config(config_path='config/config.yaml'):
    with open(os.path.join(project_root, config_path), 'r') as file:
        return yaml.safe_load(file)

def split_dataset(config, test_size=0.2, random_state=42):
    """
    Split dataset ensuring each class has proper representation in train/val sets
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    image_dir = os.path.join(project_root, 'data', 'images')
    train_image_dir = os.path.join(project_root, 'data', 'images', 'train')
    val_image_dir = os.path.join(project_root, 'data', 'images', 'val')

    # Create directories if they don't exist
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)

    # Clear existing split directories
    for dir_path in [train_image_dir, val_image_dir]:
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # Group images by class
    class_images = {}
    for class_name in config.get('class_names', []):
        class_images[class_name] = [
            f for f in os.listdir(image_dir) 
            if f.startswith(f"{class_name}_") and f.lower().endswith(('.png', '.jpg'))
        ]
        
    # Split each class separately
    train_files = []
    val_files = []
    
    for class_name, images in class_images.items():
        if not images:
            print(f"Warning: No images found for class {class_name}")
            continue
            
        # Sort to ensure consistent sequence numbering
        images.sort()
        
        # Split while maintaining sequence
        num_val = max(1, int(len(images) * test_size))
        # Take every nth image for validation to ensure even distribution
        stride = len(images) // num_val
        val_indices = list(range(0, len(images), stride))[:num_val]
        
        class_val_files = [images[i] for i in val_indices]
        class_train_files = [img for i, img in enumerate(images) if i not in val_indices]
        
        train_files.extend(class_train_files)
        val_files.extend(class_val_files)
        
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
    copy_files(train_files, image_dir, train_image_dir)
    copy_files(val_files, image_dir, val_image_dir)

    print("\nDataset splitting completed:")
    print(f"Total training images: {len(train_files)}")
    print(f"Total validation images: {len(val_files)}")
    
    # Return split information for potential use in training
    return {
        'train_files': train_files,
        'val_files': val_files,
        'class_distribution': {
            class_name: len([f for f in train_files if f.startswith(f"{class_name}_")])
            for class_name in class_images.keys()
        }
    }

if __name__ == "__main__":
    config = load_config()
    split_dataset(config)