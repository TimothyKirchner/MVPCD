# ~/Desktop/MVPCD/scripts/split_dataset.py

import sys
import os
import shutil
from sklearn.model_selection import train_test_split

# Adjust sys.path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def split_dataset(test_size=0.2, random_state=42):
    image_dir = os.path.join(project_root, 'data', 'images', 'train')
    label_dir = os.path.join(project_root, 'data', 'labels')
    
    # Directories for split datasets
    train_image_dir = os.path.join(project_root, 'data', 'images', 'train')
    val_image_dir = os.path.join(project_root, 'data', 'images', 'val')
    train_label_dir = os.path.join(project_root, 'data', 'labels', 'train')
    val_label_dir = os.path.join(project_root, 'data', 'labels', 'val')
    
    # Create directories if they don't exist
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
    image_files.sort()
    
    # Split the dataset
    train_files, val_files = train_test_split(image_files, test_size=test_size, random_state=random_state)
    
    # Function to copy images and labels
    def copy_files(file_list, src_image_dir, src_label_dir, dst_image_dir, dst_label_dir):
        for filename in file_list:
            # Copy image
            src_image = os.path.join(src_image_dir, filename)
            dst_image = os.path.join(dst_image_dir, filename)
            if src_image != dst_image:
                shutil.copy(src_image, dst_image)
            
            # Copy label
            label_filename = os.path.splitext(filename)[0] + '.txt'
            src_label = os.path.join(src_label_dir, label_filename)
            dst_label = os.path.join(dst_label_dir, label_filename)
            if os.path.exists(src_label):
                shutil.copy(src_label, dst_label)
            else:
                print(f"Warning: Label file {label_filename} does not exist for image {filename} in {src_image_dir} and {src_label_dir}")
    
    # Copy training files
    copy_files(train_files, image_dir, label_dir, train_image_dir, train_label_dir)
    print(f"Copied {len(train_files)} files to training set.")
    
    # Copy validation files
    copy_files(val_files, image_dir, label_dir, val_image_dir, val_label_dir)
    print(f"Copied {len(val_files)} files to validation set.")
    
    print("Dataset splitting completed.")

if __name__ == "__main__":
    split_dataset()
