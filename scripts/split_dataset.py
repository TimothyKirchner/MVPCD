# scripts/split_dataset.py
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(test_size=0.2, random_state=42):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    image_dir = os.path.join(project_root, 'data', 'images')  # Original images
    train_image_dir = os.path.join(project_root, 'data', 'images', 'train')
    val_image_dir = os.path.join(project_root, 'data', 'images', 'val')

    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg'))]
    image_files.sort()

    if not image_files:
        print("No images found in 'data/images/'. Please capture images first.")
        return

    train_files, val_files = train_test_split(image_files, test_size=test_size, random_state=random_state)

    def copy_files(file_list, src_image_dir, dst_image_dir):
        for filename in file_list:
            src_image = os.path.join(src_image_dir, filename)
            dst_image = os.path.join(dst_image_dir, filename)
            shutil.copy(src_image, dst_image)

    copy_files(train_files, image_dir, train_image_dir)
    print(f"Copied {len(train_files)} files to training set.")

    copy_files(val_files, image_dir, val_image_dir)
    print(f"Copied {len(val_files)} files to validation set.")

    print("Dataset splitting completed.")

if __name__ == "__main__":
    split_dataset()
