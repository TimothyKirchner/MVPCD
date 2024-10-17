# ~/Desktop/MVPCD/scripts/annotate.py

import sys
import os
import cv2
import yaml

# Adjust sys.path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.annotation_utils import create_annotation

def load_config(config_path='config/config.yaml'):
    config_path = os.path.join(project_root, config_path)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def annotate_images(config):
    image_dir = os.path.join(project_root, config['output']['image_dir'], 'processed')
    label_dir = os.path.join(project_root, config['output']['label_dir'])

    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(image_dir, filename)
            base_filename = os.path.splitext(filename)[0]
            label_filename = f"{base_filename}.txt"
            label_path = os.path.join(label_dir, label_filename)

            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image: {image_path}")
                continue

            # Assuming annotations were saved during preprocessing
            if os.path.exists(label_path):
                print(f"Annotation already exists for {filename}.")
                continue
            else:
                print(f"Missing annotation for {filename}. Skipping.")
                continue

    print("Annotation process completed.")

if __name__ == "__main__":
    config = load_config()
    annotate_images(config)
