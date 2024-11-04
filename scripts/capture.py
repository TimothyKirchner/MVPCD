# scripts/capture.py
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import cv2
import yaml
import numpy as np
from utils.camera_utils import initialize_camera, capture_frame

def load_config(config_path='config/config.yaml'):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, config_path)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(config, config_path='config/config.yaml'):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, config_path)
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

def delete_existing_images(config, class_name):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    image_dir = os.path.join(project_root, config['output']['image_dir'])
    depth_dir = os.path.join(project_root, config['output']['depth_dir'])

    # Delete images
    for filename in os.listdir(image_dir):
        if filename.startswith(f"{class_name}_") and filename.endswith(('.png', '.jpg')):
            file_path = os.path.join(image_dir, filename)
            os.remove(file_path)
            print(f"Deleted image: {file_path}")

    # Delete corresponding depth maps
    for filename in os.listdir(depth_dir):
        if filename.startswith(f"depth_{class_name}_") and filename.endswith('.npy'):
            file_path = os.path.join(depth_dir, filename)
            os.remove(file_path)
            print(f"Deleted depth map: {file_path}")

def capture_images(config, class_name):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    image_dir = os.path.join(project_root, config['output']['image_dir'])
    depth_dir = os.path.join(project_root, config['output']['depth_dir'])

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(depth_dir):
        os.makedirs(depth_dir)

    # Overwrite existing data by deleting existing images and depth maps for the class
    delete_existing_images(config, class_name)

    # Reset image counter for the class
    config['image_counters'][class_name] = 1

    camera = initialize_camera(config)
    if camera is None:
        print("Unable to initialize the camera after multiple attempts. Exiting the program.")
        sys.exit(1)

    num_images = config['capture']['num_images']
    interval = config['capture']['interval']

    for _ in range(num_images):
        image, depth = capture_frame(camera)

        if image is None or depth is None:
            print("Failed to capture image or depth map.")
            continue

        image_number = config['image_counters'][class_name]
        image_filename = os.path.join(image_dir, f"{class_name}_{image_number}.png")
        depth_filename = os.path.join(depth_dir, f"depth_{class_name}_{image_number}.npy")

        cv2.imwrite(image_filename, image)
        np.save(depth_filename, depth)

        print(f"Captured image: {image_filename} and depth map: {depth_filename}")

        cv2.imshow("Captured Image", image)
        cv2.waitKey(int(interval * 1000))

        config['image_counters'][class_name] += 1

    camera.close()
    cv2.destroyAllWindows()
    print("Image capture completed.")

if __name__ == "__main__":
    config = load_config()
    if len(sys.argv) < 2:
        print("Usage: python capture.py [classname]")
        sys.exit(1)
    class_name = sys.argv[1]
    capture_images(config, class_name)
    save_config(config)
