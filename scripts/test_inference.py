# scripts/test_inference.py
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import cv2
import yaml
from ultralytics import YOLO

def load_config(config_path='config/config.yaml'):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, config_path)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def test_inference(config):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(project_root, 'runs', 'yolov8_model5', 'weights', 'best.pt')
    if not os.path.exists(model_path):
        print(f"Trained model not found at {model_path}. Please ensure the model has been trained.")
        return

    model = YOLO(model_path)

    test_images_dir = os.path.join(project_root, 'data', 'test_images')
    if not os.path.exists(test_images_dir):
        print(f"Test images directory not found at {test_images_dir}. Please create it and add test images.")
        return

    for filename in os.listdir(test_images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(test_images_dir, filename)
            image = cv2.imread(filepath)
            if image is None:
                print(f"Could not load image: {filepath}")
                continue

            results = model(image, verbose=False)
            annotated_frame = results[0].plot()

            cv2.imshow(f"Inference on {filename}", annotated_frame)
            cv2.waitKey(0)  # Press any key to proceed to the next image
            cv2.destroyAllWindows()

if __name__ == "__main__":
    config = load_config()
    test_inference(config)
