# scripts/annotate.py
import cv2
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import yaml
from utils.bbox_utils import draw_bounding_boxes
from utils.annotation_utils import create_annotation

def load_config(config_path='config/config.yaml'):
    with open(os.path.join(os.path.dirname(__file__), '..', config_path), 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(config, config_path='config/config.yaml'):
    with open(os.path.join(os.path.dirname(__file__), '..', config_path), 'w') as file:
        yaml.dump(config, file)

def annotate_images(config):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    image_dir = os.path.join(project_root, config['output']['image_dir'])  # Removed 'processed'
    label_dir = os.path.join(project_root, config['output']['label_dir'])
    class_names = config.get('class_names', [])
    if not class_names:
        print("No class names defined in config.")
        return
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    for filename in os.listdir(image_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image: {image_path}")
                continue
            class_name = filename.split('_')[0]
            if class_name not in class_names:
                print(f"Class name '{class_name}' not found in config. Skipping {filename}.")
                continue
            class_id = class_names.index(class_name)
            bboxes = []
            while True:
                bbox = cv2.selectROI("Annotate Image", image, False, False)
                if bbox == (0, 0, 0, 0):
                    break
                bboxes.append(bbox)
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0,255,0), 2)
                cv2.imshow("Annotate Image", image)
            base_filename = os.path.splitext(filename)[0]
            label_path = os.path.join(label_dir, f"{base_filename}.txt")
            with open(label_path, 'w') as f:
                for bbox in bboxes:
                    annotation = create_annotation(image.shape, label_path, bbox, class_id)
                    f.write(annotation)
            cv2.destroyAllWindows()
            print(f"Annotations saved for {filename}")

if __name__ == "__main__":
    config = load_config()
    annotate_images(config)
