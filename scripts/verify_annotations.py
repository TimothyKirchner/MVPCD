# ~/Desktop/MVPCD/scripts/verify_annotations.py

import sys
import os
import cv2
import yaml

# Adjust sys.path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.bbox_utils import draw_bounding_boxes

def load_config(config_path='config/config.yaml'):
    config_path = os.path.join(project_root, config_path)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def verify_annotations(config):
    image_dir = os.path.join(project_root, config['output']['image_dir'], 'processed')
    label_dir = os.path.join(project_root, config['output']['label_dir'])

    if not os.path.exists(label_dir):
        print(f"Label directory does not exist: {label_dir}")
        return

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

            bboxes = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"Invalid label line skipped: {line}")
                        continue
                    class_id, x_center, y_center, width, height = map(float, parts)
                    x_center *= image.shape[1]
                    y_center *= image.shape[0]
                    width *= image.shape[1]
                    height *= image.shape[0]
                    x = int(x_center - width / 2)
                    y = int(y_center - height / 2)
                    w = int(width)
                    h = int(height)
                    bboxes.append((x, y, w, h))

            annotated_image = draw_bounding_boxes(image.copy(), bboxes)
            cv2.imshow("Verify Annotations", annotated_image)
            print(f"Reviewing annotations for: {filename}")
            print("Press 's' to save, 'd' to delete, or 'q' to quit.")

            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()

            if key == ord('s'):
                print("Annotations saved.")
            elif key == ord('d') and os.path.exists(label_path):
                os.remove(label_path)
                print("Annotations deleted.")
            elif key == ord('q'):
                print("Verification process exited.")
                break

    cv2.destroyAllWindows()
    print("Verification completed.")

if __name__ == "__main__":
    config = load_config()
    verify_annotations(config)
