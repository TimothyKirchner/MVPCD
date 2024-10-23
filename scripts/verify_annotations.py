# scripts/verify_annotations.py
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import cv2
import yaml
from utils.bbox_utils import draw_bounding_boxes

def load_config(config_path='config/config.yaml'):
    with open(os.path.join(os.path.dirname(__file__), '..', config_path), 'r') as file:
        config = yaml.safe_load(file)
    return config

def verify_annotations(config):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    image_dir = os.path.join(project_root, config['output']['image_dir'])
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

            if not os.path.exists(label_path):
                print(f"No label found for {filename}. Skipping.")
                continue

            # Read bounding boxes from label file
            bboxes = []
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"Invalid label format in {label_path}. Skipping this label.")
                        bboxes = []
                        break
                    class_id, x_center, y_center, width, height = map(float, parts)
                    # Convert YOLO format to pixel coordinates
                    img_height, img_width = image.shape[:2]
                    x_center *= img_width
                    y_center *= img_height
                    width *= img_width
                    height *= img_height
                    x = int(x_center - width / 2)
                    y = int(y_center - height / 2)
                    w = int(width)
                    h = int(height)
                    bboxes.append((x, y, w, h))

            if not bboxes:
                print(f"No valid bounding boxes for {filename}. Skipping.")
                continue

            # Initialize window
            cv2.namedWindow("Verify Annotations")

            for idx, bbox in enumerate(bboxes):
                annotated_image = draw_bounding_boxes(image.copy(), [bbox], color=(0, 255, 0), thickness=2)
                cv2.imshow("Verify Annotations", annotated_image)
                print(f"\nReviewing bounding box {idx+1}/{len(bboxes)} for: {filename}")
                print("Use the following keys:")
                print("'n' - Next bounding box")
                print("'d' - Delete entire label")
                print("'q' - Quit verification")

                key = cv2.waitKey(0) & 0xFF
                if key == ord('n'):
                    continue
                elif key == ord('d'):
                    os.remove(label_path)
                    print(f"Annotations for {filename} deleted.")
                    break
                elif key == ord('q'):
                    print("Verification process exited.")
                    cv2.destroyAllWindows()
                    sys.exit(0)
                else:
                    print("Invalid key pressed. Use 'n' to proceed, 'd' to delete, or 'q' to quit.")
            
            cv2.destroyAllWindows()

if __name__ == "__main__":
    config = load_config()
    verify_annotations(config)
