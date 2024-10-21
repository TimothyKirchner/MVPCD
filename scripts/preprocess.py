# ~/Desktop/MVPCD/scripts/preprocess.py

import sys
import os
import cv2
import numpy as np
import yaml

# Adjust sys.path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.chroma_key import apply_chroma_key
from utils.depth_processing import process_depth, load_depth_map
from utils.bbox_utils import convert_bbox_to_yolo, draw_bounding_boxes

def load_config(config_path='config/config.yaml'):
    config_path = os.path.join(project_root, config_path)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def preprocess_images(config):
    image_dir = os.path.join(project_root, config['output']['image_dir'])
    train_dir = os.path.join(project_root, 'data', 'images', 'train')
    depth_dir = os.path.join(project_root, config['output']['depth_dir'])
    label_dir = os.path.join(project_root, 'data', 'labels', 'train')
    debug_dir = os.path.join(project_root, 'data', 'debug')  # Add debug directory

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    if not os.path.exists(debug_dir):  # Create debug directory if it doesn't exist
        os.makedirs(debug_dir)

    min_depth = config.get('depth_threshold', {}).get('min', 500)
    max_depth = config.get('depth_threshold', {}).get('max', 2000)
    lower_color = np.array(config.get('chroma_key', {}).get('lower_color', [0, 0, 0]), dtype=np.uint8)
    upper_color = np.array(config.get('chroma_key', {}).get('upper_color', [179, 255, 255]), dtype=np.uint8)

    if 'rois' in config and len(config['rois']) > 0:
        roi = config['rois'][0]
    else:
        roi = None

    areas = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            filepath = os.path.join(image_dir, filename)
            image = cv2.imread(filepath)

            if image is None:
                print(f"Could not load image: {filepath}")
                continue

            # Save the unprocessed image into train_dir
            unprocessed_path = os.path.join(train_dir, filename)
            cv2.imwrite(unprocessed_path, image)
            print(f"Unprocessed image saved: {unprocessed_path}")

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            chroma_keyed = apply_chroma_key(hsv, lower_color, upper_color)

            # Correct depth filename matching
            # Remove 'processed_' prefix if present
            base_filename = filename.replace('processed_', '')
            depth_filename = base_filename.replace('image_', 'depth_').replace('.png', '.npy').replace('.jpg', '.npy')
            depth_path = os.path.join(depth_dir, depth_filename)
            depth_map = load_depth_map(depth_path)

            if depth_map is None:
                print(f"Depth map not found: {depth_path}")
                print(f"Depth map not found for {filename}. Skipping.")
                continue

            depth_mask = process_depth(depth_map, min_depth, max_depth)
            chroma_mask = cv2.cvtColor(chroma_keyed, cv2.COLOR_BGR2GRAY)
            combined_mask = cv2.bitwise_and(chroma_mask, depth_mask)

            # Save combined mask to debug folder
            debug_mask_path = os.path.join(debug_dir, f"mask_{filename}")
            cv2.imwrite(debug_mask_path, combined_mask)
            print(f"Combined mask saved: {debug_mask_path}")

            # Apply ROI mask if available
            if roi:
                roi_mask = np.zeros_like(combined_mask)
                x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
                roi_mask[y:y+h, x:x+w] = 255
                combined_mask = cv2.bitwise_and(combined_mask, roi_mask)

            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bounding_boxes = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if cv2.contourArea(contour) > 100:
                    bounding_boxes.append((x, y, w, h))

            # Filter bounding boxes within ROI
            if roi:
                def is_within_roi(bbox, roi):
                    x, y, w, h = bbox
                    return (x >= roi['x'] and y >= roi['y'] and
                            x + w <= roi['x'] + roi['width'] and
                            y + h <= roi['y'] + roi['height'])
                bounding_boxes = [bbox for bbox in bounding_boxes if is_within_roi(bbox, roi)]

            if not bounding_boxes:
                print(f"No bounding boxes found within ROI for {filename}.")
                continue

            # Keep only the largest bounding box
            largest_bbox = max(bounding_boxes, key=lambda bbox: bbox[2] * bbox[3], default=None)
            if largest_bbox:
                x, y, w, h = largest_bbox
                current_area = w * h
                areas.append(current_area)
                average_area = sum(areas) / len(areas)
                deviation_threshold = 0.2  # 20% deviation

                if abs(current_area - average_area) / average_area > deviation_threshold:
                    print(f"Bounding box area {current_area} deviates significantly from average {average_area}.")
                    # Optionally, display image for manual review
                    annotated_image = draw_bounding_boxes(image.copy(), [largest_bbox])
                    cv2.imshow("Anomalous Bounding Box", annotated_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                # Save YOLO annotation with processed image filename
                yolo_annotation = convert_bbox_to_yolo(image.shape, largest_bbox)
                base_filename_no_ext = os.path.splitext(filename)[0]
                processed_base_filename = f"processed_{base_filename_no_ext}"
                label_path = os.path.join(label_dir, f"{processed_base_filename}.txt")
                with open(label_path, 'w') as f:
                    f.write(yolo_annotation)
                    print(f"Annotations saved for {label_path}")

                # Save processed image with 'processed_' prefix
                processed_filename = f'processed_{filename}'
                processed_path = os.path.join(train_dir, processed_filename)
                cv2.imwrite(processed_path, chroma_keyed)
                print(f"Preprocessed image saved: {processed_path}")

    cv2.destroyAllWindows()
    print("Preprocessing completed.")

if __name__ == "__main__":
    config = load_config()
    preprocess_images(config)
