# scripts/preprocess.py
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
import yaml
from utils.chroma_key import apply_chroma_key
from utils.depth_processing import process_depth, load_depth_map
from utils.bbox_utils import convert_bbox_to_yolo, draw_bounding_boxes

def load_config(config_path='config/config.yaml'):
    with open(os.path.join(project_root, config_path), 'r') as file:
        config = yaml.safe_load(file)
    return config

def preprocess_images(config):
    splits = ['train', 'val']
    
    for split in splits:
        print(f"\n--- Processing {split} set ---")
        image_dir = os.path.join(project_root, config['output'][f'{split}_image_dir'])
        label_dir = os.path.join(project_root, config['output'][f'{split}_label_dir'])
        depth_dir = os.path.join(project_root, config['output']['depth_dir'])

        # Define debug mask directories per split
        debug_dir = os.path.join(project_root, 'data', 'debug')
        rgbmask_dir = os.path.join(debug_dir, 'rgbmask')
        depthmask_dir = os.path.join(debug_dir, 'depthmask')
        combinedmask_dir = os.path.join(debug_dir, 'combined_mask')
        contours_dir = os.path.join(debug_dir, 'contours')  # New contours directory
        bboxes_dir = os.path.join(debug_dir, 'bboxes')      # New bboxes directory

        # Create directories if they don't exist
        for directory in [image_dir, label_dir, rgbmask_dir, depthmask_dir, combinedmask_dir, contours_dir, bboxes_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        chroma_key = config.get('chroma_key', {})
        lower_color = np.array(chroma_key.get('lower_color', [0, 0, 0]), dtype=np.uint8)
        upper_color = np.array(chroma_key.get('upper_color', [179, 255, 255]), dtype=np.uint8)

        rois = config.get('rois', [])

        class_names = config.get('class_names', [])
        class_id_map = {name: idx for idx, name in enumerate(class_names)}

        areas = []
        processed_files = 0  # Counter to track processed files

        for filename in os.listdir(image_dir):
            if filename.lower().endswith('.png') or filename.lower().endswith('.jpg'):
                processed_files += 1
                print(f"\nProcessing file: {filename}")  # Debug statement
                filepath = os.path.join(image_dir, filename)
                image = cv2.imread(filepath)

                if image is None:
                    print(f"Could not load image: {filepath}")
                    continue

                img_height, img_width = image.shape[:2]

                # Apply chroma key
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                chroma_keyed = apply_chroma_key(hsv, lower_color, upper_color)

                # Load depth map
                depth_filename = filename.rsplit('.', 1)[0] + '.npy'  # Ensure correct extension
                depth_path = os.path.join(depth_dir, f"depth_{depth_filename}")
                depth_map = load_depth_map(depth_path)

                if depth_map is None:
                    print(f"Depth map not found: {depth_path}")
                    print(f"Depth map not found for {filename}. Skipping.")
                    continue

                # Determine class name from filename
                class_name = filename.split('_')[0]
                if class_name not in class_names:
                    print(f"Class '{class_name}' not defined in config. Skipping {filename}.")
                    continue

                # Get depth thresholds for the specific class
                depth_threshold = config.get('depth_thresholds', {}).get(class_name, {})
                min_depth = depth_threshold.get('min', 500)
                max_depth = depth_threshold.get('max', 2000)

                # Process depth mask
                depth_mask = process_depth(depth_map, min_depth, max_depth)
                chroma_mask = cv2.cvtColor(chroma_keyed, cv2.COLOR_BGR2GRAY)
                combined_mask = cv2.bitwise_and(chroma_mask, depth_mask)

                # Save masks in respective directories
                rgb_mask_path = os.path.join(rgbmask_dir, f"rgbmask_{filename}")
                depth_mask_path_save = os.path.join(depthmask_dir, f"depthmask_{filename}")
                combined_mask_path = os.path.join(combinedmask_dir, f"combined_mask_{filename}")
                contours_path = os.path.join(contours_dir, f"contours_{filename}.png")  # Path to save contours

                cv2.imwrite(rgb_mask_path, chroma_mask)
                cv2.imwrite(depth_mask_path_save, depth_mask)
                cv2.imwrite(combined_mask_path, combined_mask)
                print(f"Saved RGB mask: {rgb_mask_path}")
                print(f"Saved Depth mask: {depth_mask_path_save}")
                print(f"Saved Combined mask: {combined_mask_path}")

                # Apply ROIs if any
                if rois:
                    roi_mask = np.zeros_like(combined_mask)
                    for roi in rois:
                        x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
                        roi_mask[y:y+h, x:x+w] = 255
                    combined_mask = cv2.bitwise_and(combined_mask, roi_mask)

                # Find contours
                contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contours_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)
                    cv2.imwrite(contours_path, contours_image)
                    print(f"Saved contours: {contours_path}")
                else:
                    print(f"No contours found for {filename}.")

                # Extract bounding boxes from contours
                bounding_boxes = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if cv2.contourArea(contour) > 100:  # Filter out small contours
                        bounding_boxes.append((x, y, w, h))

                if not bounding_boxes:
                    print(f"No bounding boxes found within ROI for {filename}.")
                    continue

                # Select the largest bounding box based on area
                largest_bbox = max(bounding_boxes, key=lambda bbox: bbox[2] * bbox[3], default=None)
                if largest_bbox:
                    x, y, w, h = largest_bbox
                    current_area = w * h
                    areas.append(current_area)
                    average_area = sum(areas) / len(areas) if areas else 0
                    deviation_threshold = 0.2  # 20%

                    # Check for anomalous bounding boxes (optional)
                    if average_area > 0 and abs(current_area - average_area) / average_area > deviation_threshold:
                        annotated_image = draw_bounding_boxes(image.copy(), [largest_bbox], format='xywh')
                        cv2.imshow("Anomalous Bounding Box", annotated_image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                    class_id = class_id_map.get(class_name, 0)

                    # Convert (x, y, w, h) to (x_min, y_min, x_max, y_max)
                    x_min, y_min, w_bbox, h_bbox = largest_bbox
                    x_max = x_min + w_bbox
                    y_max = y_min + h_bbox
                    converted_bbox = (x_min, y_min, x_max, y_max)

                    # Debug: Print bounding box coordinates
                    print(f"Original BBox (x_min, y_min, x_max, y_max): {converted_bbox}")

                    # Convert to YOLO format
                    yolo_annotation = convert_bbox_to_yolo(img_width, img_height, converted_bbox, class_id)
                    
                    # Debug: Print YOLO annotation
                    print(f"YOLO Annotation: {yolo_annotation}")

                    # Save YOLO annotation to label file
                    base_filename = os.path.splitext(filename)[0]
                    label_path = os.path.join(label_dir, f"{base_filename}.txt")
                    with open(label_path, 'w') as f:
                        f.write(yolo_annotation)
                        print(f"Annotations saved for {label_path}")

                    # Draw bounding box on the image and save to debug/bboxes
                    bboxes_debug_path = os.path.join(bboxes_dir, f"bboxes_{filename}")
                    annotated_bbox_image = draw_bounding_boxes(image.copy(), [largest_bbox], format='xywh')
                    cv2.imwrite(bboxes_debug_path, annotated_bbox_image)
                    print(f"Saved bounding box image: {bboxes_debug_path}")

    print(f"\nPreprocessing completed. Total files processed: {processed_files}")

if __name__ == "__main__":
    config = load_config()
    preprocess_images(config)
