# scripts/preprocess.py

import sys
import os
import argparse
import cv2
import numpy as np
import yaml
import random

# Adjust sys.path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.chroma_key import apply_chroma_key
from utils.depth_processing import process_depth, load_depth_map

def load_config(config_path='config/config.yaml'):
    """
    Load the YAML configuration file.
    """
    with open(os.path.join(project_root, config_path), 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_arguments():
    parser = argparse.ArgumentParser(description="Preprocess images for YOLOv8 training.")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['bbox', 'segmentation'],
        default='bbox',
        help="Choose label generation mode: 'bbox' for bounding boxes only or 'segmentation' for bounding boxes and masks."
    )
    return parser.parse_args()

def get_roi_mask(image_shape, roi):
    """
    Create a binary mask indicating the ROI area.

    Parameters:
    - image_shape: Tuple[int, int, int], shape of the image (height, width, channels).
    - roi: Dict[str, int], with keys 'x', 'y', 'width', 'height'.

    Returns:
    - roi_mask: np.ndarray, binary mask with the same height and width as the image.
    """
    height, width = image_shape[:2]
    roi_mask = np.zeros((height, width), dtype=np.uint8)

    x_start = int(roi['x'])
    y_start = int(roi['y'])
    x_end = x_start + int(roi['width'])
    y_end = y_start + int(roi['height'])

    # Ensure ROI coordinates are within image bounds
    x_start = max(0, min(x_start, width - 1))
    y_start = max(0, min(y_start, height - 1))
    x_end = max(0, min(x_end, width))
    y_end = max(0, min(y_end, height))

    roi_mask[y_start:y_end, x_start:x_end] = 1

    return roi_mask

def delete_files_for_image(filename, config):
    """
    Delete all files corresponding to a given image.
    """
    base_filename = os.path.splitext(filename)[0]
    label_filename = base_filename + '.txt'

    # Directories
    image_dirs = [
        os.path.join(project_root, config['output']['image_dir']),
        os.path.join(project_root, config['output']['train_image_dir']),
        os.path.join(project_root, config['output']['val_image_dir'])
    ]
    label_dirs = [
        os.path.join(project_root, config['output']['label_dir']),
        os.path.join(project_root, config['output']['train_label_dir']),
        os.path.join(project_root, config['output']['val_label_dir'])
    ]
    debug_dirs = [os.path.join(project_root, d) for d in config.get('debug', {}).values()]

    # Delete images
    for dir_path in image_dirs:
        file_path = os.path.join(dir_path, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted image file: {file_path}")

    # Delete labels
    for dir_path in label_dirs:
        file_path = os.path.join(dir_path, label_filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted label file: {file_path}")

    # Delete debug images
    for dir_path in debug_dirs:
        debug_files = [
            os.path.join(dir_path, f"{prefix}_{filename}")
            for prefix in ['rgbmask', 'depthmask', 'combined_mask', 'contours', 'bboxes', 'maskinyolo', 'maskinyolo_visualization']
        ]
        for file_path in debug_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted debug file: {file_path}")

def preprocess_images(config, processedimages, counter, mode, class_name):
    splits = ['train', 'val']

    # Prompt user to check bounding box sizes
    check_bbox_size = False
    while True:
        user_input = input("Do you want to check for significantly larger or smaller bounding boxes? (y/n): ").strip().lower()
        if user_input == 'y':
            check_bbox_size = True
            break
        elif user_input == 'n':
            check_bbox_size = False
            break
        else:
            print("Please enter 'y' or 'n'.")

    bbox_areas = []

    for split in splits:
        print(f"\n--- Processing {split} set for class '{class_name}' ---")
        image_dir = os.path.join(project_root, config['output'][f"{split}_image_dir"])
        label_dir = os.path.join(project_root, config['output'][f"{split}_label_dir"])
        depth_dir = os.path.join(project_root, config['output']['depth_dir'])

        # Define debug mask directories per split
        debug_dir = os.path.join(project_root, 'data', 'debug')
        rgbmask_dir = os.path.join(debug_dir, 'rgbmask')
        depthmask_dir = os.path.join(debug_dir, 'depthmask')
        combinedmask_dir = os.path.join(debug_dir, 'combined_mask')
        contours_dir = os.path.join(debug_dir, 'contours')
        bboxes_dir = os.path.join(debug_dir, 'bboxes')
        maskinyolo_dir = os.path.join(debug_dir, 'maskinyolo')

        # Create directories if they don't exist
        for directory in [label_dir, rgbmask_dir, depthmask_dir, combinedmask_dir, contours_dir, bboxes_dir, maskinyolo_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        chroma_key = config.get('chroma_key', {})
        lower_color = np.array(chroma_key.get('lower_color', [0, 0, 0]), dtype=np.uint8)
        upper_color = np.array(chroma_key.get('upper_color', [179, 255, 255]), dtype=np.uint8)

        # Fixed ROIs from config
        rois_config = config.get('rois', [])

        if not rois_config:
            print("Error: 'rois' section is missing or empty in the configuration file.")
            sys.exit(1)

        class_names = config.get('class_names', [])
        class_id_map = {name: idx for idx, name in enumerate(class_names)}

        processed_files = 0

        print(f"Image directory: {image_dir}")

        # Get list of image files
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f.startswith(class_name)]

        for filename in image_files:
            skip_image = False
            if filename in processedimages:
                continue
            processed_files += 1
            print(f"\nProcessing file: {filename}")
            filepath = os.path.join(image_dir, filename)
            image = cv2.imread(filepath)

            if image is None:
                print(f"Could not load image: {filepath}")
                continue

            img_height, img_width = image.shape[:2]

            # Use the first ROI from the list
            roi_config = rois_config[0]

            # Generate ROI mask
            roi_mask = get_roi_mask(image.shape, roi_config)

            # Apply chroma key on the image
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            chroma_keyed = apply_chroma_key(hsv, lower_color, upper_color)

            # Load depth map
            depth_filename = filename.rsplit('.', 1)[0] + '.npy'
            depth_path = os.path.join(depth_dir, f"depth_{depth_filename}")
            depth_map = load_depth_map(depth_path)

            if depth_map is None:
                print(f"Depth map not found: {depth_path}")
                print(f"Depth map not found for {filename}. Skipping.")
                continue

            # Get depth thresholds for the specific class
            depth_threshold = config.get('depth_thresholds', {}).get(class_name, {})
            min_depth = depth_threshold.get('min', 500)
            max_depth = depth_threshold.get('max', 2000)

            # Process depth mask
            depth_mask = process_depth(depth_map, min_depth=min_depth, max_depth=max_depth)
            chroma_mask = cv2.cvtColor(chroma_keyed, cv2.COLOR_BGR2GRAY)
            combined_mask = cv2.bitwise_and(chroma_mask, depth_mask)

            # Threshold the combined mask to ensure it's binary
            _, binary_mask = cv2.threshold(combined_mask, 1, 255, cv2.THRESH_BINARY)

            # Apply ROI mask
            roi_mask_scaled = roi_mask * 255
            final_mask = cv2.bitwise_and(binary_mask, roi_mask_scaled)

            # Save masks
            rgb_mask_path = os.path.join(rgbmask_dir, f"rgbmask_{filename}")
            depth_mask_path_save = os.path.join(depthmask_dir, f"depthmask_{filename}")
            combined_mask_path = os.path.join(combinedmask_dir, f"combined_mask_{filename}")
            contours_path = os.path.join(contours_dir, f"contours_{filename}")

            cv2.imwrite(rgb_mask_path, chroma_mask)
            cv2.imwrite(depth_mask_path_save, depth_mask)
            cv2.imwrite(combined_mask_path, combined_mask)
            print(f"Saved RGB mask: {rgb_mask_path}")
            print(f"Saved Depth mask: {depth_mask_path_save}")
            print(f"Saved Combined mask: {combined_mask_path}")

            # Find contours on the final mask
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contours_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)
                cv2.imwrite(contours_path, contours_image)
                print(f"Saved contours: {contours_path}")
            else:
                print(f"No contours found for {filename}.")
                if mode == 'segmentation':
                    continue

            # Initialize annotations
            annotations = []
            # Prepare mask visualization (optional)
            mask_visualization = image.copy()
            annotated_image = image.copy()  # For saving images with bounding boxes

            image_bbox_areas = []

            # Process each contour
            for contour in contours:
                if cv2.contourArea(contour) < 100:
                    continue

                x, y, w, h = cv2.boundingRect(contour)

                area = w * h

                # Only consider areas where w and h are positive
                if w <= 0 or h <= 0:
                    continue

                image_bbox_areas.append(area)

                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width_norm = w / img_width
                height_norm = h / img_height

                if mode == 'segmentation':
                    contour_pts = contour.reshape(-1, 2)
                    segmentation = contour_pts.astype(np.float32)
                    segmentation[:, 0] /= img_width
                    segmentation[:, 1] /= img_height
                    segmentation = segmentation.flatten().tolist()

                    class_id = class_id_map.get(class_name, 0)
                    annotation = [class_id, x_center, y_center, width_norm, height_norm] + segmentation

                    # Draw segmentation mask
                    points = (contour_pts).astype(np.int32)
                    cv2.polylines(mask_visualization, [points], isClosed=True, color=(0, 0, 255), thickness=2)
                    cv2.fillPoly(mask_visualization, [points], color=(0, 0, 255))
                else:
                    class_id = class_id_map.get(class_name, 0)
                    annotation = [class_id, x_center, y_center, width_norm, height_norm]

                annotations.append(annotation)

                # Draw bounding box on mask_visualization and annotated_image
                cv2.rectangle(mask_visualization, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if not annotations:
                print(f"No valid annotations for {filename}. Skipping label file generation.")
                continue

            # Check bounding box sizes
            if check_bbox_size and bbox_areas:
                avg_area = sum(bbox_areas) / len(bbox_areas)
                threshold = 0.5  # 50% difference

                significant_diff = False
                for area in image_bbox_areas:
                    if area > avg_area * (1 + threshold) or area < avg_area * (1 - threshold):
                        significant_diff = True
                        break

                if significant_diff:
                    # Show image with bounding boxes
                    cv2.imshow('Bounding Box Check', mask_visualization)
                    print(f"One or more bounding boxes have area significantly different from average {avg_area:.2f}.")
                    print("Close the image window to proceed.")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    # Ask user whether to keep or delete
                    while True:
                        user_choice = input("Do you want to keep this image? (y/n): ").strip().lower()
                        if user_choice == 'y':
                            # Keep it
                            break
                        elif user_choice == 'n':
                            # Delete the files corresponding to this image
                            delete_files_for_image(filename, config)
                            # Skip processing this image
                            skip_image = True
                            break
                        else:
                            print("Please enter 'y' or 'n'.")
                    if skip_image:
                        break  # Skip saving annotations and images

            if skip_image:
                continue  # Skip saving annotations and images

            # Add areas to bbox_areas
            bbox_areas.extend(image_bbox_areas)

            # Save annotations
            base_filename = os.path.splitext(filename)[0]
            label_path = os.path.join(label_dir, f"{base_filename}.txt")
            with open(label_path, 'w') as f:
                for annotation in annotations:
                    annotation_str = ' '.join([f"{a:.6f}" if isinstance(a, float) else str(a) for a in annotation])
                    f.write(annotation_str + '\n')
            print(f"Annotations saved for {label_path}")

            # Save image
            image_save_path = os.path.join(image_dir, filename)
            cv2.imwrite(image_save_path, image)
            print(f"Saved image: {image_save_path}")

            # Save image with bounding boxes to debug/bboxes
            bboxes_debug_path = os.path.join(bboxes_dir, f"bboxes_{filename}")
            cv2.imwrite(bboxes_debug_path, annotated_image)
            print(f"Saved bounding box image: {bboxes_debug_path}")

            # Save mask visualization (optional)
            mask_visualization_path = os.path.join(maskinyolo_dir, f"maskinyolo_visualization_{filename}")
            print("mask_visualization path: ", mask_visualization_path)
            cv2.imwrite(mask_visualization_path, mask_visualization)
            print(f"Saved mask visualization image: {mask_visualization_path}")

            processedimages.append(filename)
            counter += 1
            print("Added ", filename, " to processedimages array")
            print("Processed images: ", processedimages)
            print("Filename:", filename)

    print(f"\nTotal processed images: {processed_files}")

if __name__ == "__main__":
    config = load_config()
    args = parse_arguments()
    # You need to provide class_name when calling preprocess_images
    # For standalone testing, replace 'your_class_name' with an actual class name
    class_name = 'your_class_name'  # Replace with actual class name
    preprocess_images(config, processedimages=[], counter=0, mode=args.mode, class_name=class_name)