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

def get_roi_mask(image_shape, rois):
    height, width = image_shape[:2]
    roi_mask = np.zeros((height, width), dtype=np.uint8)

    for roi in rois:
        x_min = int(roi.get('x', 0))
        y_min = int(roi.get('y', 0))
        x_max = x_min + int(roi.get('width', 0))
        y_max = y_min + int(roi.get('height', 0))

        # Ensure ROI coordinates are within image bounds
        x_min = max(0, min(x_min, width - 1))
        y_min = max(0, min(y_min, height - 1))
        x_max = max(0, min(x_max, width))
        y_max = max(0, min(y_max, height))

        roi_mask[y_min:y_max, x_min:x_max] = 1

    return roi_mask

def delete_files_for_image(filename, config):
    """
    Delete all files corresponding to a given image.
    """
    base_filename = os.path.splitext(filename)[0]
    label_filename = base_filename + '.txt'

    # Directories
    image_dirs = [
        os.path.join(project_root, config['output']['train_image_dir']),
        os.path.join(project_root, config['output']['val_image_dir']),
        os.path.join(project_root, config['output']['test_image_dir'])
    ]
    label_dirs = [
        os.path.join(project_root, config['output']['train_label_dir']),
        os.path.join(project_root, config['output']['val_label_dir']),
        os.path.join(project_root, config['output']['test_label_dir'])
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
    splits = ['train', 'val', 'test']

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

            # Extract angle index from filename
            parts = filename.split('_')
            if len(parts) >= 3 and parts[1].startswith('angle'):
                try:
                    angle_index = int(parts[1].replace('angle', ''))
                except ValueError:
                    angle_index = 0  # Default angle index if not found or invalid
            else:
                angle_index = 0  # Default angle index if not found

            # Get ROIs for the specific class and angle
            rois_config = config.get('rois', {}).get(class_name, {}).get(angle_index, [])

            if not rois_config:
                print(f"Error: 'rois' not defined for class '{class_name}', angle {angle_index}.")
                sys.exit(1)

            # Generate ROI mask
            roi_mask = get_roi_mask(image.shape, rois_config)

            # Get chroma key settings for the specific class and angle
            chroma_key_settings = config.get('chroma_key_settings', {}).get(class_name, {}).get(angle_index, {})
            lower_color = np.array(chroma_key_settings.get('lower_color', [0, 0, 0]), dtype=np.uint8)
            upper_color = np.array(chroma_key_settings.get('upper_color', [179, 255, 255]), dtype=np.uint8)

            # Apply chroma key on the image
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            chroma_keyed = apply_chroma_key(hsv, lower_color, upper_color)

            # Load depth map
            depth_filename = f"depth_{filename}".replace('.png', '.npy')
            depth_path = os.path.join(depth_dir, depth_filename)
            depth_map = load_depth_map(depth_path)

            if depth_map is None:
                print(f"Depth map not found: {depth_path}")
                print(f"Depth map not found for {filename}. Skipping.")
                continue

            # Get depth thresholds for the specific class and angle
            depth_threshold = config.get('depth_thresholds', {}).get(class_name, {}).get(angle_index, {})
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
            bbox_indices = []

            # Process each contour
            for idx, contour in enumerate(contours):
                if cv2.contourArea(contour) < 100:
                    continue

                x, y, w, h = cv2.boundingRect(contour)

                area = w * h

                # Only consider areas where w and h are positive
                if w <= 0 or h <= 0:
                    continue

                image_bbox_areas.append(area)
                bbox_indices.append(idx)

                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width_norm = w / img_width
                height_norm = h / img_height

                if mode == 'segmentation':
                    # ... (code for segmentation)
                    pass
                else:
                    class_id = class_id_map.get(class_name, 0)
                    annotation = [class_id, x_center, y_center, width_norm, height_norm]

                annotations.append(annotation)

                # Draw bounding box on mask_visualization and annotated_image
                cv2.rectangle(mask_visualization, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Add index number next to bounding box
                cv2.putText(mask_visualization, str(idx), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

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
                    # Prepare to display the image with annotations
                    window_name = f"Bounding Box Check: {filename}"

                    # Resize window to fit on screen if necessary
                    screen_res = (1280, 720)
                    scale_width = screen_res[0] / mask_visualization.shape[1]
                    scale_height = screen_res[1] / mask_visualization.shape[0]
                    scale = min(scale_width, scale_height)
                    if scale < 1:
                        window_width = int(mask_visualization.shape[1] * scale)
                        window_height = int(mask_visualization.shape[0] * scale)
                        mask_visualization_resized = cv2.resize(mask_visualization, (window_width, window_height))
                    else:
                        mask_visualization_resized = mask_visualization.copy()

                    while True:
                        # Display instructions
                        display_image = mask_visualization_resized.copy()
                        instructions = "Press 's' to save, 'd' to delete, 'c' to choose bbox, 'Esc' to exit."
                        cv2.putText(display_image, instructions, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.imshow(window_name, display_image)
                        key = cv2.waitKey(0) & 0xFF  # Wait for key press

                        if key == ord('s') or key == ord('S'):
                            # Keep image, proceed with current annotations
                            break
                        elif key == ord('d') or key == ord('D'):
                            # Delete the files corresponding to this image
                            delete_files_for_image(filename, config)
                            # Skip processing this image
                            skip_image = True
                            break
                        elif key == ord('c') or key == ord('C'):
                            # Enter input mode to select bounding box
                            input_str = ''
                            input_mode = True
                            while input_mode:
                                # Display input prompt
                                prompt_image = display_image.copy()
                                prompt_text = f"Enter bbox number to keep): {input_str}"
                                cv2.putText(prompt_image, prompt_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.7, (0, 255, 255), 2, cv2.LINE_AA)
                                cv2.imshow(window_name, prompt_image)
                                key_input = cv2.waitKey(0) & 0xFF
                                if ord('0') <= key_input <= ord('9'):
                                    input_str += chr(key_input)
                                elif key_input == 8 or key_input == 255:  # Backspace
                                    input_str = input_str[:-1]
                                elif key_input == 13 or key_input == 10:  # Enter key
                                    if input_str.isdigit():
                                        selected_idx = int(input_str)
                                        if selected_idx in bbox_indices:
                                            # Keep only the selected bounding box
                                            idx_in_annotations = bbox_indices.index(selected_idx)
                                            annotations = [annotations[idx_in_annotations]]
                                            # Update the annotated_image
                                            annotated_image = image.copy()
                                            x, y, w, h = cv2.boundingRect(contours[selected_idx])
                                            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                            input_mode = False
                                            break
                                        else:
                                            print(f"Invalid bbox number. Please enter a number between 0 and {len(bbox_indices)-1}.")
                                            input_str = ''
                                    else:
                                        print("Please enter a valid number.")
                                        input_str = ''
                                elif key_input == 27:  # Escape key
                                    input_mode = False
                                    break
                            if not input_mode:
                                break
                        elif key == 27:  # Escape key
                            print("Exiting.")
                            cv2.destroyAllWindows()
                            sys.exit(0)
                        else:
                            print("Invalid key. Press 's', 'd', 'c', or 'Esc'.")
                    cv2.destroyAllWindows()

                    if skip_image:
                        continue  # Skip saving annotations and images

            # Add areas to bbox_areas
            bbox_areas.extend(image_bbox_areas)

            # Save annotations
            base_filename = os.path.splitext(filename)[0]
            label_path = os.path.join(label_dir, f"{base_filename}.txt")
            try:
                with open(label_path, 'w') as f:
                    for annotation in annotations:
                        annotation_str = ' '.join([f"{a:.6f}" if isinstance(a, float) else str(a) for a in annotation])
                        f.write(annotation_str + '\n')
                print(f"Annotations saved for {label_path}")
            except Exception as e:
                print(f"Failed to save annotations for {label_path}: {e}")
                continue

            # Save image with bounding boxes to debug/bboxes
            bboxes_debug_path = os.path.join(bboxes_dir, f"bboxes_{filename}")
            try:
                cv2.imwrite(bboxes_debug_path, annotated_image)
                print(f"Saved bounding box image: {bboxes_debug_path}")
            except Exception as e:
                print(f"Failed to save bounding box image: {bboxes_debug_path}: {e}")

            # Save mask visualization (optional)
            mask_visualization_path = os.path.join(maskinyolo_dir, f"maskinyolo_visualization_{filename}")
            print("mask_visualization path: ", mask_visualization_path)
            try:
                cv2.imwrite(mask_visualization_path, mask_visualization)
                print(f"Saved mask visualization image: {mask_visualization_path}")
            except Exception as e:
                print(f"Failed to save mask visualization image: {mask_visualization_path}: {e}")

            processedimages.append(filename)
            counter += 1
            print("Added ", filename, " to processedimages array")
            print("Processed images: ", processedimages)
            print("Filename:", filename)

if __name__ == "__main__":
    config = load_config()
    args = parse_arguments()
    # Prompt user for class name
    class_name = input("Enter the class name to process (e.g., 'cat'): ").strip()
    if not class_name:
        print("No class name provided. Exiting.")
        sys.exit(1)
    preprocess_images(config, processedimages=[], counter=0, mode=args.mode, class_name=class_name)