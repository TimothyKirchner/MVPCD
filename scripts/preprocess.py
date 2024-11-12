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
    with open(os.path.join(project_root, config_path), 'r') as file:
        config = yaml.safe_load(file)
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

def preprocess_images(config, processedimages, counter, mode):
    splits = ['train', 'val']

    for split in splits:
        print(f"\n--- Processing {split} set ---")
        image_dir = os.path.join(project_root, config['output'][f"{split}_image_dir"])
        label_dir = os.path.join(project_root, config['output'][f"{split}_label_dir"])
        depth_dir = os.path.join(project_root, config['output']['depth_dir'])

        debug_dir = os.path.join(project_root, 'data', 'debug')
        rgbmask_dir = os.path.join(debug_dir, 'rgbmask')
        depthmask_dir = os.path.join(debug_dir, 'depthmask')
        combinedmask_dir = os.path.join(debug_dir, 'combined_mask')
        contours_dir = os.path.join(debug_dir, 'contours')
        bboxes_dir = os.path.join(debug_dir, 'bboxes')
        maskinyolo_dir = os.path.join(debug_dir, 'maskinyolo')

        for directory in [image_dir, label_dir, rgbmask_dir, depthmask_dir, combinedmask_dir, contours_dir, bboxes_dir, maskinyolo_dir]:
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

        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
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

                # Determine class name from filename
                class_name = filename.split('_')[0]
                if class_name not in class_names:
                    print(f"Class '{class_name}' not defined in config. Skipping {filename}.")
                    continue

                # Use the first ROI from the list
                roi_config = rois_config[0]

                # Generate ROI mask
                roi_mask = get_roi_mask(image.shape, roi_config)

                # Find all pixels in ROI
                roi_indices = np.argwhere(roi_mask == 1)
                if roi_indices.size == 0:
                    print(f"No ROI found in image: {filename}. Skipping.")
                    continue

                y_indices = roi_indices[:, 0]
                x_indices = roi_indices[:, 1]

                roi_x_min = np.min(x_indices)
                roi_x_max = np.max(x_indices)
                roi_y_min = np.min(y_indices)
                roi_y_max = np.max(y_indices)

                # Desired crop size
                crop_size = 480

                # Calculate possible ranges for the crop's top-left corner
                crop_x_min = max(0, roi_x_max - crop_size + 1)
                crop_x_max = min(roi_x_min, img_width - crop_size)
                crop_y_min = max(0, roi_y_max - crop_size + 1)
                crop_y_max = min(roi_y_min, img_height - crop_size)

                # Handle cases where the ROI is larger than the crop size
                if crop_x_min > crop_x_max:
                    crop_x_min = crop_x_max = max(0, min(roi_x_min, img_width - crop_size))
                if crop_y_min > crop_y_max:
                    crop_y_min = crop_y_max = max(0, min(roi_y_min, img_height - crop_size))

                # Randomly select top-left corner of the crop within the calculated ranges
                if crop_x_min == crop_x_max:
                    crop_x = crop_x_min
                else:
                    crop_x = random.randint(int(crop_x_min), int(crop_x_max))

                if crop_y_min == crop_y_max:
                    crop_y = crop_y_min
                else:
                    crop_y = random.randint(int(crop_y_min), int(crop_y_max))

                # Crop the image
                crop_x_end = crop_x + crop_size
                crop_y_end = crop_y + crop_size

                # Ensure crop does not exceed image boundaries
                crop_x_end = min(crop_x_end, img_width)
                crop_y_end = min(crop_y_end, img_height)
                crop_x = crop_x_end - crop_size
                crop_y = crop_y_end - crop_size

                cropped_image = image[crop_y:crop_y_end, crop_x:crop_x_end].copy()

                # Update image dimensions
                img_height_crop, img_width_crop = cropped_image.shape[:2]

                # Adjust ROI coordinates relative to the cropped image
                roi_x_adjusted = roi_x_min - crop_x
                roi_y_adjusted = roi_y_min - crop_y
                roi_w_adjusted = roi_x_max - roi_x_min + 1
                roi_h_adjusted = roi_y_max - roi_y_min + 1

                # Apply chroma key on the cropped image
                hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
                chroma_keyed = apply_chroma_key(hsv, lower_color, upper_color)

                # Load depth map and crop accordingly
                depth_filename = filename.rsplit('.', 1)[0] + '.npy'
                depth_path = os.path.join(depth_dir, f"depth_{depth_filename}")
                depth_map = load_depth_map(depth_path)

                if depth_map is None:
                    print(f"Depth map not found: {depth_path}")
                    print(f"Depth map not found for {filename}. Skipping.")
                    continue

                # Crop depth map
                depth_map_cropped = depth_map[crop_y:crop_y_end, crop_x:crop_x_end]

                # Get depth thresholds for the specific class
                depth_threshold = config.get('depth_thresholds', {}).get(class_name, {})
                min_depth = depth_threshold.get('min', 500)
                max_depth = depth_threshold.get('max', 2000)

                # Process depth mask
                depth_mask = process_depth(depth_map_cropped, min_depth=min_depth, max_depth=max_depth)
                chroma_mask = cv2.cvtColor(chroma_keyed, cv2.COLOR_BGR2GRAY)
                combined_mask = cv2.bitwise_and(chroma_mask, depth_mask)

                # Threshold the combined mask to ensure it's binary
                _, binary_mask = cv2.threshold(combined_mask, 1, 255, cv2.THRESH_BINARY)

                # Create translated ROI mask in cropped image
                translated_roi = {
                    'x': roi_x_adjusted,
                    'y': roi_y_adjusted,
                    'width': roi_w_adjusted,
                    'height': roi_h_adjusted
                }
                translated_roi_mask = get_roi_mask(cropped_image.shape, translated_roi)

                # Combine translated ROI mask with the existing combined mask
                final_mask = cv2.bitwise_and(binary_mask, translated_roi_mask * 255)

                # Save masks
                rgb_mask_path = os.path.join(rgbmask_dir, f"rgbmask_{filename}")
                depth_mask_path_save = os.path.join(depthmask_dir, f"depthmask_{filename}")
                combined_mask_path = os.path.join(combinedmask_dir, f"combined_mask_{filename}")
                contours_path = os.path.join(contours_dir, f"contours_{filename}.png")

                cv2.imwrite(rgb_mask_path, chroma_mask)
                cv2.imwrite(depth_mask_path_save, depth_mask)
                cv2.imwrite(combined_mask_path, combined_mask)
                print(f"Saved RGB mask: {rgb_mask_path}")
                print(f"Saved Depth mask: {depth_mask_path_save}")
                print(f"Saved Combined mask: {combined_mask_path}")

                # Find contours on the final mask
                contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contours_image = cv2.drawContours(cropped_image.copy(), contours, -1, (0, 255, 0), 2)
                    cv2.imwrite(contours_path, contours_image)
                    print(f"Saved contours: {contours_path}")
                else:
                    print(f"No contours found for {filename}.")
                    if mode == 'segmentation':
                        continue

                # Initialize annotations
                annotations = []

                # Prepare mask visualization
                mask_visualization = cropped_image.copy()

                # Process each contour
                for contour in contours:
                    if cv2.contourArea(contour) < 100:
                        continue

                    x, y, w, h = cv2.boundingRect(contour)

                    x_center = (x + w / 2) / img_width_crop
                    y_center = (y + h / 2) / img_height_crop
                    width_norm = w / img_width_crop
                    height_norm = h / img_height_crop

                    if mode == 'segmentation':
                        contour = contour.reshape(-1, 2)
                        segmentation = contour.astype(np.float32)
                        segmentation[:, 0] /= img_width_crop
                        segmentation[:, 1] /= img_height_crop
                        segmentation = segmentation.flatten().tolist()

                        class_id = class_id_map.get(class_name, 0)
                        annotation = [class_id, x_center, y_center, width_norm, height_norm] + segmentation

                        points = (contour * [img_width_crop, img_height_crop]).astype(np.int32)
                        cv2.polylines(mask_visualization, [points], isClosed=True, color=(0, 0, 255), thickness=2)
                        cv2.fillPoly(mask_visualization, [points], color=(0, 0, 255))
                    else:
                        class_id = class_id_map.get(class_name, 0)
                        annotation = [class_id, x_center, y_center, width_norm, height_norm]

                    annotations.append(annotation)

                # Save annotations
                if annotations:
                    base_filename = os.path.splitext(filename)[0]
                    label_path = os.path.join(label_dir, f"{base_filename}.txt")
                    with open(label_path, 'w') as f:
                        for annotation in annotations:
                            annotation_str = ' '.join([f"{a:.6f}" if isinstance(a, float) else str(a) for a in annotation])
                            f.write(annotation_str + '\n')
                    print(f"Annotations saved for {label_path}")
                else:
                    print(f"No valid annotations for {filename}. Skipping label file generation.")
                    continue

                # Save cropped image
                image_save_path = os.path.join(image_dir, filename)
                cv2.imwrite(image_save_path, cropped_image)
                print(f"Saved cropped image: {image_save_path}")

                # Save image with bounding boxes
                annotated_image = mask_visualization.copy()
                for contour in contours:
                    if cv2.contourArea(contour) < 100:
                        continue
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                bboxes_debug_path = os.path.join(bboxes_dir, f"bboxes_{filename}")
                cv2.imwrite(bboxes_debug_path, annotated_image)
                print(f"Saved bounding box image: {bboxes_debug_path}")

                # Save mask visualization
                mask_visualization_path = os.path.join(maskinyolo_dir, f"maskinyolo_{filename}")
                cv2.imwrite(mask_visualization_path, mask_visualization)
                print(f"Saved mask visualization image: {mask_visualization_path}")

                processedimages.insert(counter, filename)
                counter += 1
                print("Added ", filename, " to processedimages array")
                print("Processed images: ", processedimages)
                print("Filename:", filename)

    if __name__ == "__main__":
        config = load_config()
        args = parse_arguments()
        preprocess_images(config, processedimages=[], counter=0, mode=args.mode)