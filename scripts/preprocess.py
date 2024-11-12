# scripts/preprocess.py

import sys
import os
import argparse
import cv2
import numpy as np
import yaml
from utils.chroma_key import apply_chroma_key
from utils.depth_processing import process_depth, load_depth_map

# Adjust sys.path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

def preprocess_images(config, processedimages, counter, mode):
    splits = ['train', 'val']

    for split in splits:
        print(f"\n--- Processing {split} set ---")
        image_dir = os.path.join(project_root, config['output'][f"{split}_image_dir"])
        label_dir = os.path.join(project_root, config['output'][f"{split}_label_dir"])
        depth_dir = os.path.join(project_root, config['output']['depth_dir'])

        # Define debug mask directories per split
        debug_dir = os.path.join(project_root, 'data', 'debug')
        rgbmask_dir = os.path.join(debug_dir, 'rgbmask')
        depthmask_dir = os.path.join(debug_dir, 'depthmask')
        combinedmask_dir = os.path.join(debug_dir, 'combined_mask')
        contours_dir = os.path.join(debug_dir, 'contours')       # Contours directory
        bboxes_dir = os.path.join(debug_dir, 'bboxes')           # Bounding boxes directory
        maskinyolo_dir = os.path.join(debug_dir, 'maskinyolo')   # Mask visualization directory

        # Create directories if they don't exist
        for directory in [image_dir, label_dir, rgbmask_dir, depthmask_dir, combinedmask_dir, contours_dir, bboxes_dir, maskinyolo_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        chroma_key = config.get('chroma_key', {})
        lower_color = np.array(chroma_key.get('lower_color', [0, 0, 0]), dtype=np.uint8)
        upper_color = np.array(chroma_key.get('upper_color', [179, 255, 255]), dtype=np.uint8)

        rois = config.get('rois', [])

        class_names = config.get('class_names', [])
        class_id_map = {name: idx for idx, name in enumerate(class_names)}

        processed_files = 0  # Counter to track processed files

        print(f"Image directory: {image_dir}")

        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                if filename in processedimages:
                    continue
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

                # Threshold the combined mask to ensure it's binary
                _, binary_mask = cv2.threshold(combined_mask, 1, 255, cv2.THRESH_BINARY)

                # Find contours on the binary mask
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contours_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)
                    cv2.imwrite(contours_path, contours_image)
                    print(f"Saved contours: {contours_path}")
                else:
                    print(f"No contours found for {filename}.")
                    if mode == 'segmentation':
                        # Skip images without contours if in segmentation mode
                        continue

                # Initialize a list to store annotations
                annotations = []

                # Prepare mask visualization
                mask_visualization = image.copy()

                # Process each contour
                for contour in contours:
                    # Filter out small contours
                    if cv2.contourArea(contour) < 100:
                        continue

                    # Bounding box
                    x, y, w, h = cv2.boundingRect(contour)

                    # Convert bounding box to normalized coordinates
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    width_norm = w / img_width
                    height_norm = h / img_height

                    # Get the class ID
                    class_id = class_id_map.get(class_name, 0)

                    # For segmentation mode, get the segmentation mask polygon points
                    if mode == 'segmentation':
                        # Use the contour points directly
                        contour = contour.reshape(-1, 2)

                        # Normalize the contour points
                        segmentation = np.zeros_like(contour, dtype=np.float32)
                        segmentation[:, 0] = contour[:, 0] / img_width
                        segmentation[:, 1] = contour[:, 1] / img_height

                        # Ensure all coordinates are between 0 and 1
                        segmentation = np.clip(segmentation, 0.0, 1.0)

                        segmentation = segmentation.flatten().tolist()

                        # Create the annotation line with segmentation
                        annotation = [class_id, x_center, y_center, width_norm, height_norm] + segmentation

                        # Denormalize segmentation points for visualization
                        points = (contour).astype(np.int32)

                        # Draw the segmentation mask on the visualization image
                        cv2.polylines(mask_visualization, [points], isClosed=True, color=(0, 0, 255), thickness=2)

                        # Optionally, fill the mask
                        cv2.fillPoly(mask_visualization, [points], color=(0, 0, 255))

                    else:
                        # For bbox only, no segmentation
                        annotation = [class_id, x_center, y_center, width_norm, height_norm]

                    annotations.append(annotation)

                # Save annotations to label file
                if annotations:
                    base_filename = os.path.splitext(filename)[0]
                    label_path = os.path.join(label_dir, f"{base_filename}.txt")
                    with open(label_path, 'w') as f:
                        for annotation in annotations:
                            # Convert all numbers to strings with up to 6 decimal places
                            annotation_str = ' '.join([f"{a:.6f}" if isinstance(a, float) else str(a) for a in annotation])
                            f.write(annotation_str + '\n')
                    print(f"Annotations saved for {label_path}")
                else:
                    print(f"No valid annotations for {filename}. Skipping label file generation.")
                    continue  # Skip if no annotations

                # Save image with bounding boxes for debugging
                annotated_image = image.copy()
                for contour in contours:
                    # Filter out small contours
                    if cv2.contourArea(contour) < 100:
                        continue
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                bboxes_debug_path = os.path.join(bboxes_dir, f"bboxes_{filename}")
                cv2.imwrite(bboxes_debug_path, annotated_image)
                print(f"Saved bounding box image: {bboxes_debug_path}")

                # Save the mask visualization image
                mask_visualization_path = os.path.join(maskinyolo_dir, f"maskinyolo_{filename}")
                cv2.imwrite(mask_visualization_path, mask_visualization)
                print(f"Saved mask visualization image: {mask_visualization_path}")

                processedimages.insert(counter, filename)
                counter = counter + 1
                print("Added ", filename, " to processedimages array")
                print("Processed images: ", processedimages)
                print("Filename:", filename)

        print(f"\nPreprocessing completed for {split} set. Total files processed: {processed_files}")

if __name__ == "__main__":
    config = load_config()
    args = parse_arguments()
    preprocess_images(config, processedimages=[], counter=0, mode=args.mode)