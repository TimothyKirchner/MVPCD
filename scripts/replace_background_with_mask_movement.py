# scripts/replace_background_with_mask_movement.py

import sys
import os
import cv2
import numpy as np
import yaml
import random
import logging
from glob import glob

# Adjust sys.path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Preset colors to be used for background replacement (BGR format)
PRESET_COLORS = [
    [255, 255, 255],  # White
    [240, 240, 240],  # Light Gray
    [200, 200, 200],  # Medium Gray
    [175, 175, 175],  # Darker Gray
    [128, 128, 128],  # Dark Gray
    [0, 175, 0],      # Green
    [0, 128, 0],      # Dark Green
    [0, 125, 255],    # Light Blue
    [0, 0, 255],      # Blue
    [192, 192, 192],  # Silver
    # Add more colors as needed
]

def load_config(config_path='config/config.yaml'):
    """
    Load the YAML configuration file.
    """
    config_full_path = os.path.join(project_root, config_path)
    if not os.path.exists(config_full_path):
        print(f"Configuration file not found at {config_full_path}.")
        sys.exit(1)
    with open(config_full_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(config, config_path='config/config.yaml'):
    """
    Save the YAML configuration file.
    """
    config_full_path = os.path.join(project_root, config_path)
    with open(config_full_path, 'w') as file:
        yaml.dump(config, file)

def validate_config(config):
    """
    Validate the loaded configuration for required keys.
    """
    required_keys = [
        'output', 'debug', 'class_names', 'camera'
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing '{key}' in configuration.")
    # Further validation as needed

def initialize_logging(log_file):
    """
    Initialize logging to the specified log file.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    # Also set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s:%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def get_label_path(label_dir, filename):
    """
    Get the corresponding label file path for a given image filename.
    """
    base_name = os.path.splitext(filename)[0]
    return os.path.join(label_dir, f"{base_name}.txt")

def update_label(label_path, bbox, image_width, image_height):
    """
    Update the label file with the new bounding box coordinates.

    Parameters:
    - label_path: Path to the label file.
    - bbox: Tuple (x_center, y_center, width, height) in pixels.
    - image_width: Width of the image in pixels.
    - image_height: Height of the image in pixels.
    """
    # Convert bbox to YOLO format (normalized)
    x_center_norm = bbox[0] / image_width
    y_center_norm = bbox[1] / image_height
    width_norm = bbox[2] / image_width
    height_norm = bbox[3] / image_height

    # Read the existing label to get the class_id
    if not os.path.exists(label_path):
        logging.warning(f"Label file does not exist: {label_path}. Creating a new one.")
        print(f"Label file does not exist: {label_path}. Creating a new one.")
        # Assuming class_id is 0 if not present. Adjust as needed.
        class_id = 0
    else:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        if not lines:
            logging.warning(f"No label found in {label_path}. Creating a new one.")
            print(f"No label found in {label_path}. Creating a new one.")
            class_id = 0
        else:
            # Extract class_id from the first line
            class_id = lines[0].split()[0]

    # Write the updated label
    with open(label_path, 'w') as f:
        f.write(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")

def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2):
    """
    Draw a bounding box on the image.

    Parameters:
    - image: The image on which to draw.
    - bbox: Tuple (x_center, y_center, width, height) in pixels.
    - color: Bounding box color.
    - thickness: Thickness of the bounding box lines.

    Returns:
    - Image with bounding box drawn.
    """
    x_center, y_center, w, h = bbox
    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)

    # Ensure coordinates are within image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1] - 1, x2)
    y2 = min(image.shape[0] - 1, y2)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image

def replace_background_with_mask_movement(config, class_name):
    """
    Replace the background of each image with preset colors from PRESET_COLORS
    using filled contours from preprocess.py for a specific class.
    Move the object along with its mask and update the labels accordingly.
    """
    # Set up logging
    log_file = os.path.join(project_root, 'scripts', 'remove_greenscreen.log')
    initialize_logging(log_file)
    logging.info("Started background replacement with mask movement process.")
    print("Started background replacement with mask movement process.")

    # Get image and label directories from config
    train_image_dir = os.path.join(project_root, config['output']['train_image_dir'])
    val_image_dir = os.path.join(project_root, config['output']['val_image_dir'])
    train_label_dir = os.path.join(project_root, config['output']['train_label_dir'])
    val_label_dir = os.path.join(project_root, config['output']['val_label_dir'])
    contours_dir = os.path.join(project_root, config['debug']['contours'])
    coloring_mask_dir = os.path.join(project_root, 'data', 'debug', 'coloringmask')
    placement_dir = os.path.join(project_root, 'data', 'debug', 'placement')

    dirs = [
        (train_image_dir, train_label_dir),
        (val_image_dir, val_label_dir)
    ]

    # Create necessary directories if they don't exist
    os.makedirs(coloring_mask_dir, exist_ok=True)
    os.makedirs(placement_dir, exist_ok=True)

    if not os.path.exists(contours_dir):
        logging.error(f"Contours directory does not exist: {contours_dir}")
        print(f"Contours directory does not exist: {contours_dir}")
        sys.exit(1)

    processed_count = 0

    for image_dir, label_dir in dirs:
        if not os.path.exists(image_dir):
            logging.warning(f"Image directory does not exist: {image_dir}. Skipping.")
            print(f"Image directory does not exist: {image_dir}. Skipping.")
            continue

        # Get list of image files starting with class_name
        image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f.startswith(class_name)
        ]

        print(f"\nProcessing {len(image_files)} images in directory: {image_dir}")

        for filename in image_files:
            image_path = os.path.join(image_dir, filename)
            label_path = get_label_path(label_dir, filename)

            # Read the original image
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                logging.error(f"Could not read image: {image_path}. Skipping.")
                print(f"Could not read image: {image_path}. Skipping.")
                continue

            # Construct contour filename
            contour_filename = f"contours_{filename}"
            contour_path = os.path.join(contours_dir, contour_filename)
            print(f"Processing contour path: {contour_path}")

            if not os.path.exists(contour_path):
                logging.warning(f"Contour image '{contour_path}' does not exist for class '{class_name}'. Skipping.")
                print(f"Contour image '{contour_path}' does not exist for class '{class_name}'. Skipping.")
                continue

            # Read the contour image
            contour_image = cv2.imread(contour_path, cv2.IMREAD_COLOR)
            if contour_image is None:
                logging.error(f"Could not read contour image: {contour_path}. Skipping.")
                print(f"Could not read contour image: {contour_path}. Skipping.")
                continue

            # Ensure contour image matches the size of the original image
            if contour_image.shape[:2] != image.shape[:2]:
                logging.warning(f"Contour image size {contour_image.shape[:2]} does not match image size {image.shape[:2]}. Resizing contour image.")
                print(f"Contour image size {contour_image.shape[:2]} does not match image size {image.shape[:2]}. Resizing contour image.")
                contour_image = cv2.resize(contour_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Define green color range (BGR format)
            lower_green = np.array([0, 200, 0])
            upper_green = np.array([100, 255, 100])
            mask_contour = cv2.inRange(contour_image, lower_green, upper_green)

            # Find contours from the mask
            contours, _ = cv2.findContours(mask_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                logging.warning(f"No contours found in contour image '{contour_path}' for class '{class_name}'. Skipping.")
                print(f"No contours found in contour image '{contour_path}' for class '{class_name}'. Skipping.")
                continue

            # Create a mask for this class
            accumulated_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

            # Draw filled contours on the accumulated mask
            cv2.drawContours(accumulated_mask, contours, -1, (255), thickness=cv2.FILLED)

            if accumulated_mask is None or np.count_nonzero(accumulated_mask) == 0:
                logging.warning(f"No valid contours found for image '{filename}'. Skipping.")
                print(f"No valid contours found for image '{filename}'. Skipping.")
                continue

            # Save the accumulated mask for debugging
            coloring_mask_path = os.path.join(coloring_mask_dir, f"coloring_mask_{filename}")
            cv2.imwrite(coloring_mask_path, accumulated_mask)

            # Invert the mask to get the background
            mask_inv = cv2.bitwise_not(accumulated_mask)

            # Check if image has alpha channel
            if len(image.shape) == 3 and image.shape[2] == 4:
                # Separate the BGR and Alpha channels
                bgr = image[:, :, :3]
                alpha = image[:, :, 3]
                has_alpha = True
            elif len(image.shape) == 3 and image.shape[2] == 3:
                bgr = image
                has_alpha = False
            else:
                logging.warning(f"Unsupported image format for '{filename}'. Skipping.")
                print(f"Unsupported image format for '{filename}'. Skipping.")
                continue

            # Create an empty background image
            background = np.zeros_like(bgr, dtype=np.uint8)

            # Get indices where background is to be replaced
            background_indices = np.where(mask_inv == 255)

            # Number of background pixels
            num_background_pixels = len(background_indices[0])

            if num_background_pixels == 0:
                logging.warning(f"No background pixels found in image '{filename}'. Skipping.")
                print(f"No background pixels found in image '{filename}'. Skipping.")
                continue

            # Randomly assign colors from PRESET_COLORS to background pixels
            random_colors = np.array(PRESET_COLORS, dtype=np.uint8)
            random_indices = np.random.randint(0, len(PRESET_COLORS), size=num_background_pixels)
            selected_colors = random_colors[random_indices]

            # Assign the selected colors to the background
            background[background_indices[0], background_indices[1]] = selected_colors

            # Now, composite the object and the new background
            # Extract the object using the accumulated mask
            mask_3ch = cv2.merge([accumulated_mask, accumulated_mask, accumulated_mask])
            object_part = cv2.bitwise_and(bgr, mask_3ch)

            # Extract the background using the inverse mask
            mask_inv_3ch = cv2.merge([mask_inv, mask_inv, mask_inv])
            background_part = cv2.bitwise_and(background, mask_inv_3ch)

            # Combine object and background
            composite = cv2.add(object_part, background_part)

            # If original image had alpha channel, set alpha to object mask
            if has_alpha:
                composite = cv2.cvtColor(composite, cv2.COLOR_BGR2BGRA)
                composite[:, :, 3] = accumulated_mask  # Set alpha channel based on mask

            # Determine object's bounding box in the mask
            x, y, w, h = cv2.boundingRect(accumulated_mask)
            x_center = x + w / 2
            y_center = y + h / 2
            bbox = (x_center, y_center, w, h)

            # Reposition the object and its mask to a random location
            # Calculate the maximum top-left coordinates to ensure the object fits
            max_x = image.shape[1] - w
            max_y = image.shape[0] - h
            if max_x <= 0 or max_y <= 0:
                logging.warning(f"Object size ({w}x{h}) is larger than image size for '{filename}'. Skipping repositioning.")
                print(f"Object size ({w}x{h}) is larger than image size for '{filename}'. Skipping repositioning.")
                continue

            new_x = random.randint(0, max_x)
            new_y = random.randint(0, max_y)
            new_bbox = (new_x + w / 2, new_y + h / 2, w, h)

            # Extract the object image and mask
            object_img = object_part[y:y+h, x:x+w]
            object_mask = accumulated_mask[y:y+h, x:x+w]

            # Remove the object from the current position in composite and mask
            composite[y:y+h, x:x+w] = background[y:y+h, x:x+w]
            accumulated_mask[y:y+h, x:x+w] = 0

            # Place the object image and mask at the new position
            composite[new_y:new_y+h, new_x:new_x+w] = object_img
            accumulated_mask[new_y:new_y+h, new_x:new_x+w] = object_mask

            # Update the alpha channel if needed
            if has_alpha:
                composite[:, :, 3] = accumulated_mask

            # Save the composited image, replacing the original
            # Choose appropriate format based on original format
            ext = os.path.splitext(filename)[1].lower()
            save_path = image_path  # Default save path

            if ext in ['.jpg', '.jpeg']:
                # JPEG does not support alpha channel, convert to BGR
                if composite.shape[2] == 4:
                    composite_bgr = cv2.cvtColor(composite, cv2.COLOR_BGRA2BGR)
                    save_path = os.path.splitext(image_path)[0] + '.jpg'  # Ensure extension is .jpg
                    success = cv2.imwrite(save_path, composite_bgr)
                else:
                    success = cv2.imwrite(save_path, composite)
            elif ext == '.png':
                # PNG supports alpha channel, keep as is
                success = cv2.imwrite(save_path, composite)
            else:
                # For other formats, save as PNG to preserve alpha
                save_path = os.path.splitext(image_path)[0] + '.png'
                success = cv2.imwrite(save_path, composite)

            if success:
                # Update the label with the new bounding box
                update_label(label_path, new_bbox, image.shape[1], image.shape[0])

                # Draw the bounding box on the composite image for visualization
                image_with_bbox = composite.copy()
                image_with_bbox = draw_bbox(image_with_bbox, new_bbox)

                # Save the visualized image
                visualized_path = os.path.join(placement_dir, f"bbox_{filename}")
                if os.path.splitext(save_path)[1].lower() == '.png':
                    cv2.imwrite(visualized_path, image_with_bbox)
                else:
                    # Convert to BGR if image has alpha channel
                    if image_with_bbox.shape[2] == 4:
                        image_with_bbox = cv2.cvtColor(image_with_bbox, cv2.COLOR_BGRA2BGR)
                    cv2.imwrite(visualized_path, image_with_bbox)

                logging.info(f"Replaced background and moved object with mask for image: {save_path}")
                print(f"Replaced background and moved object with mask for image: {save_path}")
                processed_count += 1
            else:
                logging.error(f"Failed to save image: {save_path}")
                print(f"Failed to save image: {save_path}")

    return processed_count

def main():
    """
    Main function to replace backgrounds and move objects with masks.
    """
    config = load_config()
    try:
        validate_config(config)
    except ValueError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)

    # Prompt user for class name
    class_name = input("Enter the class name to process (e.g., 'cat'): ").strip()
    if not class_name:
        print("No class name provided. Exiting.")
        sys.exit(1)

    processed_count = replace_background_with_mask_movement(config, class_name)
    print(f"Replaced backgrounds and moved objects with masks for class '{class_name}'. Total processed: {processed_count}")

if __name__ == "__main__":
    main()
