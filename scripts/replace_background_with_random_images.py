# scripts/replace_background_with_random_images.py

import sys
import os
import cv2
import numpy as np
import yaml
import random
import logging

# Adjust sys.path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
    required_keys = [
        'output', 'debug', 'class_names'
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing '{key}' in configuration.")
    # Further validation as needed

def replace_background_with_random_images(config, class_name):
    """
    Replace the background of each image with random background images
    from the backgrounds folder using filled contours from preprocess.py
    for a specific class.
    """
    # Set up logging
    logging.basicConfig(
        filename=os.path.join(project_root, 'scripts', 'replace_background_with_random_images.log'),
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )

    # Get image directories from config
    train_dir = os.path.join(project_root, config['output']['train_image_dir'])
    val_dir = os.path.join(project_root, config['output']['val_image_dir'])
    contours_dir = os.path.join(project_root, config['debug']['contours'])
    backgrounds_dir = os.path.join(project_root, 'data', 'backgrounds')
    coloring_mask_dir = os.path.join(project_root, 'data', 'debug', 'coloringmask_random_bg')

    dirs = [train_dir, val_dir]

    # Create coloring_mask_dir if it doesn't exist
    os.makedirs(coloring_mask_dir, exist_ok=True)

    if not os.path.exists(contours_dir):
        logging.error(f"Contours directory does not exist: {contours_dir}")
        print(f"Contours directory does not exist: {contours_dir}")
        sys.exit(1)

    if not os.path.exists(backgrounds_dir):
        logging.error(f"Backgrounds directory does not exist: {backgrounds_dir}")
        print(f"Backgrounds directory does not exist: {backgrounds_dir}")
        sys.exit(1)

    # Load background images
    background_images = []
    for bg_filename in os.listdir(backgrounds_dir):
        if bg_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            bg_path = os.path.join(backgrounds_dir, bg_filename)
            bg_image = cv2.imread(bg_path)
            if bg_image is not None:
                background_images.append(bg_image)
    if not background_images:
        logging.error("No background images found in backgrounds directory.")
        print("No background images found in backgrounds directory.")
        sys.exit(1)

    for dir_path in dirs:
        if not os.path.exists(dir_path):
            logging.warning(f"Image directory does not exist: {dir_path}. Skipping.")
            print(f"Image directory does not exist: {dir_path}. Skipping.")
            continue

        # Get list of image files starting with class_name
        image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f.startswith(class_name)]

        for filename in image_files:
            image_path = os.path.join(dir_path, filename)

            # Read the original image
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                logging.error(f"Could not read image: {image_path}. Skipping.")
                print(f"Could not read image: {image_path}. Skipping.")
                continue

            # Initialize a variable to accumulate masks for this image
            accumulated_mask = None

            # Construct contour filename
            contour_filename = f"contours_{filename}"
            contour_path = os.path.join(contours_dir, contour_filename)
            print("contour_path: ", contour_path)

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

            # Define green color range to extract the mask
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

            if accumulated_mask is None:
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

            # Select a random background image
            bg_image = random.choice(background_images)
            bg_height, bg_width = bg_image.shape[:2]

            # Resize background to match the image size
            bg_resized = cv2.resize(bg_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)

            # Ensure background is the same size as the original image
            if bg_resized.shape[:2] != image.shape[:2]:
                logging.warning(f"Background image resized to {bg_resized.shape[:2]} does not match image size {image.shape[:2]}.")
                print(f"Background image resized to {bg_resized.shape[:2]} does not match image size {image.shape[:2]}.")
                continue

            # Create a 3-channel mask for background
            mask_inv_3ch = cv2.merge([mask_inv, mask_inv, mask_inv])

            # Extract the background using the inverted mask
            background_part = cv2.bitwise_and(bg_resized, mask_inv_3ch)

            # Extract the object using the mask
            mask_3ch = cv2.merge([accumulated_mask, accumulated_mask, accumulated_mask])
            object_part = cv2.bitwise_and(bgr, mask_3ch)

            # Combine object and new background
            composite = cv2.add(object_part, background_part)

            # If original image had alpha channel, set alpha to object mask
            if has_alpha:
                composite = cv2.cvtColor(composite, cv2.COLOR_BGR2BGRA)
                composite[:, :, 3] = accumulated_mask  # Set alpha channel based on mask

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
                logging.info(f"Replaced background with random image for: {save_path}")
                print(f"Replaced background with random image for: {save_path}")
            else:
                logging.error(f"Failed to save image: {save_path}")
                print(f"Failed to save image: {save_path}")

def main():
    """
    Main function to replace backgrounds using random images.
    """
    config = load_config()
    try:
        validate_config(config)
    except ValueError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)

    # You need to provide class_name when calling the function
    # For standalone testing, replace 'your_class_name' with an actual class name
    class_name = 'your_class_name'  # Replace with actual class name
    replace_background_with_random_images(config, class_name)

if __name__ == "__main__":
    main()