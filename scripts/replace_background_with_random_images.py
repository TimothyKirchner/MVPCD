# scripts/replace_background_with_random_images.py

import sys
import os
import cv2
import numpy as np
import yaml
import random
import logging
from glob import glob
from rectpack import newPacker

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

def load_background_fragments(backgrounds_dir):
    """
    Load background fragments from the specified directory.
    Returns a list of (width, height, image) tuples.
    """
    background_files = glob(os.path.join(backgrounds_dir, '*.png')) + \
                       glob(os.path.join(backgrounds_dir, '*.jpg')) + \
                       glob(os.path.join(backgrounds_dir, '*.jpeg'))
    if not background_files:
        print("No background fragments found in backgrounds directory.")
        sys.exit(1)
    
    background_fragments = []
    for idx, file_path in enumerate(background_files):
        img = cv2.imread(file_path)
        if img is None:
            print(f"Warning: Could not load image {file_path}. Skipping.")
            continue
        height, width = img.shape[:2]
        background_fragments.append((width, height, img))
    
    if not background_fragments:
        print("No valid background fragments loaded.")
        sys.exit(1)
    
    print(f"Loaded {len(background_fragments)} background fragments.")
    return background_fragments

def arrange_fragments_bin_packing(fragments, canvas_size):
    """
    Arrange fragments into the canvas using bin packing.
    Returns a list of placement tuples: (fragment_image, x, y)
    """
    bin_width, bin_height = canvas_size
    packer = newPacker(mode='maxrects')

    # Add the single bin where all fragments will be placed
    packer.add_bin(bin_width, bin_height)

    # Add rectangles to packer
    for idx, (w, h, img) in enumerate(fragments):
        packer.add_rect(w, h, rid=idx)

    # Start packing
    packer.pack()

    # Retrieve packing results
    placements = []
    all_rects = packer.rect_list()
    for rect in all_rects:
        bin_index, x, y, w, h, rid = rect
        fragment_img = fragments[rid][2]
        placements.append((fragment_img, x, y))
        print(f"Placed fragment {rid} at position ({x}, {y}) with size ({w}x{h})")
    
    return placements

def create_mosaic_canvas(canvas_size):
    """
    Create a blank canvas of the specified size.
    """
    bin_width, bin_height = canvas_size
    canvas = np.zeros((bin_height, bin_width, 3), dtype=np.uint8)
    return canvas

def composite_fragments_on_canvas(canvas, placements):
    """
    Place fragments onto the canvas at specified positions.
    """
    for fragment_img, x, y in placements:
        frag_height, frag_width = fragment_img.shape[:2]
        # Ensure the fragment fits within the canvas
        if y + frag_height > canvas.shape[0] or x + frag_width > canvas.shape[1]:
            print(f"Warning: Fragment at ({x}, {y}) with size ({frag_width}x{frag_height}) exceeds canvas bounds. Skipping.")
            continue
        # Overlay fragment onto canvas
        canvas[y:y+frag_height, x:x+frag_width] = fragment_img
    return canvas

def composite_object_foreground(mosaic, object_image, mask):
    """
    Composite the object onto the mosaic background using the mask.
    Ensures that the object is in the foreground.
    """
    # Ensure all images are the same size
    if mosaic.shape[:2] != object_image.shape[:2]:
        print("Resizing object image and mask to match mosaic size.")
        object_image = cv2.resize(object_image, (mosaic.shape[1], mosaic.shape[0]))
        mask = cv2.resize(mask, (mosaic.shape[1], mosaic.shape[0]))
    
    # Create binary mask
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Invert mask for background
    mask_inv = cv2.bitwise_not(binary_mask)
    
    # Extract background from mosaic
    background = cv2.bitwise_and(mosaic, mosaic, mask=mask_inv)
    
    # Extract object from object_image
    foreground = cv2.bitwise_and(object_image, object_image, mask=binary_mask)
    
    # Combine background and foreground
    composite = cv2.add(background, foreground)
    
    return composite

def replace_images_with_mosaic(config, class_name):
    """
    Replace entire images with a mosaic of background fragments for a specific class.
    """
    # Set up logging
    log_file = os.path.join(project_root, 'scripts', 'replace_background_with_random_images.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    
    # Get image directories from config
    train_dir = os.path.join(project_root, config['output']['train_image_dir'])
    val_dir = os.path.join(project_root, config['output']['val_image_dir'])
    output_dirs = [train_dir, val_dir]
    
    # Get mask directory (assuming 'maskinyolo' as per preprocess.py)
    mask_dir = os.path.join(project_root, 'data', 'debug', 'maskinyolo')
    
    # Background fragments directory
    backgrounds_dir = os.path.join(project_root, 'data', 'backgrounds')
    
    # Validate backgrounds directory
    if not os.path.exists(backgrounds_dir):
        logging.error(f"Backgrounds directory does not exist: {backgrounds_dir}")
        print(f"Backgrounds directory does not exist: {backgrounds_dir}")
        sys.exit(1)
    
    # Validate mask directory
    if not os.path.exists(mask_dir):
        logging.error(f"Mask directory does not exist: {mask_dir}")
        print(f"Mask directory does not exist: {mask_dir}")
        sys.exit(1)
    
    # Load background fragments
    background_fragments = load_background_fragments(backgrounds_dir)
    
    for dir_path in output_dirs:
        if not os.path.exists(dir_path):
            logging.warning(f"Image directory does not exist: {dir_path}. Skipping.")
            print(f"Image directory does not exist: {dir_path}. Skipping.")
            continue
    
        # Get list of image files starting with class_name
        image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f.startswith(class_name)]
    
        print(f"\nProcessing {len(image_files)} images in directory: {dir_path}")
    
        for filename in image_files:
            image_path = os.path.join(dir_path, filename)
    
            # Read the original image
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"Could not read image: {image_path}. Skipping.")
                print(f"Could not read image: {image_path}. Skipping.")
                continue
    
            img_height, img_width = image.shape[:2]
            canvas_size = (img_width, img_height)  # rectpack uses (width, height)
    
            # Construct corresponding mask filename
            mask_filename = f"maskinyolo_visualization_{filename}"
            mask_path = os.path.join(mask_dir, mask_filename)
    
            if not os.path.exists(mask_path):
                logging.warning(f"Mask file does not exist: {mask_path}. Skipping image.")
                print(f"Mask file does not exist: {mask_path}. Skipping image.")
                continue
    
            # Read the mask image (assuming it's a binary mask)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logging.error(f"Could not read mask: {mask_path}. Skipping image.")
                print(f"Could not read mask: {mask_path}. Skipping image.")
                continue
    
            # Load the object image (the current image to keep in foreground)
            object_image = image.copy()
    
            # Arrange fragments using bin packing
            placements = arrange_fragments_bin_packing(background_fragments, canvas_size)
    
            # Create mosaic canvas
            mosaic = create_mosaic_canvas(canvas_size)
    
            # Composite fragments onto the canvas
            mosaic = composite_fragments_on_canvas(mosaic, placements)
    
            # Composite object onto the mosaic using the mask
            composite_image = composite_object_foreground(mosaic, object_image, mask)
    
            # Save the composite image, replacing the original
            success = cv2.imwrite(image_path, composite_image)
            if success:
                logging.info(f"Replaced image with mosaic: {image_path}")
                print(f"Replaced image with mosaic: {image_path}")
            else:
                logging.error(f"Failed to save mosaic image: {image_path}")
                print(f"Failed to save mosaic image: {image_path}")

def main():
    """
    Main function to replace entire images with mosaics.
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
    
    replace_images_with_mosaic(config, class_name)
    print(f"Replaced images for class '{class_name}' with mosaics.")

if __name__ == "__main__":
    main()