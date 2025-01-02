# scripts/replace_background_with_random_insert.py

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

def load_fragments(backgrounds_dir):
    """
    Load all background fragment images from the specified directory.
    """
    fragment_files = glob(os.path.join(backgrounds_dir, '*.png')) + \
                     glob(os.path.join(backgrounds_dir, '*.jpg')) + \
                     glob(os.path.join(backgrounds_dir, '*.jpeg'))
    fragments = [cv2.imread(f) for f in fragment_files if cv2.imread(f) is not None]
    if not fragments:
        print("No valid background fragments found.")
        sys.exit(1)
    return fragments

def initialize_logging(log_file):
    """
    Initialize logging to the specified log file.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )

def get_label_path(label_dir, filename):
    """
    Get the corresponding label file path for a given image filename.
    """
    base_name = os.path.splitext(filename)[0]
    return os.path.join(label_dir, f"{base_name}.txt")

def update_label(label_path, updated_labels):
    """
    Update the label file with the new bounding box coordinates.

    Parameters:
    - label_path: Path to the label file.
    - updated_labels: List of label lines to write to the file.
    """
    with open(label_path, 'w') as f:
        for line in updated_labels:
            f.write(line + '\n')

def place_fragments_full_image(mosaic, image_size, fragments, debug_dir=None):
    """
    Place background fragments across the entire image by iterating from top-left to bottom-right.

    Parameters:
    - mosaic: np.ndarray, the black canvas where fragments are placed.
    - image_size: Tuple[int, int], size of the image (width, height).
    - fragments: List[np.ndarray], list of background fragments.
    - debug_dir: str, directory to save debug images.
    """
    canvas_width, canvas_height = image_size
    current_y = 0

    logging.info(f"Starting fragment placement on full image of size ({canvas_width}x{canvas_height})")
    print(f"Starting fragment placement on full image of size ({canvas_width}x{canvas_height})")

    while current_y < canvas_height:
        current_x = 0
        row_height = 0  # Height of the tallest fragment in the current row

        while current_x < canvas_width:
            remaining_width = canvas_width - current_x
            remaining_height = canvas_height - current_y

            # Shuffle fragments to ensure randomness
            random.shuffle(fragments)
            placed = False

            for fragment in fragments:
                frag_h, frag_w = fragment.shape[:2]

                # Check if fragment fits in remaining width and height
                if frag_w <= remaining_width and frag_h <= remaining_height:
                    # Place the fragment at (current_x, current_y)
                    mosaic[current_y:current_y+frag_h, current_x:current_x+frag_w] = fragment
                    logging.info(f"Placed fragment at ({current_x}, {current_y}) with size ({frag_w}x{frag_h})")
                    print(f"Placed fragment at ({current_x}, {current_y}) with size ({frag_w}x{frag_h})")

                    # Update position
                    current_x += frag_w
                    row_height = max(row_height, frag_h)

                    # Save debug image if needed
                    if debug_dir:
                        debug_path = os.path.join(debug_dir, f"placed_{current_x}_{current_y}.png")
                        cv2.imwrite(debug_path, mosaic)

                    placed = True
                    break  # Move to the next position after placing a fragment

            if not placed:
                # No fragment fits; leave the remaining space
                logging.info(f"No fragment fits at ({current_x}, {current_y}). Leaving space.")
                print(f"No fragment fits at ({current_x}, {current_y}). Leaving space.")
                break  # Exit the loop for this row

        if row_height == 0:
            # No fragments were placed in this row; prevent infinite loop by moving down a step
            logging.warning(f"No fragments placed in row starting at y={current_y}. Moving down.")
            print(f"No fragments placed in row starting at y={current_y}. Moving down.")
            current_y += 10  # Move down by a small step (adjust as needed)
        else:
            # Move down by the height of the tallest fragment in the row
            current_y += row_height

def fill_unused_spaces(mosaic, fragments, debug_dir=None):
    """
    Fill any remaining unused spaces in the mosaic by cutting and placing fragments.

    Parameters:
    - mosaic: np.ndarray, the current mosaic image.
    - fragments: List[np.ndarray], list of background fragments.
    - debug_dir: str, directory to save debug images.
    """
    canvas_height, canvas_width = mosaic.shape[:2]

    # Convert to grayscale and threshold to find black regions
    gray_mosaic = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_mosaic, 1, 255, cv2.THRESH_BINARY)
    # Invert to get black regions
    thresh_inv = cv2.bitwise_not(thresh)

    # Find contours of the black regions
    contours, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Skip very small regions
        if w < 50 or h < 50:
            continue

        # Attempt to place a fragment by cutting it to fit the space
        placed = False
        random.shuffle(fragments)  # Shuffle to ensure randomness

        for fragment in fragments:
            frag_h, frag_w = fragment.shape[:2]

            if frag_w >= w and frag_h >= h:
                # Cut the fragment to fit the space
                fragment_cut = fragment[0:h, 0:w]
                mosaic[y:y+h, x:x+w] = fragment_cut
                logging.info(f"Filled space at ({x}, {y}, {w}x{h}) with cut fragment.")
                print(f"Filled space at ({x}, {y}, {w}x{h}) with cut fragment.")

                # Save debug image if needed
                if debug_dir:
                    debug_path = os.path.join(debug_dir, f"filled_{x}_{y}.png")
                    cv2.imwrite(debug_path, mosaic)

                placed = True
                break  # Move to the next space after placing a fragment

        if not placed:
            logging.warning(f"Could not find a fragment to fill space at ({x}, {y}, {w}x{h}).")
            print(f"Could not find a fragment to fill space at ({x}, {y}, {w}x{h}).")

def composite_object_with_contour(original_image, mosaic, contour_image, obj_bbox, move_object=True, debug_mask_dir=None):
    """
    Extract object using contour data, and overlay it onto the mosaic background.
    Optionally move the object to a new random location.

    Parameters:
    - original_image: np.ndarray, the original image.
    - mosaic: np.ndarray, the mosaic background image.
    - contour_image: np.ndarray, the contour image from preprocess.py.
    - obj_bbox: Tuple (x_center_norm, y_center_norm, width_norm, height_norm) of the object to process.
    - move_object: bool, whether to move the object to a new location.
    - debug_mask_dir: str, directory to save the mask to be inserted.

    Returns:
    - composite: np.ndarray, the composite image with the object processed.
    - new_bbox: Tuple (x_center, y_center, width, height), new bounding box of the object.
    """
    # Ensure images are the same size
    mosaic_resized = cv2.resize(mosaic, (original_image.shape[1], original_image.shape[0]))
    contour_resized = cv2.resize(contour_image, (original_image.shape[1], original_image.shape[0]))

    # Create a binary mask from the contour image
    lower_green = np.array([0, 200, 0])
    upper_green = np.array([100, 255, 100])
    mask = cv2.inRange(contour_resized, lower_green, upper_green)

    # Create a new mask with filled contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logging.warning("No contours found in contour image. Cannot process object.")
        return mosaic_resized, None

    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, contours, -1, color=255, thickness=cv2.FILLED)

    # Convert normalized bbox to pixel coordinates
    img_height, img_width = original_image.shape[:2]
    x_center_norm, y_center_norm, width_norm, height_norm = obj_bbox
    x_center = x_center_norm * img_width
    y_center = y_center_norm * img_height
    width = width_norm * img_width
    height = height_norm * img_height

    # Calculate top-left corner
    x = int(x_center - width / 2)
    y = int(y_center - height / 2)
    w = int(width)
    h = int(height)

    # Ensure the bbox is within image bounds
    x = max(0, min(x, img_width - w))
    y = max(0, min(y, img_height - h))
    w = min(w, img_width - x)
    h = min(h, img_height - y)

    # Extract object image and mask
    object_image = original_image[y:y+h, x:x+w]
    object_mask = filled_mask[y:y+h, x:x+w]

    # Check if object_mask is empty
    if cv2.countNonZero(object_mask) == 0:
        logging.warning("Object mask is empty. Skipping this object.")
        return mosaic_resized, None

    # Save the mask to be inserted for debugging
    if debug_mask_dir:
        os.makedirs(debug_mask_dir, exist_ok=True)
        mask_to_insert_path = os.path.join(debug_mask_dir, f"mask_to_insert_{random.randint(1000,9999)}.png")
        cv2.imwrite(mask_to_insert_path, object_mask)

    # Create an alpha channel for the object image using the mask
    if object_image.shape[2] == 3:
        object_image = cv2.cvtColor(object_image, cv2.COLOR_BGR2BGRA)
    object_image[:, :, 3] = object_mask

    # Optionally move the object
    if move_object:
        # Ensure object fits within the image dimensions after moving
        max_x = mosaic_resized.shape[1] - w
        max_y = mosaic_resized.shape[0] - h
        if max_x <= 0 or max_y <= 0:
            logging.warning("Object is too large to move within the image.")
            return mosaic_resized, None

        # Random new position
        new_x = random.randint(0, max_x)
        new_y = random.randint(0, max_y)

        # Update the bounding box
        new_x_center = new_x + w / 2
        new_y_center = new_y + h / 2
        new_bbox = (new_x_center, new_y_center, w, h)
    else:
        new_x, new_y = x, y
        new_x_center = x_center
        new_y_center = y_center
        new_bbox = (x_center, y_center, w, h)

    # Place the object onto the mosaic background
    composite = mosaic_resized.copy()

    # Check if the object fits in the new position
    if new_x + w > composite.shape[1] or new_y + h > composite.shape[0]:
        logging.warning("Object does not fit in the new position.")
        return mosaic_resized, None

    alpha_s = object_image[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    # Get the region of interest on the composite image
    for c in range(0, 3):
        composite[new_y:new_y+h, new_x:new_x+w, c] = (alpha_s * object_image[:, :, c] +
                                                      alpha_l * composite[new_y:new_y+h, new_x:new_x+w, c])

    # Normalize new_bbox
    new_x_center_norm = new_x_center / img_width
    new_y_center_norm = new_y_center / img_height
    new_width_norm = w / img_width
    new_height_norm = h / img_height
    normalized_new_bbox = (new_x_center_norm, new_y_center_norm, new_width_norm, new_height_norm)

    return composite, normalized_new_bbox

def replace_images_with_mosaic(config, class_name):
    """
    Replace entire images with a mosaic of background fragments for a specific class.
    Moves the object and updates the label accordingly.
    """
    # Set up logging
    log_file = os.path.join(project_root, 'scripts', 'replace_background_with_random_insert.log')
    initialize_logging(log_file)

    # Get image directories from config
    train_dir = os.path.join(project_root, config['output']['train_image_dir'])
    val_dir = os.path.join(project_root, config['output']['val_image_dir'])
    label_train_dir = os.path.join(project_root, config['output']['train_label_dir'])
    label_val_dir = os.path.join(project_root, config['output']['val_label_dir'])
    output_dirs = [(train_dir, label_train_dir), (val_dir, label_val_dir)]

    # Get contour directory from preprocess.py output (assuming 'contours' directory)
    contours_dir = os.path.join(project_root, config['debug']['contours'])

    backgrounds_dir = os.path.join(project_root, 'data', 'backgrounds')

    # Validate directories
    if not os.path.exists(backgrounds_dir):
        logging.error(f"Backgrounds directory does not exist: {backgrounds_dir}")
        print(f"Backgrounds directory does not exist: {backgrounds_dir}")
        sys.exit(1)

    if not os.path.exists(contours_dir):
        logging.error(f"Contours directory does not exist: {contours_dir}")
        print(f"Contours directory does not exist: {contours_dir}")
        sys.exit(1)

    # Load background fragments
    fragments = load_fragments(backgrounds_dir)
    logging.info(f"Loaded {len(fragments)} background fragments.")
    print(f"Loaded {len(fragments)} background fragments.")

    # Determine canvas size based on resolution
    camera_config = config.get('camera', {})
    resolution = camera_config.get('resolution', [1280, 720])
    resolution_mapping = {
        (1280, 720): 'HD720',
        (1920, 1080): 'HD1080',
    }
    if tuple(resolution) not in resolution_mapping:
        print("Unsupported resolution provided in config. Defaulting to HD720 (1280x720).")
        logging.warning("Unsupported resolution provided in config. Defaulting to HD720 (1280x720).")
        canvas_size = (1280, 720)
    else:
        canvas_size = tuple(resolution)
        print(f"Using canvas size: {canvas_size} ({resolution_mapping[tuple(resolution)]})")
        logging.info(f"Using canvas size: {canvas_size} ({resolution_mapping[tuple(resolution)]})")

    # Define debug directories
    debug_placement_dir = os.path.join(project_root, 'data', 'debug', 'placement')
    debug_mask_dir = os.path.join(project_root, 'data', 'debug', 'masktoinsert')
    os.makedirs(debug_placement_dir, exist_ok=True)
    os.makedirs(debug_mask_dir, exist_ok=True)

    for dir_path, label_dir in output_dirs:
        if not os.path.exists(dir_path):
            logging.warning(f"Image directory does not exist: {dir_path}. Skipping.")
            print(f"Image directory does not exist: {dir_path}. Skipping.")
            continue

        # Get list of all image files
        image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f.startswith(class_name)]

        print(f"\nProcessing {len(image_files)} images in directory: {dir_path}")

        for filename in image_files:
            image_path = os.path.join(dir_path, filename)
            label_path = get_label_path(label_dir, filename)

            # Read the original image
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"Could not read image: {image_path}. Skipping.")
                print(f"Could not read image: {image_path}. Skipping.")
                continue

            img_height, img_width = image.shape[:2]
            image_size = (img_width, img_height)  # (width, height)

            # Construct corresponding contour filename
            contour_filename = f"contours_{filename}"
            contour_path = os.path.join(contours_dir, contour_filename)

            if not os.path.exists(contour_path):
                logging.warning(f"Contour file does not exist: {contour_path}. Skipping image.")
                print(f"Contour file does not exist: {contour_path}. Skipping image.")
                continue

            # Read the contour image
            contour_image = cv2.imread(contour_path)
            if contour_image is None:
                logging.error(f"Could not read contour image: {contour_path}. Skipping image.")
                print(f"Could not read contour image: {contour_path}. Skipping image.")
                continue

            # Check if label file exists
            if not os.path.exists(label_path):
                logging.warning(f"Label file does not exist: {label_path}. Skipping image.")
                print(f"Label file does not exist: {label_path}. Skipping image.")
                continue

            # Read label file
            with open(label_path, 'r') as f:
                label_lines = f.readlines()

            # Lists to hold updated labels and objects to process
            updated_label_lines = []
            objects_to_process = []

            for line in label_lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    logging.warning(f"Invalid label format in {label_path}: {line}. Skipping this label.")
                    continue
                class_id = int(parts[0])
                x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts[1:5])

                if class_id == config['class_names'].index(class_name):
                    # This object is of the specified class
                    objects_to_process.append((class_id, (x_center_norm, y_center_norm, width_norm, height_norm)))
                else:
                    # Keep other labels unchanged
                    updated_label_lines.append(line)

            if not objects_to_process:
                # No objects of the specified class in this image
                continue

            # Initialize a black canvas
            mosaic = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)  # Note: height, width

            # Place fragments across the entire image
            place_fragments_full_image(mosaic, image_size, fragments, debug_dir=debug_placement_dir)

            # Fill any remaining spaces by cutting and placing fragments
            fill_unused_spaces(mosaic, fragments, debug_dir=debug_placement_dir)

            composite = mosaic.copy()
            for obj in objects_to_process:
                class_id, bbox = obj
                composite, new_bbox = composite_object_with_contour(
                    image, composite, contour_image, bbox, move_object=True, debug_mask_dir=debug_mask_dir)

                if new_bbox is not None:
                    # Prepare the updated label line
                    updated_line = f"{class_id} {new_bbox[0]:.6f} {new_bbox[1]:.6f} {new_bbox[2]:.6f} {new_bbox[3]:.6f}"
                    updated_label_lines.append(updated_line)
                else:
                    # Keep the original label if processing failed
                    original_line = f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
                    updated_label_lines.append(original_line)

            # Save the composite image, replacing the original
            success = cv2.imwrite(image_path, composite)
            if success:
                logging.info(f"Replaced image with mosaic: {image_path}")
                print(f"Replaced image with mosaic: {image_path}")
                # Update the label file
                update_label(label_path, updated_label_lines)
                logging.info(f"Updated label for image: {image_path}")
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

    if class_name not in config['class_names']:
        print(f"Class name '{class_name}' not found in configuration class_names.")
        sys.exit(1)

    replace_images_with_mosaic(config, class_name)
    print(f"Replaced images for class '{class_name}' with mosaics.")

if __name__ == "__main__":
    main()