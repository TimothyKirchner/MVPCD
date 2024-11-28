# scripts/check_multiple_classes.py

import os
import cv2
import yaml
import logging
import sys
import numpy as np

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
    with open(config_full_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging():
    """
    Set up logging for the script.
    """
    log_file = os.path.join(project_root, 'scripts', 'check_multiple_classes.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    logging.info("=== Starting Check Multiple Classes Process ===")
    print(f"Logging initialized. Logs will be saved to '{log_file}'.")

def get_class_names(config):
    """
    Retrieve the list of class names from the config.
    """
    return config.get('class_names', [])

def draw_bounding_boxes(image, boxes, class_ids, class_names):
    """
    Draw bounding boxes and class labels on the image.
    """
    for idx, (box, class_id) in enumerate(zip(boxes, class_ids)):
        x_center, y_center, width, height = box
        img_height, img_width = image.shape[:2]
        
        # Convert from YOLO format to pixel coordinates
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        # Choose a color based on class_id
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Put class name and index
        label = f"{class_names[class_id]}:{idx}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, color, 2)
    return image

def find_images_with_multiple_classes(image_dir, label_dir, class_names):
    """
    Identify images that have multiple classes in their label files.
    
    Returns a list of tuples: (image_path, label_path, class_ids, boxes)
    """
    problematic_images = []
    
    for filename in os.listdir(label_dir):
        if not filename.endswith('.txt'):
            continue
        label_path = os.path.join(label_dir, filename)
        image_filename = os.path.splitext(filename)[0] + '.jpg'  # Assuming JPG, adjust if needed
        image_path = os.path.join(image_dir, image_filename)
        
        if not os.path.exists(image_path):
            logging.warning(f"Image file '{image_path}' does not exist for label '{label_path}'. Skipping.")
            continue
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        class_ids = []
        boxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                logging.warning(f"Invalid label format in '{label_path}': {line.strip()}")
                continue
            class_id = int(parts[0])
            if class_id >= len(class_names):
                logging.warning(f"Class ID {class_id} in '{label_path}' is out of range.")
                continue
            class_ids.append(class_id)
            box = list(map(float, parts[1:5]))  # x_center, y_center, width, height
            boxes.append(box)
        
        unique_classes = set(class_ids)
        if len(unique_classes) > 1:
            problematic_images.append((image_path, label_path, class_ids, boxes))
    
    return problematic_images

def handle_image(image_path, label_path, class_ids, boxes, class_names):
    """
    Display the image with bounding boxes and handle user input to save or delete.
    """
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image '{image_path}'.")
        print(f"Failed to load image '{image_path}'. Skipping.")
        return
    
    # Draw bounding boxes
    annotated_image = draw_bounding_boxes(image.copy(), boxes, class_ids, class_names)
    
    # Display the image
    window_name = f"Check Annotation: {os.path.basename(image_path)}"
    cv2.imshow(window_name, annotated_image)
    print(f"\nProcessing image: {image_path}")
    print("Options: [s] Save | [d] Delete | [n] Next")
    logging.info(f"Displaying image '{image_path}' for user decision.")
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s') or key == ord('S'):
            # Save: Do nothing, keep the files
            logging.info(f"User chose to SAVE '{image_path}'.")
            print("Saved.")
            break
        elif key == ord('d') or key == ord('D'):
            # Delete: Remove image and label files
            try:
                os.remove(image_path)
                os.remove(label_path)
                logging.info(f"User chose to DELETE '{image_path}' and '{label_path}'.")
                print("Deleted.")
            except Exception as e:
                logging.error(f"Failed to delete '{image_path}' or '{label_path}': {e}")
                print(f"Failed to delete files: {e}")
            break
        elif key == ord('n') or key == ord('N'):
            # Next: Skip without any action
            logging.info(f"User chose to SKIP '{image_path}'.")
            print("Skipped.")
            break
        else:
            print("Invalid key pressed. Please press 's' to Save, 'd' to Delete, or 'n' for Next.")
    
    cv2.destroyAllWindows()

def main():
    setup_logging()
    config = {
        'camera': {
            'fps': 30,
            'resolution': [1280, 720]
        },
        'capture': {
            'interval': 0.1,
            'num_images': 100
        },
        'chroma_key_settings': {},  # Initialize as empty dict for per-class per-angle settings
        'debug': {
            'bboxes': 'data/debug/bboxes',
            'combined_mask': 'data/debug/combined_mask',
            'contours': 'data/debug/contours',
            'depthmask': 'data/debug/depthmask',
            'rgbmask': 'data/debug/rgbmask',
            "maskinyolo": "data/debug/maskinyolo",
            "coloringmask_random_bg": "data/debug/coloringmask_random_bg",
            "coloringmask": "data/debug/coloringmask",
            "placement": "data/debug/placement",
            "backgrounds": "data/backgrounds"
        },
        'depth_thresholds': {},
        'image_counters': {},
        'output': {
            'depth_dir': 'data/depth_maps',
            'image_dir': 'data/images',
            'label_dir': 'data/labels',
            'train_image_dir': 'data/images/train',
            'train_label_dir': 'data/labels/train',
            'val_image_dir': 'data/images/val',
            'val_label_dir': 'data/labels/val',
            'test_image_dir': 'data/images/test',        # Added Test Image Directory
            'test_label_dir': 'data/labels/test',        # Added Test Label Directory
            "background_image_dir": "data/backgrounds"
        },
        'rois': {},
        'class_names': []
    }
    class_names = get_class_names(config)
    
    image_dir = os.path.join(project_root, config.get('output', {}).get('image_dir', 'data/images'))
    label_dir = os.path.join(project_root, config.get('output', {}).get('label_dir', 'data/labels'))
    
    if not os.path.exists(image_dir):
        logging.error(f"Image directory '{image_dir}' does not exist.")
        print(f"Image directory '{image_dir}' does not exist. Exiting.")
        sys.exit(1)
    
    if not os.path.exists(label_dir):
        logging.error(f"Label directory '{label_dir}' does not exist.")
        print(f"Label directory '{label_dir}' does not exist. Exiting.")
        sys.exit(1)
    
    problematic_images = find_images_with_multiple_classes(image_dir, label_dir, class_names)
    
    if not problematic_images:
        print("No images with multiple classes found.")
        logging.info("No images with multiple classes found.")
        return
    
    print(f"\nFound {len(problematic_images)} images with multiple classes.")
    logging.info(f"Found {len(problematic_images)} images with multiple classes.")
    
    for image_path, label_path, class_ids, boxes in problematic_images:
        handle_image(image_path, label_path, class_ids, boxes, class_names)
    
    print("\nProcessing completed.")
    logging.info("Processing completed.")

if __name__ == "__main__":
    main()