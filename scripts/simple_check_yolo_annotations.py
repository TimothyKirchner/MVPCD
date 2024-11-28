import os
import cv2
import numpy as np
import yaml

# ============================
# === CONFIGURE THESE PATHS ===
# ============================

# Absolute path to the images folder
IMAGES_DIR = "/home/comflow-pc/Desktop/MVPCD/data/images/train"

# Absolute path to the labels folder
LABELS_DIR = "/home/comflow-pc/Desktop/MVPCD/data/labels/train"

# Optional: Path to class names YAML file
CLASS_NAMES_PATH = "/home/comflow-pc/Desktop/MVPCD/config/class_names.yaml"  # Set to None if not using

# ======================================
# === LOAD CLASS NAMES IF PROVIDED ===
# ======================================

class_names = []
if CLASS_NAMES_PATH and os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, 'r') as f:
        data = yaml.safe_load(f)
        class_names = data.get('class_names', [])
    print(f"Loaded class names: {class_names}")
else:
    print("No class names file provided or file does not exist. Class IDs will be used instead.")

# ================================
# === HELPER FUNCTIONS ===========
# ================================

def get_corresponding_label(image_filename, labels_dir):
    basename, _ = os.path.splitext(image_filename)
    label_filename = basename + '.txt'
    return os.path.join(labels_dir, label_filename)

def draw_annotations(image, annotations, class_names):
    for ann in annotations:
        if len(ann) < 5:
            print("Warning: Annotation does not have enough fields. Skipping this annotation.")
            continue  # Not enough data to draw
        try:
            class_id = int(ann[0])
            x_center = float(ann[1])
            y_center = float(ann[2])
            width = float(ann[3])
            height = float(ann[4])
            
            # Convert normalized coordinates to absolute
            img_height, img_width = image.shape[:2]
            x_center_abs = x_center * img_width
            y_center_abs = y_center * img_height
            width_abs = width * img_width
            height_abs = height * img_height
            
            x_min = int(x_center_abs - width_abs / 2)
            y_min = int(y_center_abs - height_abs / 2)
            x_max = int(x_center_abs + width_abs / 2)
            y_max = int(y_center_abs + height_abs / 2)
            
            # Draw bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Put class name if available
            if class_names and class_id < len(class_names):
                class_name = class_names[class_id]
                cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 255, 0), 2)
            else:
                cv2.putText(image, str(class_id), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 255, 0), 2)
            
            # Draw segmentation mask if available
            if len(ann) > 5:
                mask_coords = list(map(float, ann[5:]))
                if len(mask_coords) % 2 != 0:
                    print("Warning: Segmentation mask coordinates are not in (x, y) pairs. Skipping mask.")
                    continue
                points = []
                for i in range(0, len(mask_coords), 2):
                    x = int(mask_coords[i] * img_width)
                    y = int(mask_coords[i+1] * img_height)
                    points.append([x, y])
                points = np.array(points, dtype=np.int32)
                cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=2)
                cv2.fillPoly(image, [points], color=(255, 0, 0))
        except ValueError:
            print("Warning: Invalid annotation values. Skipping this annotation.")
            continue
    return image

# ======================================
# === MAIN SCRIPT ======================
# ======================================

def main():
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    # Get list of images
    images = [f for f in os.listdir(IMAGES_DIR) if os.path.splitext(f)[1].lower() in image_extensions]
    images.sort()  # Optional: sort the list
    
    if not images:
        print("No images found in the specified images directory.")
        return
    
    print(f"Found {len(images)} images in '{IMAGES_DIR}'.")
    
    idx = 0  # Start index
    total_images = len(images)
    
    while 0 <= idx < total_images:
        image_filename = images[idx]
        image_path = os.path.join(IMAGES_DIR, image_filename)
        label_path = get_corresponding_label(image_filename, LABELS_DIR)
        
        if not os.path.exists(label_path):
            print(f"[{idx + 1}/{total_images}] Label file '{label_path}' not found for image '{image_filename}'. Skipping.")
            idx += 1
            continue
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[{idx + 1}/{total_images}] Failed to read image '{image_path}'. Skipping.")
            idx += 1
            continue
        
        # Read label
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        annotations = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue  # Skip empty lines
            annotations.append(parts)
        
        # Draw annotations
        annotated_image = draw_annotations(image.copy(), annotations, class_names)
        
        # Display the image
        window_name = f"Annotation {idx + 1}/{total_images}: {image_filename}"
        
        # Resize window to fit on screen if necessary
        screen_res = (1280, 720)
        scale_width = screen_res[0] / annotated_image.shape[1]
        scale_height = screen_res[1] / annotated_image.shape[0]
        scale = min(scale_width, scale_height)
        if scale < 1:
            window_width = int(annotated_image.shape[1] * scale)
            window_height = int(annotated_image.shape[0] * scale)
            annotated_image = cv2.resize(annotated_image, (window_width, window_height))
        
        while True:
            # Display instructions
            display_image = annotated_image.copy()
            instructions = "Press 'e' for next, 'q' for previous, 'c' to jump to image, 'Esc' to exit."
            cv2.putText(display_image, instructions, (10, 25), cv2.FONT_HERSHEY_COMPLEX, 
                        1.0, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window_name, display_image)
            key = cv2.waitKey(0) & 0xFF  # Wait for key press
            
            if key == ord('e'):  # Next image
                idx += 1
                break
            elif key == ord('q'):  # Previous image
                idx -= 1
                if idx < 0:
                    idx = 0
                    print("Already at the first image.")
                break
            elif key == ord('c'):  # Jump to image number
                # Enter input mode
                input_str = ''
                input_mode = True
                while input_mode:
                    # Display input prompt
                    prompt_image = display_image.copy()
                    prompt_text = f"Enter image number (1-{total_images}): {input_str}"
                    cv2.putText(prompt_image, prompt_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow(window_name, prompt_image)
                    key_input = cv2.waitKey(0) & 0xFF
                    if ord('0') <= key_input <= ord('9'):
                        input_str += chr(key_input)
                    elif key_input == 8 or key_input == 255:  # Backspace (some systems use 8, others 255)
                        input_str = input_str[:-1]
                    elif key_input == 13 or key_input == 10:  # Enter key
                        if input_str.isdigit():
                            new_idx = int(input_str) - 1
                            if 0 <= new_idx < total_images:
                                idx = new_idx
                                input_mode = False
                                break
                            else:
                                print(f"Invalid image number. Please enter a number between 1 and {total_images}.")
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
                print("Exiting annotation check.")
                cv2.destroyAllWindows()
                return
            else:
                print("Invalid key. Please press 'e', 'q', 'c', or 'Esc'.")
        cv2.destroyAllWindows()
    print("Annotation checking completed.")

if __name__ == "__main__":
    main()