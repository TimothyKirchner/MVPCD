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
            x_center_abs = int(x_center * img_width)
            y_center_abs = int(y_center * img_height)
            width_abs = int(width * img_width)
            height_abs = int(height * img_height)
            
            x_min = x_center_abs - width_abs // 2
            y_min = y_center_abs - height_abs // 2
            x_max = x_center_abs + width_abs // 2
            y_max = y_center_abs + height_abs // 2
            
            # Draw bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Put class name if available
            if class_names and class_id < len(class_names):
                class_name = class_names[class_id]
                cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)
            else:
                cv2.putText(image, str(class_id), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)
            
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
    
    for idx, image_filename in enumerate(images, 1):
        image_path = os.path.join(IMAGES_DIR, image_filename)
        label_path = get_corresponding_label(image_filename, LABELS_DIR)
        
        if not os.path.exists(label_path):
            print(f"[{idx}/{len(images)}] Label file '{label_path}' not found for image '{image_filename}'. Skipping.")
            continue
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[{idx}/{len(images)}] Failed to read image '{image_path}'. Skipping.")
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
        window_name = f"Annotation {idx}/{len(images)}: {image_filename}"
        cv2.imshow(window_name, annotated_image)
        print(f"[{idx}/{len(images)}] Displaying '{image_filename}'. Press any key to continue or 'q' to quit.")
        
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        
        if key == ord('q') or key == ord('Q'):
            print("Quitting annotation check.")
            break
    
    print("Annotation checking completed.")

if __name__ == "__main__":
    main()
