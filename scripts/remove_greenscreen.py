# scripts/remove_greenscreen.py

import sys
import os
import cv2
import numpy as np
import yaml

# Adjust sys.path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_config(config_path='config/config.yaml'):
    config_full_path = os.path.join(project_root, config_path)
    with open(config_full_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(config, config_path='config/config.yaml'):
    config_full_path = os.path.join(project_root, config_path)
    with open(config_full_path, 'w') as file:
        yaml.dump(config, file)

def adjust_green_thresholds(config):
    # Get image directory from config
    image_dir = os.path.join(project_root, config['output']['image_dir'])

    # Get a sample image to adjust thresholds
    sample_image = None
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            sample_image_path = os.path.join(image_dir, filename)
            sample_image = cv2.imread(sample_image_path)
            if sample_image is not None:
                break

    if sample_image is None:
        print("No images found in the images directory.")
        return config['chroma_key']['lower_color'], config['chroma_key']['upper_color']

    hsv_sample = cv2.cvtColor(sample_image, cv2.COLOR_BGR2HSV)

    # Load initial thresholds from config
    lower_h, lower_s, lower_v = config.get('chroma_key', {}).get('lower_color', [35, 100, 100])
    upper_h, upper_s, upper_v = config.get('chroma_key', {}).get('upper_color', [85, 255, 255])

    cv2.namedWindow("Adjust Green Thresholds")
    cv2.createTrackbar('Lower H', 'Adjust Green Thresholds', lower_h, 179, lambda x: None)
    cv2.createTrackbar('Lower S', 'Adjust Green Thresholds', lower_s, 255, lambda x: None)
    cv2.createTrackbar('Lower V', 'Adjust Green Thresholds', lower_v, 255, lambda x: None)
    cv2.createTrackbar('Upper H', 'Adjust Green Thresholds', upper_h, 179, lambda x: None)
    cv2.createTrackbar('Upper S', 'Adjust Green Thresholds', upper_s, 255, lambda x: None)
    cv2.createTrackbar('Upper V', 'Adjust Green Thresholds', upper_v, 255, lambda x: None)

    while True:
        lower_h = cv2.getTrackbarPos('Lower H', 'Adjust Green Thresholds')
        lower_s = cv2.getTrackbarPos('Lower S', 'Adjust Green Thresholds')
        lower_v = cv2.getTrackbarPos('Lower V', 'Adjust Green Thresholds')
        upper_h = cv2.getTrackbarPos('Upper H', 'Adjust Green Thresholds')
        upper_s = cv2.getTrackbarPos('Upper S', 'Adjust Green Thresholds')
        upper_v = cv2.getTrackbarPos('Upper V', 'Adjust Green Thresholds')

        lower_bound = np.array([lower_h, lower_s, lower_v])
        upper_bound = np.array([upper_h, upper_s, upper_v])

        mask = cv2.inRange(hsv_sample, lower_bound, upper_bound)
        mask_inv = cv2.bitwise_not(mask)
        result = cv2.bitwise_and(sample_image, sample_image, mask=mask_inv)

        cv2.imshow("Adjust Green Thresholds", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Update config with new thresholds
            config['chroma_key']['lower_color'] = [int(lower_h), int(lower_s), int(lower_v)]
            config['chroma_key']['upper_color'] = [int(upper_h), int(upper_s), int(upper_v)]
            save_config(config)
            print("Green thresholds updated in config.")
            break

    cv2.destroyAllWindows()
    return config['chroma_key']['lower_color'], config['chroma_key']['upper_color']

def remove_green_background(config):
    # Get image directory from config
    train_dir = os.path.join(project_root, config['output']['train_image_dir'])
    val_dir = os.path.join(project_root, config['output']['val_image_dir'])

    dirs = [train_dir, val_dir]

    lower_color = np.array(config.get('chroma_key', {}).get('lower_color', [35, 100, 100]), dtype=np.uint8)
    upper_color = np.array(config.get('chroma_key', {}).get('upper_color', [85, 255, 255]), dtype=np.uint8)

    for dir in dirs:
        for filename in os.listdir(dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(dir, filename)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Could not read image: {image_path}")
                    continue

                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower_color, upper_color)
                mask_inv = cv2.bitwise_not(mask)

                # Optional: Apply morphological operations to clean up the mask
                kernel = np.ones((3,3), np.uint8)
                mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel)
                mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_DILATE, kernel)

                result = cv2.bitwise_and(image, image, mask=mask_inv)

                # Save the processed image, overwriting the original
                cv2.imwrite(image_path, result)
                print(f"Processed and saved image: {image_path}")

def main():
    config = load_config()
    adjust_green_thresholds(config)
    remove_green_background(config)
    print("Green screen removal completed for all images.")

if __name__ == "__main__":
    main()
