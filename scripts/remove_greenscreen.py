# scripts/remove_greenscreen.py

import sys
import os
import cv2
import numpy as np
import yaml
import random

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

                # Create a new image with alpha channel
                bgr = image
                alpha = mask_inv

                # Combine BGR and alpha channel
                bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
                bgra[:, :, 3] = alpha

                # Save the image with transparency
                cv2.imwrite(image_path, bgra)
                print(f"Removed green background and saved image with alpha channel: {image_path}")

def add_random_background(config):
    train_dir = os.path.join(project_root, config['output']['train_image_dir'])
    val_dir = os.path.join(project_root, config['output']['val_image_dir'])

    dirs = [train_dir, val_dir]

    for dir in dirs:
        for filename in os.listdir(dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(dir, filename)
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel if present

                if image is None:
                    print(f"Could not read image: {image_path}")
                    continue

                # Check if image has alpha channel (transparency)
                if image.shape[2] < 4:
                    print(f"Image does not have an alpha channel: {image_path}")
                    continue

                # Separate the alpha channel from the image
                bgr = image[:, :, :3]
                alpha = image[:, :, 3]

                # Normalize the alpha mask to range [0,1]
                alpha_mask = alpha.astype(float) / 255.0
                # Convert alpha mask to 3 channels
                alpha_mask_3channel = cv2.merge([alpha_mask, alpha_mask, alpha_mask])

                # Generate a random background that is different for every image
                height, width = bgr.shape[:2]

                # Start with random noise background
                background = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

                # Add random shapes
                num_shapes = random.randint(5, 15)
                for _ in range(num_shapes):
                    shape_type = random.choice(['circle', 'rectangle', 'line'])
                    color = [random.randint(0, 255) for _ in range(3)]

                    if shape_type == 'circle':
                        center = (random.randint(0, width - 1), random.randint(0, height - 1))
                        radius = random.randint(5, min(width, height) // 4)
                        thickness = random.choice([-1, random.randint(1, 5)])  # -1 for filled circles
                        cv2.circle(background, center, radius, color, thickness)
                    elif shape_type == 'rectangle':
                        pt1 = (random.randint(0, width - 1), random.randint(0, height - 1))
                        pt2 = (random.randint(0, width - 1), random.randint(0, height - 1))
                        thickness = random.choice([-1, random.randint(1, 5)])  # -1 for filled rectangles
                        cv2.rectangle(background, pt1, pt2, color, thickness)
                    elif shape_type == 'line':
                        pt1 = (random.randint(0, width - 1), random.randint(0, height - 1))
                        pt2 = (random.randint(0, width - 1), random.randint(0, height - 1))
                        thickness = random.randint(1, 5)  # Thickness must be >= 1 for lines
                        cv2.line(background, pt1, pt2, color, thickness)

                # Convert images to float for blending
                bgr = bgr.astype(float)
                background = background.astype(float)
                alpha_mask_3channel = alpha_mask_3channel.astype(float)

                # Composite the foreground and background using the alpha mask
                foreground = cv2.multiply(alpha_mask_3channel, bgr)
                background = cv2.multiply(1.0 - alpha_mask_3channel, background)
                composite = cv2.add(foreground, background)
                composite = composite.astype(np.uint8)

                # Save the composited image, replacing the original
                cv2.imwrite(image_path, composite)
                print(f"Added random background and saved image: {image_path}")

def crop_images_around_roi(config):
    train_dir = os.path.join(project_root, config['output']['train_image_dir'])
    val_dir = os.path.join(project_root, config['output']['val_image_dir'])

    dirs = [train_dir, val_dir]

    # Get ROIs from config
    rois = config.get('rois', [])
    if not rois:
        print("No ROIs defined in config. Skipping cropping.")
        return

    # Since ROIs might be different for different images, but in this context,
    # we'll assume the same ROI applies to all images.

    # For simplicity, we'll use the first ROI defined
    roi = rois[0]
    roi_center_x = roi['x'] + roi['width'] // 2
    roi_center_y = roi['y'] + roi['height'] // 2

    crop_size = 640  # Desired crop size (640x640)

    for dir in dirs:
        for filename in os.listdir(dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(dir, filename)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Could not read image: {image_path}")
                    continue

                img_height, img_width = image.shape[:2]

                # Calculate the top-left corner of the crop
                x1 = roi_center_x - crop_size // 2
                y1 = roi_center_y - crop_size // 2

                # Ensure the crop coordinates are within image boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = x1 + crop_size
                y2 = y1 + crop_size

                # Adjust if crop goes beyond image boundaries
                if x2 > img_width:
                    x1 = img_width - crop_size
                    x2 = img_width
                if y2 > img_height:
                    y1 = img_height - crop_size
                    y2 = img_height
                x1 = max(0, x1)
                y1 = max(0, y1)

                # Crop the image
                cropped_image = image[y1:y2, x1:x2]

                # If the cropped image is not 640x640 (e.g., near the image edges), resize it
                if cropped_image.shape[0] != crop_size or cropped_image.shape[1] != crop_size:
                    cropped_image = cv2.resize(cropped_image, (crop_size, crop_size))

                # Save the cropped image, replacing the original
                cv2.imwrite(image_path, cropped_image)
                print(f"Cropped and saved image: {image_path}")

def main():
    config = load_config()
    adjust_green_thresholds(config)
    remove_green_background(config)
    print("Green screen removal completed.")
    add_random_background(config)
    print("Random backgrounds added to images.")
    crop_images_around_roi(config)
    print("Images cropped around ROI.")

if __name__ == "__main__":
    main()