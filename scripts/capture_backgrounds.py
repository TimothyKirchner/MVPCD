# scripts/capture_backgrounds.py

import sys
import os
import cv2
import numpy as np
import yaml
import time

# Adjust sys.path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import ZED SDK
import pyzed.sl as sl

def load_config(config_path='config/config.yaml'):
    """Load the YAML configuration file."""
    config_full_path = os.path.join(project_root, config_path)
    if not os.path.exists(config_full_path):
        print(f"Configuration file not found at {config_full_path}.")
        sys.exit(1)
    with open(config_full_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(config, config_path='config/config.yaml'):
    """Save the YAML configuration file."""
    config_full_path = os.path.join(project_root, config_path)
    with open(config_full_path, 'w') as file:
        yaml.dump(config, file)

def capture_workspace_images(config, max_retries=5):
    """
    Capture one or more images of the workspace using the ZED camera.
    Implements retry logic for camera initialization.
    """
    camera_config = config.get('camera', {})
    resolution = camera_config.get('resolution', [1280, 720])
    fps = camera_config.get('fps', 30)

    # Initialize ZED Camera with retries
    zed = sl.Camera()
    init_params = sl.InitParameters()
    
    # Map resolution from [width, height] to ZED SDK resolution enum
    resolution_mapping = {
        (1280, 720): sl.RESOLUTION.HD720,
    }
    init_resolution = resolution_mapping.get(tuple(resolution), sl.RESOLUTION.HD720)
    if tuple(resolution) not in resolution_mapping:
        print("Resolution not recognized. Using default HD720.")
    
    init_params.camera_resolution = init_resolution
    init_params.camera_fps = fps
    init_params.depth_mode = sl.DEPTH_MODE.NONE  # Disable depth sensing if not needed

    attempt = 0
    while attempt < max_retries:
        status = zed.open(init_params)
        if status == sl.ERROR_CODE.SUCCESS:
            print("ZED camera initialized successfully.")
            break
        else:
            print(f"Failed to open ZED camera: {status}. Retrying ({attempt + 1}/{max_retries})...")
            attempt += 1
            time.sleep(2)  # Wait before retrying

    if attempt == max_retries:
        print("Exceeded maximum retries. Exiting.")
        sys.exit(1)

    # Prepare image capture
    image_zed = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    workspace_images = []

    print('Press "c" to capture an image, "q" to quit.')

    while True:
        err = zed.grab(runtime_parameters)
        if err == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            frame = image_zed.get_data()
            # Convert to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            cv2.imshow('Workspace - Press "c" to capture, "q" to quit', frame_bgr)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                # Save the captured frame
                workspace_images.append(frame_bgr.copy())
                print(f"Captured workspace image {len(workspace_images)}")
            elif key == ord('q'):
                break
        else:
            print("Failed to grab image from camera.")

    zed.close()
    cv2.destroyAllWindows()

    if not workspace_images:
        print("No workspace images captured.")
        sys.exit(1)

    return workspace_images

def select_rois_on_images(images):
    """
    Allow the user to mark one or more bounding boxes (ROIs) on each image.
    Returns a list of tuples containing image and list of ROIs for that image.
    """
    images_with_rois = []

    for idx, image in enumerate(images):
        clone = image.copy()
        rois = []

        print(f"\nSelect ROIs on image {idx + 1}.")
        print("Instructions:")
        print("- Draw rectangles with the mouse to select ROIs.")
        print("- Press 's' to save the ROIs and proceed to the next image.")
        print("- Press 'r' to reset the ROIs on the current image.")
        print("- Press 'q' to quit.")

        roi_window_name = f"Image {idx + 1} - Select ROIs"

        def select_roi(event, x, y, flags, param):
            nonlocal roi_start, drawing, rois, image

            if event == cv2.EVENT_LBUTTONDOWN:
                roi_start = (x, y)
                drawing = True

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    image = clone.copy()
                    cv2.rectangle(image, roi_start, (x, y), (0, 255, 0), 2)
                    for roi in rois:
                        cv2.rectangle(image, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 255), 2)

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                roi_end = (x, y)
                x1, y1 = roi_start
                x2, y2 = roi_end
                x_min, x_max = sorted([x1, x2])
                y_min, y_max = sorted([y1, y2])
                rois.append((x_min, y_min, x_max, y_max))
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        roi_start = (0, 0)
        drawing = False

        cv2.namedWindow(roi_window_name)
        cv2.setMouseCallback(roi_window_name, select_roi)

        while True:
            cv2.imshow(roi_window_name, image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                if rois:
                    images_with_rois.append((clone.copy(), rois.copy()))
                    print(f"ROIs saved for image {idx + 1}.")
                else:
                    print("No ROIs selected on this image.")
                break
            elif key == ord('r'):
                # Reset ROIs
                rois.clear()
                image = clone.copy()
                print("ROIs reset.")
            elif key == ord('q'):
                cv2.destroyWindow(roi_window_name)
                sys.exit(0)

        cv2.destroyWindow(roi_window_name)

    return images_with_rois

def split_rois_into_fragments(images_with_rois, fragment_size=480):
    """
    Splits the designated workspace area into many 480x480 size picture fragments.
    Returns a list of fragments.
    """
    fragments = []
    fragment_counter = 0

    for idx, (image, rois) in enumerate(images_with_rois):
        img_height, img_width = image.shape[:2]

        for roi_idx, roi in enumerate(rois):
            x_min, y_min, x_max, y_max = roi

            roi_width = x_max - x_min
            roi_height = y_max - y_min

            # Calculate the number of fragments in x and y directions
            x_steps = max(1, roi_width // fragment_size)
            y_steps = max(1, roi_height // fragment_size)

            if x_steps == 1:
                x_overlap = 0
            else:
                x_overlap = (roi_width - fragment_size * x_steps) // max(1, x_steps - 1)

            if y_steps == 1:
                y_overlap = 0
            else:
                y_overlap = (roi_height - fragment_size * y_steps) // max(1, y_steps - 1)

            for i in range(x_steps):
                for j in range(y_steps):
                    x_start = x_min + i * (fragment_size + x_overlap)
                    y_start = y_min + j * (fragment_size + y_overlap)
                    x_end = x_start + fragment_size
                    y_end = y_start + fragment_size

                    # Ensure the fragment is within image boundaries
                    if x_end > img_width:
                        x_end = img_width
                        x_start = x_end - fragment_size
                    if y_end > img_height:
                        y_end = img_height
                        y_start = y_end - fragment_size

                    fragment = image[y_start:y_end, x_start:x_end].copy()
                    fragments.append(fragment)
                    fragment_counter += 1

    print(f"Total fragments created: {fragment_counter}")
    return fragments

def save_fragments(fragments, output_dir):
    """
    Saves the fragments into the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    for idx, fragment in enumerate(fragments):
        fragment_filename = f"background_{idx+1}.png"
        fragment_path = os.path.join(output_dir, fragment_filename)
        cv2.imwrite(fragment_path, fragment)
        print(f"Saved fragment: {fragment_path}")

def capture_backgrounds(config, max_retries=5):
    """
    Main function to capture workspace images, select ROIs, split into fragments, and save them.
    """
    # Step 1: Capture workspace images
    workspace_images = capture_workspace_images(config, max_retries=max_retries)

    # Step 2: Allow user to select ROIs on images
    images_with_rois = select_rois_on_images(workspace_images)

    # Step 3: Split ROIs into fragments
    fragments = split_rois_into_fragments(images_with_rois, fragment_size=480)

    # Step 4: Save fragments into MVPCD/data/backgrounds
    backgrounds_dir = os.path.join(project_root, 'data', 'backgrounds')
    save_fragments(fragments, backgrounds_dir)

    print("Background capturing and processing completed.")

if __name__ == "__main__":
    config = load_config()
    # You can pass max_retries here if needed, e.g., max_retries=5
    capture_backgrounds(config, max_retries=5)