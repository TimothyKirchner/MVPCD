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
        (1920, 1080): sl.RESOLUTION.HD1080,
    }
    init_resolution = resolution_mapping.get(tuple(resolution), sl.RESOLUTION.HD720)
    if tuple(resolution) not in resolution_mapping:
        print("Unsupported resolution provided. Defaulting to HD720 (1280x720).")
    
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
    imagestaken = False

    while True:
        err = zed.grab(runtime_parameters)
        if err == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            frame = image_zed.get_data()
            # Convert to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            cv2.imshow('Workspace - Press "c" to capture, "q" to quit', frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # Save the captured frame
                imagestaken = True
                print("imagestaken = ", imagestaken)
                workspace_images.append(frame_bgr.copy())
                print(f"Captured workspace image {len(workspace_images)}")
            elif key == ord('q'):
                if imagestaken == True:
                    print("imagestaken = ", imagestaken)
                    print("Images of the Background were taken and are now goin to be sliced for the background mosaic.")
                    break
                elif imagestaken == False:
                    print("imagestaken = ", imagestaken)
                    print("There havent been any images taken yet. Please take some Images for the background first. You can take them by pressing \"c\"")
                else:
                    print("well this shouldnt have happened at all!")
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

def crop_rois(images_with_rois):
    """
    Crop the images based on the selected ROIs.
    Returns a list of cropped ROI images.
    """
    cropped_rois = []
    for idx, (image, rois) in enumerate(images_with_rois):
        for roi_idx, roi in enumerate(rois):
            x_min, y_min, x_max, y_max = roi
            cropped_roi = image[y_min:y_max, x_min:x_max].copy()
            if cropped_roi.size == 0:
                print(f"Warning: ROI {roi_idx + 1} in image {idx + 1} is empty. Skipping.")
                continue
            cropped_rois.append(cropped_roi)
            print(f"Cropped ROI {roi_idx + 1} from image {idx + 1} and added to fragments.")

    print(f"Total cropped ROIs: {len(cropped_rois)}")
    return cropped_rois

def save_rois(cropped_rois, output_dir):
    """
    Saves the cropped ROIs into the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    for idx, roi in enumerate(cropped_rois):
        roi_filename = f"background_roi_{idx+1}.png"
        roi_path = os.path.join(output_dir, roi_filename)
        cv2.imwrite(roi_path, roi)
        print(f"Saved ROI fragment: {roi_path}")

def capture_backgrounds(config, max_retries=5):
    """
    Main function to capture workspace images, select ROIs, crop them, and save as background fragments.
    """
    print("It is adviced to designate the Workspace you want to work in, in realistic conditions, take a picture with c, then take more pictures if the workspace can be varied in its condition (clean/dirty, blue matt on table/green matt). Then you exit the capture with q and draw in bounding boxes in the area of the workspace. It is good and encouraged to draw in many of various choices. \"Random\" Boxes are fine as they give the mosaic algorithm more choice of sizes to fill out the background. ")
    # Step 1: Capture workspace images
    workspace_images = capture_workspace_images(config, max_retries=max_retries)

    # Step 2: Allow user to select ROIs on images
    images_with_rois = select_rois_on_images(workspace_images)

    # Step 3: Crop ROIs from images
    cropped_rois = crop_rois(images_with_rois)

    if not cropped_rois:
        print("No ROI fragments created. Ensure ROIs are correctly selected.")
        sys.exit(1)

    # Step 4: Save cropped ROIs into MVPCD/data/backgrounds
    backgrounds_dir = os.path.join(project_root, 'data', 'backgrounds')
    save_rois(cropped_rois, backgrounds_dir)

    print("Background capturing and processing completed.")

if __name__ == "__main__":
    config = load_config()
    # You can pass max_retries here if needed, e.g., max_retries=5
    capture_backgrounds(config, max_retries=5)