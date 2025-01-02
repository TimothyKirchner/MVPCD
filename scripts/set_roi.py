# scripts/set_roi.py
import sys
import os
import time  # Added for timer functionality
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import cv2
import yaml
from utils.camera_utils import initialize_camera, capture_frame

def load_config(config_path='config/config.yaml'):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, config_path)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(config, config_path='config/config.yaml'):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, config_path)
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

def set_rois(config, class_name, angle_index):
    camera = initialize_camera(config)
    image, _ = capture_frame(camera)
    camera.close()

    if image is None:
        print("Could not capture image from camera.")
        return

    rois = []
    drawing = False  # True if mouse is pressed
    ix, iy = -1, -1  # Initial x,y coordinates

    clone = image.copy()
    cv2.namedWindow("Set ROI")

    last_restart_time = time.time()  # Initialize timer

    def draw_rectangle(event, x, y, flags, param):
        nonlocal ix, iy, drawing, image, rois
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                image = clone.copy()
                cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 0), 2)
            x1, y1 = min(ix, x), min(iy, y)
            x2, y2 = max(ix, x), max(iy, y)
            rois.append({'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1})
            print(f"ROI added: {rois[-1]}")

    cv2.setMouseCallback("Set ROI", draw_rectangle)

    print("Draw ROIs by dragging the mouse.")
    print("Press 'q' to finish selecting ROIs.")
    print("Press 'r' to reinitialize the camera and viewer.")
    print("Press 'v' to close and reopen the OpenCV window.")

    try:
        while True:
            cv2.imshow("Set ROI", image)
            key = cv2.waitKey(1) & 0xFF

            # Handle key presses
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Reinitializing the camera and viewer...")
                try:
                    # Reinitialize camera
                    camera = initialize_camera(config)
                    image, _ = capture_frame(camera)
                    camera.close()
                    if image is None:
                        print("Could not capture image from camera.")
                        break
                    clone = image.copy()
                except Exception as e:
                    print(f"Error during reinitialization: {e}")
                    break
                last_restart_time = time.time()  # Reset timer
            elif key == ord('v'):
                print("Closing and reopening the OpenCV window...")
                cv2.destroyWindow("Set ROI")
                cv2.namedWindow("Set ROI")
                cv2.setMouseCallback("Set ROI", draw_rectangle)
                last_restart_time = time.time()  # Reset timer

            # Automatically restart the window every 2 minutes
            current_time = time.time()
            if current_time - last_restart_time >= 120:
                print("Automatically restarting the OpenCV window after 2 minutes...")
                cv2.destroyWindow("Set ROI")
                cv2.namedWindow("Set ROI")
                cv2.setMouseCallback("Set ROI", draw_rectangle)
                last_restart_time = current_time  # Reset timer

    finally:
        cv2.destroyAllWindows()

    if rois:
        if 'rois' not in config:
            config['rois'] = {}
        if class_name not in config['rois']:
            config['rois'][class_name] = {}
        config['rois'][class_name][angle_index] = rois
        save_config(config)
        print(f"{len(rois)} ROI(s) saved in config/config.yaml for class '{class_name}', angle {angle_index}")
    else:
        print("No ROIs defined.")

if __name__ == "__main__":
    config = load_config()
    if len(sys.argv) < 3:
        print("Usage: python set_rois.py [classname] [angle_index]")
        sys.exit(1)
    class_name = sys.argv[1]
    angle_index = int(sys.argv[2])
    set_rois(config, class_name, angle_index)