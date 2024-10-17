# ~/Desktop/MVPCD/utils/camera_utils.py

import pyzed.sl as sl
import cv2
import numpy as np

def initialize_camera(config):
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = config['camera']['fps']
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Set depth mode to NEURAL
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Set units to millimeters
    init_params.depth_minimum_distance = config.get('depth_threshold', {}).get('min_distance', 300)  # Minimum depth distance in millimeters
    init_params.depth_maximum_distance = config.get('depth_threshold', {}).get('max_distance', 2000)  # Maximum depth distance in millimeters

    camera = sl.Camera()
    status = camera.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open camera: {status}")
        return None
    return camera

def capture_frame(camera):
    runtime_parameters = sl.RuntimeParameters()
    image_zed = sl.Mat()
    depth_zed = sl.Mat()

    if camera.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        camera.retrieve_image(image_zed, sl.VIEW.LEFT)
        camera.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
        image = image_zed.get_data()
        depth = depth_zed.get_data()
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaNs and Infs with 0
        return image[:, :, :3], depth  # Return BGR image and depth map
    else:
        return None, None
