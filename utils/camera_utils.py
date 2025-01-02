# utils/camera_utils.py
import pyzed.sl as sl
import cv2
import numpy as np
import time

def initialize_camera(config, max_retries=5):
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = config['camera']['fps']
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.depth_minimum_distance = config.get('depth_threshold', {}).get('min', 300)
    init_params.depth_maximum_distance = config.get('depth_threshold', {}).get('max', 2000)

    camera = sl.Camera()
    for attempt in range(1, max_retries + 1):
        status = camera.open(init_params)
        if status == sl.ERROR_CODE.SUCCESS:
            print(f"Camera initialized successfully on attempt {attempt}.")
            return camera
        else:
            print(f"Attempt {attempt}: Failed to open camera: {status}")
            camera.close()
            time.sleep(1)  # Wait a moment before retrying
            camera = sl.Camera()  # Re-instantiate the camera object

    print("Exceeded maximum retry attempts. Unable to initialize the camera.")
    return None

def capture_frame(camera):
    runtime_parameters = sl.RuntimeParameters()
    image_zed = sl.Mat()
    depth_zed = sl.Mat()

    if camera.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        camera.retrieve_image(image_zed, sl.VIEW.LEFT)
        camera.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
        image = image_zed.get_data()
        depth = depth_zed.get_data()
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        return image[:, :, :3], depth
    else:
        return None, None
