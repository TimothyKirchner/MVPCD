## MVPCD: Multi-View Point Cloud Detection Pipeline
# Table of Contents
1. Introduction
2. Features
3. Prerequisites
4. Installation
5. Directory Structure
6. Configuration
7. Usage
   1. Adding a New Object Class
   2. Adjusting Depth Thresholds
   3. Adjusting Chroma Key Settings
   4. Capturing Images
   5. Setting the Region of Interest (ROI)
   6. Preprocessing Images
   7. Splitting the Dataset
   8. Training the YOLOv8 Model
   9. Running Inference
8. Scripts Overview
9. Utilities
10. Contributing
11. License
12. Acknowledgments


## 1. Introduction

The MVPCD project is a comprehensive pipeline for creating custom object detection models using the YOLOv8 architecture. It leverages depth sensing and chroma keying techniques to facilitate efficient data collection, preprocessing, annotation, training, and inference.

By integrating a ZED stereo camera for depth mapping and utilizing chroma keying (green screen), the pipeline automates the annotation process, reducing manual efforts and improving the accuracy of object detection models.

# 2. Features

- Depth Sensing Integration: Uses ZED stereo camera to capture depth maps.
- Chroma Keying: Applies chroma key techniques to isolate objects based on color thresholds.
- Automated Annotation: Combines depth and chroma key masks to automate bounding box extraction.
- Custom Object Detection: Allows training of YOLOv8 models on user-defined objects.
- Live Adjustment Tools: Real-time adjustment of depth and chroma key thresholds.
- Region of Interest (ROI): Focus detection on specific areas within the camera frame.
- Dataset Management: Automated splitting of data into training and validation sets.
- Live Inference: Run the trained model on live camera feed for real-time object detection.
- Debugging Tools: Save intermediate masks and images for troubleshooting.

# 3. Prerequisites
1. Hardware:
- ZED Stereo Camera
2. Software:
- Python 3.7 or higher
- CUDA Toolkit (for GPU acceleration)
- ZED SDK (for camera integration)
3. Installation
- Clone the Repository:

bash
Copy code
git clone https://github.com/TimothyKirchner/MVPCD/tree/remaster
cd MVPCD
Create a Virtual Environment (optional but recommended):

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:

bash
Copy code
pip install -r requirements.txt
Note: Ensure that the ZED SDK is properly installed before proceeding.

Download YOLOv8 Pre-trained Weights:

Place the yolov8s.pt file in the root directory (MVPCD/).

Directory Structure
kotlin
Copy code
MVPCD/
├── .gitignore
├── README.md
├── yolov8s.pt
├── config/
│   └── config.yaml
├── data/
│   ├── debug/
│   │   ├── bboxes/
│   │   ├── combined_mask/
│   │   ├── contours/
│   │   ├── depthmask/
│   │   └── rgbmask/
│   ├── depth_maps/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── mvpcd.yaml
├── runs/
│   └── detect/
│       └── mvpcd_yolov8/
│           └── weights/
│               └── best.pt
├── scripts/
│   ├── __init__.py
│   ├── annotate.py
│   ├── capture.py
│   ├── live_depth_feed.py
│   ├── live_rgb_chromakey.py
│   ├── main.py
│   ├── preprocess.py
│   ├── run_inference.py
│   ├── set_roi.py
│   ├── split_dataset.py
│   ├── test_camera.py
│   ├── train_model.py
│   ├── verify_annotations.py
│   └── view_depth.py
└── utils/
    ├── __init__.py
    ├── annotation_utils.py
    ├── bbox_utils.py
    ├── camera_utils.py
    ├── chroma_key.py
    └── depth_processing.py
Configuration
All configurations are managed through the config/config.yaml file. This file includes settings for:

Camera Parameters: FPS, resolution, etc.
Capture Settings: Number of images, interval between captures.
Chroma Key Thresholds: HSV color ranges for chroma keying.
Depth Thresholds: Minimum and maximum depth values for filtering.
Class Names: List of object classes to detect.
Debug Directories: Paths to save intermediate images and masks.
Output Directories: Paths for images, labels, and depth maps.
Example config.yaml:

yaml
Copy code
camera:
  fps: 30
  resolution:
    - 1280
    - 720
capture:
  interval: 0.25
  num_images: 50
chroma_key:
  lower_color:
    - 0
    - 0
    - 0
  upper_color:
    - 179
    - 4
    - 255
class_names: []
debug:
  bboxes: data/debug/bboxes
  combined_mask: data/debug/combined_mask
  contours: data/debug/contours
  depthmask: data/debug/depthmask
  rgbmask: data/debug/rgbmask
depth_thresholds: {}
image_counters: {}
output:
  depth_dir: data/depth_maps
  image_dir: data/images
  label_dir: data/labels
  train_image_dir: data/images/train
  train_label_dir: data/labels/train
  val_image_dir: data/images/val
  val_label_dir: data/labels/val
rois: []
Usage
The main script orchestrates the entire pipeline. Run it using:

bash
Copy code
python scripts/main.py
The script will guide you through the following steps:

1. Adding a New Object Class
Prompt: "Do you want to add an object? (y/n)"
Action: If 'y', you'll be asked to input the class name.
Note: Class names must be unique and not empty.
1. Adjusting Depth Thresholds
Purpose: Fine-tune depth thresholds to filter objects based on distance.
Script: live_depth_feed.py
Usage:
Press 'r' to restart the viewer.
Press 'v' to restart the OpenCV window.
Press 'q' to quit and save values.
1. Adjusting Chroma Key Settings
Purpose: Adjust HSV color thresholds for effective chroma keying.
Script: live_rgb_chromakey.py
Usage:
Use trackbars to adjust HSV values.
Press 'q' to quit and save settings.
1. Capturing Images
Script: capture.py
Action: Captures images and corresponding depth maps for the specified class.
Settings: Number of images and interval between captures can be adjusted in config.yaml.
1. Setting the Region of Interest (ROI)
Script: set_roi.py
Action: Define specific areas in the frame to focus the object detection.
Usage:
Use the mouse to draw ROIs on the captured image.
Press 'q' to finish selecting ROIs.
1. Preprocessing Images
Script: preprocess.py
Action: Applies chroma key and depth masks, extracts bounding boxes, and generates labels.
Debugging: Intermediate masks and images are saved in data/debug/ for review.
1. Splitting the Dataset
Script: split_dataset.py
Action: Splits images and labels into training and validation sets, ensuring balanced class representation.
1. Training the YOLOv8 Model
Script: train_model.py
Action: Trains the YOLOv8 model using the prepared dataset.
Parameters:
Epochs: Default is 100.
Learning Rate: Default is 0.0001.
Batch Size: Default is 8.
Usage: Input desired values when prompted.
1. Running Inference
Script: run_inference.py
Action: Runs the trained model on live camera feed for real-time object detection.
Usage: Press 'q' to exit the inference mode.
Scripts Overview
main.py: Orchestrates the entire pipeline.
capture.py: Captures images and depth maps.
live_depth_feed.py: Adjusts depth thresholds in real-time.
live_rgb_chromakey.py: Adjusts chroma key settings in real-time.
set_roi.py: Allows users to define ROIs.
preprocess.py: Processes images to extract bounding boxes and create labels.
split_dataset.py: Splits data into training and validation sets.
train_model.py: Trains the YOLOv8 model.
run_inference.py: Runs live inference using the trained model.
verify_annotations.py: Allows users to review and verify annotations.
test_camera.py: Tests camera functionality.
view_depth.py: Views depth maps for debugging.
Utilities
camera_utils.py: Handles camera initialization and frame capture.
chroma_key.py: Applies chroma key masks to images.
depth_processing.py: Processes depth maps to create depth masks.
bbox_utils.py: Contains functions for bounding box conversion and drawing.
annotation_utils.py: Assists in creating YOLO-formatted annotations.
Contributing
Contributions are welcome! Please follow these steps:

Fork the Repository

Create a Branch

bash
Copy code
git checkout -b feature/YourFeature
Commit Your Changes

bash
Copy code
git commit -m "Add your message"
Push to the Branch

bash
Copy code
git push origin feature/YourFeature
Open a Pull Request

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
YOLOv8: Ultralytics YOLOv8
ZED Stereo Camera: Stereolabs ZED
Contributors: Thanks to all who have contributed to this project.