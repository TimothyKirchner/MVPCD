# ~/Desktop/MVPCD/utils/bbox_utils.py

import cv2

def convert_bbox_to_yolo(image_shape, bbox, class_id=0):
    height, width, _ = image_shape
    x, y, w, h = bbox
    x_center = (x + w / 2) / width
    y_center = (y + h / 2) / height
    bbox_width = w / width
    bbox_height = h / height
    return f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n"

def draw_bounding_boxes(image, bboxes, color=(0, 255, 0), thickness=2):
    """
    Draws bounding boxes on the image.

    Args:
        image (numpy.ndarray): The image on which to draw the bounding boxes.
        bboxes (list of tuples): List of bounding boxes in the format (x, y, w, h).
        color (tuple): Color of the bounding boxes.
        thickness (int): Thickness of the bounding box lines.

    Returns:
        numpy.ndarray: The image with drawn bounding boxes.
    """
    for bbox in bboxes:
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    return image
