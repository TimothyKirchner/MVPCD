# utils/bbox_utils.py
import cv2

def convert_bbox_to_yolo(image_width, image_height, bbox, class_id):
    """
    Convert bounding box to YOLO format.

    Parameters:
    - image_width (int): Width of the image.
    - image_height (int): Height of the image.
    - bbox (tuple): Bounding box in (x_min, y_min, x_max, y_max) format.
    - class_id (int): Class ID.

    Returns:
    - str: YOLO formatted string.
    """
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min

    # Normalize the coordinates
    x_center_norm = x_center / image_width
    y_center_norm = y_center / image_height
    width_norm = width / image_width
    height_norm = height / image_height

    # Ensure the normalized values are within [0, 1]
    x_center_norm = min(max(x_center_norm, 0), 1)
    y_center_norm = min(max(y_center_norm, 0), 1)
    width_norm = min(max(width_norm, 0), 1)
    height_norm = min(max(height_norm, 0), 1)

    return f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"

def draw_bounding_boxes(image, bboxes, format='xywh', color=(255, 0, 0), thickness=2):
    """
    Draw bounding boxes on the image.

    Parameters:
    - image (numpy.ndarray): The image on which to draw.
    - bboxes (list of tuples): List of bounding boxes.
    - format (str): Format of the bounding boxes. 'xywh' for (x, y, w, h) and 'xyxy' for (x_min, y_min, x_max, y_max).
    - color (tuple): Color of the bounding box.
    - thickness (int): Thickness of the bounding box lines.

    Returns:
    - numpy.ndarray: Image with bounding boxes drawn.
    """
    for bbox in bboxes:
        if format == 'xywh':
            x, y, w, h = bbox
            start_point = (int(x), int(y))
            end_point = (int(x + w), int(y + h))
        elif format == 'xyxy':
            x_min, y_min, x_max, y_max = bbox
            start_point = (int(x_min), int(y_min))
            end_point = (int(x_max), int(y_max))
        else:
            raise ValueError("Invalid format for bounding boxes. Use 'xywh' or 'xyxy'.")

        cv2.rectangle(image, start_point, end_point, color, thickness)
    return image
