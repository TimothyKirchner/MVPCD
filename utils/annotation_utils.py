# utils/annotation_utils.py
def create_annotation(image_shape, annotation_path, bbox, class_id=0):
    height, width, _ = image_shape
    x, y, w, h = bbox
    x_center = (x + w / 2) / width
    y_center = (y + h / 2) / height
    bbox_width = w / width
    bbox_height = h / height
    return f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n"
