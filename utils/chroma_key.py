# ~/Desktop/MVPCD/utils/chroma_key.py

import cv2
import numpy as np

def apply_chroma_key(image, lower_color, upper_color):
    """
    Removes the background based on the specified color ranges.
    """
    mask = cv2.inRange(image, lower_color, upper_color)
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(image, image, mask=mask_inv)
    return result
