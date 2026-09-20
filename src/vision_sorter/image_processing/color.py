"""OpenCV HSV ranges: hue 0..179, saturation/value 0..255."""
import cv2
import numpy as np
from vision_sorter.image_processing.reader import Image
from vision_sorter.image_processing.preprocessing import validate


def validate_hsv(lower: tuple[int, int, int], upper: tuple[int, int, int]) -> None:
    if len(lower) != 3 or len(upper) != 3:
        raise ValueError("HSV bounds must have three components")
    if any(not 0 <= lo <= hi <= maximum for lo, hi, maximum in zip(lower, upper, (179, 255, 255))):
        raise ValueError("Invalid HSV bounds; split hue wraparound into two masks")


def hsv_mask(image: Image, lower: tuple[int, int, int], upper: tuple[int, int, int]) -> Image:
    validate(image)
    validate_hsv(lower, upper)
    if image.ndim != 3:
        raise ValueError("HSV masking requires a color image")
    hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
