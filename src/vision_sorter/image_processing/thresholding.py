"""Binary thresholding; bright foreground is the default."""
import cv2
from vision_sorter.image_processing.reader import Image
from vision_sorter.image_processing.preprocessing import grayscale, kernel_size


def threshold(image: Image, value: int = 127, invert: bool = False) -> Image:
    if not 0 <= value <= 255:
        raise ValueError("Threshold must be between 0 and 255")
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.threshold(grayscale(image), value, 255, mode)[1]


def otsu_threshold(image: Image, invert: bool = False) -> Image:
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.threshold(grayscale(image), 0, 255, mode | cv2.THRESH_OTSU)[1]


def adaptive_threshold(image: Image, block_size: int = 11, constant: float = 2) -> Image:
    kernel_size(block_size)
    if block_size < 3:
        raise ValueError("Adaptive block size must be at least 3")
    return cv2.adaptiveThreshold(grayscale(image), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block_size, constant)
