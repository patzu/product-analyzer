"""Small, composable OpenCV operations on uint8 images."""
import cv2
import numpy as np
from vision_sorter.image_processing.reader import Image


def validate(image: Image) -> None:
    if image.size == 0 or image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError("Expected a nonempty uint8 grayscale, BGR, or BGRA image")
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise ValueError("Expected three or four color channels")


def grayscale(image: Image) -> Image:
    validate(image)
    if image.ndim == 2:
        return image.copy()
    code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(image, code)


def kernel_size(size: int) -> int:
    if size < 1 or size % 2 == 0:
        raise ValueError("Kernel size must be positive and odd")
    return size


def gaussian_blur(image: Image, size: int = 5) -> Image:
    validate(image)
    size = kernel_size(size)
    return cv2.GaussianBlur(image, (size, size), 0)


def morphology(image: Image, operation: int, size: int = 3) -> Image:
    validate(image)
    kernel = np.ones((kernel_size(size), kernel_size(size)), np.uint8)
    return cv2.morphologyEx(image, operation, kernel)


def opening(image: Image, size: int = 3) -> Image:
    return morphology(image, cv2.MORPH_OPEN, size)


def closing(image: Image, size: int = 3) -> Image:
    return morphology(image, cv2.MORPH_CLOSE, size)


def erosion(image: Image, size: int = 3) -> Image:
    return morphology(image, cv2.MORPH_ERODE, size)


def dilation(image: Image, size: int = 3) -> Image:
    return morphology(image, cv2.MORPH_DILATE, size)
