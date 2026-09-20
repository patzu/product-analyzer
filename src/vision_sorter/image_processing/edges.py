"""Canny edge extraction."""
import cv2
from vision_sorter.image_processing.reader import Image
from vision_sorter.image_processing.preprocessing import grayscale


def canny_edges(image: Image, low: float = 50, high: float = 150) -> Image:
    if not 0 <= low < high:
        raise ValueError("Canny thresholds must satisfy 0 <= low < high")
    return cv2.Canny(grayscale(image), low, high)
