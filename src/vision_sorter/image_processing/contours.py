"""External contour measurements; holes are not separate objects."""
import cv2
from vision_sorter.domain.models import Detection
from vision_sorter.image_processing.reader import Image


def measure_contours(binary: Image, minimum_area: float) -> list[Detection]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < minimum_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        center_x = moments["m10"] / moments["m00"] if moments["m00"] else x + width / 2
        center_y = moments["m01"] / moments["m00"] if moments["m00"] else y + height / 2
        detections.append(Detection(area, float(cv2.arcLength(contour, True)), x, y, width, height, center_x, center_y))
    return sorted(detections, key=lambda item: (item.x, item.y))
