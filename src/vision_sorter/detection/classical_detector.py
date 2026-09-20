"""Classical computer vision, not an AI or learned model."""
from dataclasses import replace
import math
import cv2
import numpy as np
from vision_sorter.domain.models import Detection
from vision_sorter.image_processing.reader import Image
from vision_sorter.image_processing.pipeline import preprocess
from vision_sorter.image_processing.contours import measure_contours
from vision_sorter.image_processing.color import hsv_mask, validate_hsv


class ClassicalDetector:
    def __init__(self, minimum_area: float = 100, invert: bool = False,
                 hsv_lower: tuple[int, int, int] | None = None,
                 hsv_upper: tuple[int, int, int] | None = None):
        if not math.isfinite(minimum_area) or minimum_area <= 0:
            raise ValueError("Minimum contour area must be finite and positive")
        if (hsv_lower is None) != (hsv_upper is None):
            raise ValueError("Both HSV bounds are required")
        if hsv_lower is not None:
            validate_hsv(hsv_lower, hsv_upper)
        self.minimum_area = minimum_area
        self.invert = invert
        self.hsv_lower = hsv_lower
        self.hsv_upper = hsv_upper

    def detect(self, image: Image) -> list[Detection]:
        binary = preprocess(image, invert=self.invert)["morphology"]
        detections = measure_contours(binary, self.minimum_area)
        if self.hsv_lower is None:
            return detections
        color = hsv_mask(image, self.hsv_lower, self.hsv_upper)
        measured = []
        for item in detections:
            region = np.s_[item.y:item.y + item.height, item.x:item.x + item.width]
            foreground = binary[region] > 0
            fraction = float(np.count_nonzero((color[region] > 0) & foreground) / np.count_nonzero(foreground))
            measured.append(replace(item, color_fraction=fraction))
        return measured


def annotate(image: Image, detections: list[Detection], labels: list[str] | None = None) -> Image:
    output = image.copy()
    for index, item in enumerate(detections):
        cv2.rectangle(output, (item.x, item.y), (item.x + item.width - 1, item.y + item.height - 1), (0, 200, 0), 2)
        label = labels[index] if labels is not None else f"{item.label} area={item.area:.0f}px2"
        cv2.putText(output, label, (item.x, max(15, item.y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
    return output
