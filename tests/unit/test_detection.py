import cv2
import numpy as np
import pytest
from vision_sorter.detection.classical_detector import ClassicalDetector, annotate


def test_two_objects_geometry_and_noise_filtering():
    image = np.zeros((150, 250, 3), np.uint8)
    cv2.rectangle(image, (20, 30), (60, 70), (255, 255, 255), -1)
    cv2.rectangle(image, (140, 50), (200, 110), (255, 255, 255), -1)
    image[5, 5] = 255
    objects = ClassicalDetector(100).detect(image)
    assert len(objects) == 2
    first = objects[0]
    assert (first.x, first.y, first.width, first.height) == (20, 30, 41, 41)
    assert first.area == pytest.approx(1598, abs=5)
    assert first.perimeter > 150
    assert (first.center_x, first.center_y) == pytest.approx((40, 50))
    assert first.confidence is None
    assert not np.array_equal(annotate(image, objects), image)


def test_dark_foreground_and_color():
    image = np.full((100, 100, 3), 255, np.uint8)
    image[30:70, 30:70] = (0, 0, 200)
    detector = ClassicalDetector(invert=True, hsv_lower=(0, 150, 100), hsv_upper=(10, 255, 255))
    objects = detector.detect(image)
    assert len(objects) == 1
    assert objects[0].color_fraction > 0.95
    assert ClassicalDetector().detect(np.zeros((30, 30, 3), np.uint8)) == []
