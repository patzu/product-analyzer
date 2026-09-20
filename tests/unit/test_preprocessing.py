import cv2
import numpy as np
import pytest
from vision_sorter.image_processing.preprocessing import grayscale, gaussian_blur, opening, closing, erosion, dilation
from vision_sorter.image_processing.thresholding import threshold, otsu_threshold, adaptive_threshold
from vision_sorter.image_processing.edges import canny_edges
from vision_sorter.image_processing.color import hsv_mask
from vision_sorter.image_processing.pipeline import preprocess


def test_grayscale_blur_and_thresholds():
    image = np.zeros((31, 31, 3), np.uint8)
    image[10:21, 10:21] = 255
    gray = grayscale(image)
    assert gray.shape == (31, 31)
    assert 0 < gaussian_blur(gray)[10, 10] < 255
    assert np.array_equal(threshold(gray), gray)
    assert np.array_equal(otsu_threshold(gray), gray)
    assert set(np.unique(adaptive_threshold(gray))) <= {0, 255}
    assert canny_edges(gray).sum() > 0


def test_morphology():
    image = np.zeros((15, 15), np.uint8)
    image[4:11, 4:11] = 255
    noisy = image.copy()
    noisy[1, 1] = 255
    assert np.array_equal(opening(noisy), image)
    hole = image.copy()
    hole[7, 7] = 0
    assert np.array_equal(closing(hole), image)
    assert erosion(image).sum() < image.sum() < dilation(image).sum()


def test_hsv_and_saved_stages(tmp_path):
    red = np.full((20, 20, 3), (0, 0, 255), dtype=np.uint8)
    assert hsv_mask(red, (0, 200, 200), (10, 255, 255)).all()
    assert not hsv_mask(red, (50, 200, 200), (70, 255, 255)).any()
    preprocess(red, destination=tmp_path)
    assert len(list(tmp_path.glob("*.png"))) == 6


@pytest.mark.parametrize("operation", [lambda a: gaussian_blur(a, 2), lambda a: adaptive_threshold(a, 1),
    lambda a: threshold(a, 256), lambda a: canny_edges(a, 100, 50),
    lambda a: hsv_mask(a, (180, 0, 0), (190, 255, 255))])
def test_invalid_parameters(operation):
    with pytest.raises(ValueError):
        operation(np.zeros((10, 10, 3), np.uint8))
