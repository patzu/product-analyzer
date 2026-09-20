"""Detector interface (similar to a Java interface)."""
from typing import Protocol
from vision_sorter.domain.models import Detection
from vision_sorter.image_processing.reader import Image


class Detector(Protocol):
    def detect(self, image: Image) -> list[Detection]:
        """Return object measurements in image pixel coordinates."""
        ...
