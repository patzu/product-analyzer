"""Bounded video/camera inspection; no tracking or physical sorting."""
from collections.abc import Iterator
from pathlib import Path
import cv2
from vision_sorter.domain.models import InspectionResult
from vision_sorter.inspection.inspection_service import InspectionService


def inspect_capture(service: InspectionService, source: Path | int, max_frames: int = 100,
                    stride: int = 1, save_intermediate: bool = False) -> Iterator[InspectionResult]:
    if max_frames < 1 or stride < 1:
        raise ValueError("max_frames and stride must be positive")
    camera = isinstance(source, int)
    capture = cv2.VideoCapture(source if camera else str(source))
    try:
        if not capture.isOpened():
            raise OSError(f"Cannot open capture source: {source}")
        frame_number = 0
        inspected = 0
        expected_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        while inspected < max_frames:
            ok, frame = capture.read()
            if not ok:
                if camera or frame_number == 0 or (expected_frames > 0 and frame_number < expected_frames):
                    raise OSError(f"Capture read failed at frame {frame_number}: {source}")
                break
            if frame_number % stride == 0:
                yield service.inspect_image(frame, f"{source}#frame={frame_number}", save_intermediate)
                inspected += 1
            frame_number += 1
    finally:
        capture.release()
