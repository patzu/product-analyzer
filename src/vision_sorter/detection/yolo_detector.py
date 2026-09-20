"""Optional YOLO inference adapter. No weights are downloaded implicitly."""
from pathlib import Path
from vision_sorter.domain.models import Detection
from vision_sorter.image_processing.reader import Image


class YoloDetector:
    def __init__(self, weights: Path, confidence: float = 0.25):
        if not weights.is_file():
            raise FileNotFoundError(f"Provide local YOLO weights: {weights}")
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Optional YOLO dependency missing; install .[ml] deliberately") from exc
        self.model = YOLO(str(weights), task="detect")
        self.confidence = confidence

    def detect(self, image: Image) -> list[Detection]:
        results = self.model.predict(source=image, conf=self.confidence, verbose=False)
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
                width, height = x2 - x1, y2 - y1
                if width <= 0 or height <= 0:
                    raise ValueError("YOLO returned a degenerate bounding box")
                label = result.names[int(box.cls.item())]
                # YOLO measures box geometry, not the contour/surface of the product.
                detections.append(Detection(float(width * height), float(2 * (width + height)),
                    x1, y1, width, height, (x1 + x2) / 2, (y1 + y2) / 2,
                    label, float(box.conf.item())))
        return detections
