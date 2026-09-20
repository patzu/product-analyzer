"""Coordinates detection, rules, artifacts, storage, and mock sorting."""
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from vision_sorter.config import Config
from vision_sorter.domain.models import InspectionResult, Status
from vision_sorter.detection.base import Detector
from vision_sorter.detection.classical_detector import ClassicalDetector, annotate
from vision_sorter.classification.rules import RuleClassifier, RuleConfig, aggregate
from vision_sorter.image_processing.reader import Image, ImageReadError, read_image, write_image, discover_images
from vision_sorter.image_processing.pipeline import preprocess
from vision_sorter.storage.repositories import InspectionRepository
from vision_sorter.sorting.mock_actuator import MockActuator
from vision_sorter.sorting.decision import lane_for

logger = logging.getLogger(__name__)


class InspectionService:
    def __init__(self, config: Config, rules: RuleConfig | None = None,
                 detector: Detector | None = None, invert: bool = False):
        config.initialize()
        self.config = config
        self.rules = rules or RuleConfig()
        self.detector = detector or ClassicalDetector(config.minimum_contour_area, invert,
            self.rules.hsv_lower, self.rules.hsv_upper)
        self.classifier = RuleClassifier(self.rules)
        self.repository = InspectionRepository(config.database_path)
        self.actuator = MockActuator()
        self.invert = invert

    def _persist(self, result: InspectionResult) -> InspectionResult:
        result.sort_lane = lane_for(result.status)
        result.detector = type(self.detector).__name__
        result.configuration = {"rules": asdict(self.rules), "minimum_contour_area": self.config.minimum_contour_area,
                                "invert": self.invert}
        self.repository.save(result)
        self.actuator.route(result.id, result.sort_lane)
        return result

    def inspect_image(self, image: Image, source: str, save_intermediate: bool = False) -> InspectionResult:
        identifier = str(uuid4())
        output = self.config.root / "runs/inspection" / identifier
        detections = self.detector.detect(image)
        classifications = [self.classifier.classify(item) for item in detections]
        overall = aggregate(classifications)
        annotated = output / "annotated.png"
        write_image(annotated, annotate(image, detections, [item.status.value for item in classifications]))
        if save_intermediate:
            preprocess(image, self.invert, output / "stages")
        result = InspectionResult(identifier, source, datetime.now(timezone.utc).isoformat(),
            overall.status, overall.reason, detections, classifications, annotated_path=str(annotated))
        return self._persist(result)

    def inspect_path(self, path: Path, save_intermediate: bool = False) -> InspectionResult:
        try:
            image = read_image(path)
        except (OSError, ImageReadError) as exc:
            logger.error("Inspection input failed: %s: %s", path, exc)
            return self._persist(InspectionResult(str(uuid4()), str(path.resolve()),
                datetime.now(timezone.utc).isoformat(), Status.UNKNOWN, "image could not be read", error=str(exc)))
        return self.inspect_image(image, str(path.resolve()), save_intermediate)

    def inspect_folder(self, folder: Path, save_intermediate: bool = False) -> list[InspectionResult]:
        return [self.inspect_path(path, save_intermediate) for path in discover_images(folder, self.config.image_extensions)]
