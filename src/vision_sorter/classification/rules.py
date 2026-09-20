"""Explainable dimensional and optional color rules."""
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from vision_sorter.domain.models import Classification, Detection, Status
from vision_sorter.image_processing.color import validate_hsv


@dataclass(frozen=True)
class RuleConfig:
    minimum_area: float = 100
    maximum_area: float | None = None
    minimum_width: float = 1
    maximum_width: float | None = None
    minimum_height: float = 1
    maximum_height: float | None = None
    minimum_confidence: float | None = None
    hsv_lower: tuple[int, int, int] | None = None
    hsv_upper: tuple[int, int, int] | None = None
    minimum_color_fraction: float | None = None

    def __post_init__(self) -> None:
        for name in ("area", "width", "height"):
            low, high = getattr(self, f"minimum_{name}"), getattr(self, f"maximum_{name}")
            if not math.isfinite(low) or low < 0 or (high is not None and (not math.isfinite(high) or high < low)):
                raise ValueError(f"Invalid {name} limits")
        for value in (self.minimum_confidence, self.minimum_color_fraction):
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("Confidence and color fraction limits must be between 0 and 1")
        if (self.hsv_lower is None) != (self.hsv_upper is None):
            raise ValueError("Both HSV bounds are required")
        if self.hsv_lower is not None:
            validate_hsv(self.hsv_lower, self.hsv_upper)
        if self.minimum_color_fraction is not None and self.hsv_lower is None:
            raise ValueError("Color fraction requires HSV bounds")

    @classmethod
    def from_json(cls, path: Path) -> "RuleConfig":
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Rules must be a JSON object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown rule names: {sorted(unknown)}")
        for name in ("hsv_lower", "hsv_upper"):
            if values.get(name) is not None:
                values[name] = tuple(values[name])
        return cls(**values)


class RuleClassifier:
    def __init__(self, config: RuleConfig | None = None):
        self.config = config or RuleConfig()

    def classify(self, detection: Detection) -> Classification:
        def result(status: Status, reason: str) -> Classification:
            return Classification(status, reason)
        for value in (detection.area, detection.width, detection.height):
            if not math.isfinite(value) or value <= 0:
                return result(Status.UNKNOWN, "invalid object measurements")
        confidence = self.config.minimum_confidence
        if confidence is not None:
            if detection.confidence is None:
                return result(Status.UNKNOWN, "object confidence unavailable")
            if not math.isfinite(detection.confidence) or not 0 <= detection.confidence <= 1 or detection.confidence < confidence:
                return result(Status.UNKNOWN, "object confidence insufficient")
        for name in ("area", "width", "height"):
            value = getattr(detection, name)
            low, high = getattr(self.config, f"minimum_{name}"), getattr(self.config, f"maximum_{name}")
            if value < low:
                return result(Status.DEFECTIVE, f"{name} below minimum limit")
            if high is not None and value > high:
                return result(Status.DEFECTIVE, f"{name} above maximum limit")
        if self.config.minimum_color_fraction is not None:
            fraction = detection.color_fraction
            if fraction is None or not math.isfinite(fraction) or not 0 <= fraction <= 1:
                return result(Status.UNKNOWN, "color measurement unavailable or invalid")
            if fraction < self.config.minimum_color_fraction:
                return result(Status.DEFECTIVE, "color fraction below minimum limit")
        return result(Status.GOOD, "all configured rules passed")


def aggregate(classifications: list[Classification]) -> Classification:
    """A frame is rejected if any detected object fails; empty scenes need review."""
    if not classifications:
        return Classification(Status.UNKNOWN, "no objects detected")
    for status in (Status.DEFECTIVE, Status.UNKNOWN):
        matches = [item.reason for item in classifications if item.status == status]
        if matches:
            return Classification(status, "; ".join(dict.fromkeys(matches)))
    return Classification(Status.GOOD, "all configured rules passed")
