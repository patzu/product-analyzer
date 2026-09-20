"""Plain data objects shared by detection, inspection, and persistence."""
from dataclasses import dataclass, field
from enum import StrEnum


class Status(StrEnum):
    GOOD = "GOOD"
    DEFECTIVE = "DEFECTIVE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ImageMetadata:
    path: str
    file_size: int | None
    width: int | None = None
    height: int | None = None
    channels: int | None = None
    pixel_count: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class Detection:
    area: float
    perimeter: float
    x: int
    y: int
    width: int
    height: int
    center_x: float
    center_y: float
    label: str = "object"
    confidence: float | None = None
    color_fraction: float | None = None


@dataclass(frozen=True)
class Classification:
    status: Status
    reason: str


@dataclass
class InspectionResult:
    id: str
    source: str
    created_at: str
    status: Status
    reason: str
    detections: list[Detection] = field(default_factory=list)
    classifications: list[Classification] = field(default_factory=list)
    error: str | None = None
    annotated_path: str | None = None
    sort_lane: str = "REVIEW"
    detector: str = "classical"
    configuration: dict = field(default_factory=dict)
