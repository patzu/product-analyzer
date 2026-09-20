"""Portable configuration independent of the current working directory."""
import os
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    root: Path = field(default_factory=lambda: Path(os.environ.get("VISION_SORTER_HOME", ".")).resolve())
    data_dir: Path | None = None
    report_dir: Path | None = None
    database_path: Path | None = None
    minimum_contour_area: float = 100.0
    image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser().resolve()
        object.__setattr__(self, "root", root)
        for name, default in (("data_dir", "data"), ("report_dir", "reports"), ("database_path", "data/inspections.sqlite3")):
            path = Path(getattr(self, name) or default).expanduser()
            object.__setattr__(self, name, path if path.is_absolute() else root / path)
        if not math.isfinite(self.minimum_contour_area) or self.minimum_contour_area <= 0:
            raise ValueError("minimum_contour_area must be positive")

    def initialize(self) -> None:
        for name in ("raw", "processed", "labeled", "train", "validation", "test"):
            (self.data_dir / name).mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        for name in ("models/pretrained", "models/trained", "runs/inspection", "runs/experiments"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
