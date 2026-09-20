"""CSV exports retain failures rather than silently excluding them."""
from dataclasses import asdict, fields
from pathlib import Path
from collections.abc import Iterable
import pandas as pd
from vision_sorter.domain.models import ImageMetadata
from vision_sorter.storage.repositories import InspectionRepository


def metadata_csv(records: Iterable[ImageMetadata], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([asdict(record) for record in records], columns=[f.name for f in fields(ImageMetadata)])
    frame.to_csv(destination, index=False, encoding="utf-8-sig")
    return destination


def inspection_csv(repository: InspectionRepository, destination: Path) -> Path:
    """Export every persisted result using bounded repository pages."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = ["id", "created_at", "source", "status", "reason", "sort_lane", "error", "object_count"]
    offset = 0
    first = True
    while True:
        records = repository.list(limit=1000, offset=offset)
        rows = [{**{key: record.get(key) for key in columns[:-1]}, "object_count": len(record["detections"])} for record in records]
        pd.DataFrame(rows, columns=columns).to_csv(destination, mode="w" if first else "a", header=first,
                                                  index=False, encoding="utf-8")
        first = False
        if len(records) < 1000:
            break
        offset += len(records)
    return destination
