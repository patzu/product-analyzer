"""Repository isolates SQL from the inspection workflow."""
import json
from dataclasses import asdict
from pathlib import Path
from vision_sorter.domain.models import InspectionResult
from vision_sorter.storage.database import connect, initialize_database


class InspectionRepository:
    def __init__(self, path: Path):
        self.path = path
        initialize_database(path)

    def save(self, result: InspectionResult) -> None:
        with connect(self.path) as connection:
            connection.execute("INSERT INTO inspections VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (result.id, result.created_at, result.source, result.status, result.reason,
                 result.sort_lane, result.error, json.dumps(asdict(result), allow_nan=False)))

    def get(self, identifier: str) -> dict | None:
        with connect(self.path) as connection:
            row = connection.execute("SELECT payload FROM inspections WHERE id = ?", (identifier,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def list(self, limit: int = 100, offset: int = 0) -> list[dict]:
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("Limit must be 1..1000 and offset nonnegative")
        with connect(self.path) as connection:
            rows = connection.execute("SELECT payload FROM inspections ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def summary(self) -> dict[str, int]:
        with connect(self.path) as connection:
            rows = connection.execute("SELECT status, COUNT(*) AS count FROM inspections GROUP BY status").fetchall()
        counts = {status: 0 for status in ("GOOD", "DEFECTIVE", "UNKNOWN")}
        counts.update({row["status"]: row["count"] for row in rows})
        return counts
