"""SQLite connections are short-lived and closed deterministically."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Iterator


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS inspections (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, source TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('GOOD', 'DEFECTIVE', 'UNKNOWN')),
            reason TEXT NOT NULL, sort_lane TEXT NOT NULL, error TEXT,
            payload TEXT NOT NULL)""")
        connection.execute("CREATE INDEX IF NOT EXISTS inspection_time ON inspections(created_at)")
