"""A mock only: records commands and never opens a hardware connection."""
import logging
from collections import deque


class MockActuator:
    def __init__(self) -> None:
        self.events: deque[dict[str, str]] = deque(maxlen=1000)

    def route(self, inspection_id: str, lane: str) -> None:
        if lane not in ("ACCEPT", "REJECT", "REVIEW"):
            raise ValueError(f"Unknown sorting lane: {lane}")
        self.events.append({"inspection_id": inspection_id, "lane": lane})
        logging.getLogger(__name__).info("Mock sorting: inspection=%s lane=%s", inspection_id, lane)
