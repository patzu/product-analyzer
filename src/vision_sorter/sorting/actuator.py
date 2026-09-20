from typing import Protocol


class Actuator(Protocol):
    def route(self, inspection_id: str, lane: str) -> None:
        """Apply or simulate a routing command."""
        ...
