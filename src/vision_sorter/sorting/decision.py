"""Conservative routing for a whole inspected image or frame."""
from vision_sorter.domain.models import Status


def lane_for(status: Status) -> str:
    return {Status.GOOD: "ACCEPT", Status.DEFECTIVE: "REJECT", Status.UNKNOWN: "REVIEW"}[status]
