import numpy as np
import pandas as pd
from vision_sorter.config import Config
from vision_sorter.classification.rules import RuleConfig
from vision_sorter.inspection.inspection_service import InspectionService
from vision_sorter.image_processing.reader import write_image
from vision_sorter.domain.models import Status


def test_folder_persistence_annotations_and_mock_sorting(tmp_path):
    config = Config(root=tmp_path)
    config.initialize()
    image = np.zeros((100, 100, 3), np.uint8)
    image[20:60, 20:60] = 255
    write_image(config.data_dir / "raw/good.png", image)
    (config.data_dir / "raw/broken.png").write_bytes(b"broken")
    service = InspectionService(config, RuleConfig(maximum_area=2000))
    results = service.inspect_folder(config.data_dir / "raw", True)
    assert {r.status for r in results} == {Status.GOOD, Status.UNKNOWN}
    assert service.repository.summary() == {"GOOD": 1, "DEFECTIVE": 0, "UNKNOWN": 1}
    good = next(r for r in results if r.status == Status.GOOD)
    assert service.repository.get(good.id)["detections"][0]["area"] > 1000
    assert len(list((tmp_path / "runs/inspection" / good.id / "stages").glob("*.png"))) == 6
    assert {e["lane"] for e in service.actuator.events} == {"ACCEPT", "REVIEW"}
    assert service.repository.get("not-found") is None
    assert len(service.repository.list(limit=1, offset=1)) == 1


def test_defective_and_empty_scene(tmp_path):
    service = InspectionService(Config(root=tmp_path), RuleConfig(maximum_area=200))
    image = np.zeros((80, 80, 3), np.uint8)
    assert service.inspect_image(image, "empty").status == Status.UNKNOWN
    image[20:60, 20:60] = 255
    result = service.inspect_image(image, "large")
    assert result.status == Status.DEFECTIVE
    assert result.sort_lane == "REJECT"
