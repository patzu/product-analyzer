import subprocess
import sys
import cv2
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from vision_sorter.api.app import create_app
from vision_sorter.config import Config
from vision_sorter.image_processing.reader import write_image
from vision_sorter.inspection.capture import inspect_capture
from vision_sorter.inspection.inspection_service import InspectionService
from vision_sorter.reporting.report_service import inspection_csv


def test_api_validation_and_inspection(tmp_path):
    config = Config(root=tmp_path)
    client = TestClient(create_app(config))
    image = np.zeros((100, 100, 3), np.uint8)
    image[20:60, 20:60] = 255
    write_image(config.data_dir / "raw/image.png", image)
    assert client.get("/health").json()["actuator"] == "mock"
    assert client.post("/inspections", json={"path": "../outside.png"}).status_code == 400
    assert client.post("/inspections", json={"path": "missing.png"}).status_code == 404
    response = client.post("/inspections", json={"path": "image.png"})
    assert response.status_code == 201
    assert response.json()["status"] == "GOOD"
    assert client.get('/inspections/' + response.json()['id']).status_code == 200
    assert client.get("/summary").json()["GOOD"] == 1
    assert client.get("/inspections?limit=0").status_code == 422
    assert client.get("/").status_code == 200


def test_video_and_report(tmp_path):
    video = tmp_path / "fixture.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (80, 80))
    assert writer.isOpened(), "MJPG writer unavailable"
    frame = np.zeros((80, 80, 3), np.uint8)
    frame[20:50, 20:50] = 255
    for _ in range(4):
        writer.write(frame)
    writer.release()
    service = InspectionService(Config(root=tmp_path))
    results = list(inspect_capture(service, video, max_frames=2, stride=2))
    assert len(results) == 2
    assert results[1].source.endswith("#frame=2")
    report = inspection_csv(service.repository, tmp_path / "report.csv")
    assert len(pd.read_csv(report)) == 2


def test_cli_demo_inspection_and_failure(tmp_path):
    def run(*args):
        return subprocess.run([sys.executable, "-m", "vision_sorter.main", "--root", str(tmp_path), *args], capture_output=True, text=True)
    assert run("demo").returncode == 0
    assert run("inspect", str(tmp_path / "data/raw")).returncode == 0
    assert run("report").returncode == 0
    failed = run("inspect", str(tmp_path / "missing.png"))
    assert failed.returncode == 1
    assert "UNKNOWN" in failed.stdout
