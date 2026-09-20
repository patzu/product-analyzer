import shutil
import numpy as np
import pandas as pd
import pytest
from vision_sorter.dataset import prepare_dataset, split_dataset
from vision_sorter.image_processing.reader import write_image
from vision_sorter.detection.yolo_detector import YoloDetector
from vision_sorter.classification.pytorch_classifier import PyTorchClassifier


def test_manifest_split_duplicates_and_no_overwrite(tmp_path):
    source = tmp_path / "source"
    for index in range(10):
        write_image(source / f"{index}.png", np.full((10, 10, 3), index, np.uint8))
    shutil.copy2(source / "0.png", source / "duplicate.png")
    manifest = tmp_path / "manifest.csv"
    prepare_dataset(source, manifest)
    counts = split_dataset(manifest, tmp_path / "split")
    assert sum(counts.values()) == 11
    rows = pd.read_csv(tmp_path / "split/split.csv")
    assert rows.groupby("sha256")["partition"].nunique().max() == 1
    with pytest.raises(FileExistsError):
        split_dataset(manifest, tmp_path / "split")
    (source / "0.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        split_dataset(manifest, tmp_path / "other")


def test_missing_model_never_downloads_weights(tmp_path):
    with pytest.raises(FileNotFoundError):
        YoloDetector(tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError):
        PyTorchClassifier(tmp_path / "missing.pt", ["good"])
