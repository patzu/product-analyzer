import numpy as np
import pandas as pd
import pytest
from vision_sorter.image_processing.reader import discover_images, image_metadata, write_image
from vision_sorter.reporting.report_service import metadata_csv


def test_recursive_metadata_and_failures(tmp_path):
    image = tmp_path / "nested" / "محصول.PNG"
    write_image(image, np.zeros((12, 20, 4), dtype=np.uint8))
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    (tmp_path / "ignore.txt").write_text("text")
    paths = discover_images(tmp_path, (".png", ".jpg"))
    assert paths == sorted([image, broken])
    good = image_metadata(image)
    assert (good.width, good.height, good.channels, good.pixel_count) == (20, 12, 4, 240)
    bad = image_metadata(broken)
    assert bad.error and bad.width is None
    csv = metadata_csv([good, bad], tmp_path / "report.csv")
    assert len(pd.read_csv(csv)) == 2


def test_missing_folder_and_empty_report(tmp_path):
    with pytest.raises(NotADirectoryError):
        discover_images(tmp_path / "missing", (".png",))
    assert list(pd.read_csv(metadata_csv([], tmp_path / "empty.csv")).columns)[0] == "path"
