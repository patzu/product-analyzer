"""Unicode-safe image IO and deterministic recursive discovery."""
import logging
import os
from pathlib import Path
import cv2
import numpy as np
from numpy.typing import NDArray
from vision_sorter.domain.models import ImageMetadata

logger = logging.getLogger(__name__)
Image = NDArray[np.uint8]


class ImageReadError(ValueError):
    """A file exists but cannot be decoded into an image."""


def discover_images(folder: Path, extensions: tuple[str, ...]) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    allowed = {extension.lower() for extension in extensions}
    def fail(error: OSError) -> None:
        raise error
    paths = []
    for directory, _, names in os.walk(folder, followlinks=False, onerror=fail):
        paths.extend(Path(directory) / name for name in names if Path(name).suffix.lower() in allowed)
    return sorted(paths)


def read_image(path: Path, *, unchanged: bool = False) -> Image:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise ImageReadError(f"Empty image: {path}")
    try:
        image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ImageReadError(f"Cannot decode image: {path}: {exc}") from exc
    if image is None:
        raise ImageReadError(f"Cannot decode image: {path}")
    return image


def write_image(path: Path, image: Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise OSError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def image_metadata(path: Path) -> ImageMetadata:
    size = None
    try:
        size = path.stat().st_size
        image = read_image(path, unchanged=True)
        height, width = image.shape[:2]
        channels = 1 if image.ndim == 2 else image.shape[2]
        return ImageMetadata(str(path.resolve()), size, width, height, channels, width * height)
    except (OSError, ImageReadError) as exc:
        logger.error("Image ingestion failed: %s: %s", path, exc)
        return ImageMetadata(str(path.resolve()), size, error=str(exc))
