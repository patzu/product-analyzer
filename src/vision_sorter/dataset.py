"""Reproducible image manifests and duplicate-aware copy-only splits."""
import hashlib
import random
import shutil
from pathlib import Path
from collections import defaultdict
import pandas as pd
from vision_sorter.config import Config
from vision_sorter.image_processing.reader import discover_images, image_metadata


def prepare_dataset(source: Path, manifest: Path) -> None:
    """Validate all images before writing a manifest; labels are added separately."""
    rows = []
    for path in discover_images(source, Config().image_extensions):
        metadata = image_metadata(path)
        if metadata.error:
            raise ValueError(metadata.error)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"path": path.resolve().as_posix(), "sha256": digest,
                     "width": metadata.width, "height": metadata.height})
    if not rows:
        raise ValueError("Dataset contains no supported images")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x", encoding="utf-8", newline="") as stream:
        pd.DataFrame(rows).to_csv(stream, index=False)


def split_dataset(manifest: Path, destination: Path, seed: int = 42,
                  train_fraction: float = 0.7, validation_fraction: float = 0.15) -> dict[str, int]:
    """Keep byte-identical images in one partition; never overwrite a split."""
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1 - train_fraction:
        raise ValueError("Invalid train/validation fractions")
    frame = pd.read_csv(manifest, dtype=str)
    if frame.empty or not {"path", "sha256"} <= set(frame.columns):
        raise ValueError("Manifest requires nonempty path and sha256 columns")
    groups = defaultdict(list)
    for row in frame.to_dict("records"):
        path = Path(row["path"])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Image missing or changed since manifest: {path}")
        groups[row["sha256"]].append(path)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    train_end = int(len(keys) * train_fraction)
    validation_end = train_end + int(len(keys) * validation_fraction)
    destination.mkdir(parents=True, exist_ok=False)
    counts = {"train": 0, "validation": 0, "test": 0}
    rows = []
    for name in counts:
        (destination / name).mkdir()
    for index, digest in enumerate(keys):
        partition = "train" if index < train_end else "validation" if index < validation_end else "test"
        for number, source in enumerate(groups[digest]):
            output = destination / partition / f"{digest}-{number}{source.suffix.lower()}"
            shutil.copy2(source, output)
            counts[partition] += 1
            rows.append({"source": str(source), "sha256": digest, "partition": partition, "output": str(output)})
    pd.DataFrame(rows).to_csv(destination / "split.csv", index=False)
    return counts
