# Dataset workflow

Keep original images in data/raw and working derivatives in data/processed.
Store reviewed labels in data/labeled. The named train/validation/test folders
are available for a manually curated dataset; automated splitting writes into
a NEW destination folder to avoid overwriting data.

```powershell
.venv\Scripts\python scripts/prepare_dataset.py data/raw reports/dataset.csv
.venv\Scripts\python scripts/split_dataset.py reports/dataset.csv data/split-v1 --seed 42
```

Preparation checks every supported image and writes dimensions, absolute paths,
and SHA-256 hashes. It fails on unreadable images and refuses to overwrite an
existing manifest. Splitting verifies hashes, shuffles distinct content groups
with a seed, and copies images into train/validation/test at approximately
70/15/15 percent by unique hash group. Identical images stay together. The
`split.csv` file maps every copy to its source. Small datasets may have empty
partitions due to rounding; this is not a stratified split. Absolute manifest
paths must be regenerated if the source dataset moves.

The splitter handles images only, not YOLO label sidecars. Near-duplicates and
frames from the same video are not recognized by content hashing; keep capture
sessions, product batches, and near-identical frames together manually to avoid
leakage. Perform augmentations only after partitioning. Do not evaluate on the
synthetic software-test fixtures and describe that as industrial accuracy.

Record camera model, exposure, lighting, background, belt speed, product batch,
class balance, and labeling policy. Keep a frozen test set from independent
sessions. Report detection precision/recall, class confusion, false accept/reject
rates, UNKNOWN rate, and end-to-end latency. No such metrics are available yet.
