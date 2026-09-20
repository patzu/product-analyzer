"""Copy images into a new reproducible train/validation/test split."""
import argparse
from pathlib import Path
from vision_sorter.dataset import split_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(split_dataset(args.manifest, args.destination, args.seed))
