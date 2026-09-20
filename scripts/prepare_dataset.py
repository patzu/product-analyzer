"""Create a checked manifest without modifying source images."""
import argparse
from pathlib import Path
from vision_sorter.dataset import prepare_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    prepare_dataset(args.source, args.manifest)
