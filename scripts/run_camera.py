"""Convenience wrapper; install the project in editable mode first."""
import sys
from vision_sorter.main import main

if __name__ == "__main__":
    sys.argv.insert(1, "camera")
    raise SystemExit(main())
