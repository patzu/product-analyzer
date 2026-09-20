# Phase validation journal

The supplied attachment ended at Phase 5. The original workspace was empty:
there was no existing source, Git repository, or data to migrate. Nothing was
deleted. Python 3.12.9 and a local virtual environment were set up for validation.
Commands below use `.venv/Scripts/python` (or the equivalent Windows path).

## Phase 1 — Clean foundation

Files: pyproject.toml, requirements.txt, .gitignore, package __init__ files,
config.py, main.py, tests/unit/test_config.py, and requested data/output folders.
Implemented pathlib configuration, workspace initialization, CLI packaging,
pytest configuration, and editable imports for terminal and IntelliJ.
Run: `python -m vision_sorter.main --root . init`.
Validation: initialization and isolated custom-path smoke checks passed before
Phase 2. The configuration pytest also passed in the subsequent three-test run.
Limit: no image pipeline at this stage. Next: ingestion and reporting.

## Phase 2 — Ingestion and reporting

Files: domain/models.py, image_processing/reader.py, logging_config.py,
reporting/report_service.py, main.py, tests/unit/test_reader.py.
Implemented sorted recursive discovery, extension filtering, Unicode paths,
original image dimensions/channels/pixels/size, corrupt-file error records,
JSON logs, and CSV output (including a header for an empty report).
Run: `python -m vision_sorter.main scan data/raw`.
Validation: 3 tests passed and an empty-folder CLI scan generated a valid CSV.
Limit: image headers/metadata are not object detections. Next: preprocessing.

## Phase 3 — OpenCV preprocessing

Files: image_processing/preprocessing.py, thresholding.py, edges.py, color.py,
pipeline.py, main.py, tests/unit/test_preprocessing.py.
Implemented grayscale, Gaussian blur, fixed/Otsu/adaptive thresholding, Canny,
opening/closing/erosion/dilation, HSV masks, and optional six-stage outputs.
Run: `python -m vision_sorter.main preprocess path/to/image.png`.
Validation: 11 tests passed with binary mask, noise, hole, color, and invalid
parameter fixtures. Limit: processing expects uint8 and configured polarity.
Next: contour detection.

## Phase 4 — Classical detection

Files: detection/base.py, classical_detector.py, image_processing/contours.py,
main.py, tests/unit/test_detection.py.
Implemented minimum-area filtering, contour area/perimeter, bounding rectangles,
moment centroids, optional color fraction, annotation, and saved outputs.
Run: `python -m vision_sorter.main detect path/to/image.png`.
Validation: 13 tests passed, including two-object geometry and dark foreground.
Limit: touching products can merge; measurements are pixels, confidence is None.
This is classical computer vision, not a learned AI model. Next: acceptance rules.

## Phase 5 — Inspection classification

Files: classification/rules.py, config.example.json, tests/unit/test_rules.py.
Implemented all six dimension limits, optional HSV/color/confidence rules,
configuration validation, GOOD/DEFECTIVE/UNKNOWN with reasons, and aggregation.
Run the phase alone: `python -m pytest tests/unit/test_rules.py -q`.
Run connected workflow: `python -m vision_sorter.main inspect data/raw --rules config.example.json`.
Validation: 21 tests passed; boundary, invalid measurement, missing confidence,
color, and empty-scene behavior exercised. Limit: thresholds need calibration
against actual products. Next: integrate persistence and mock sorting.

## Integration — Runnable portfolio prototype

Files: inspection/inspection_service.py and capture.py; storage/database.py and
repositories.py; sorting interfaces/mock/disabled-serial modules; api/app.py;
expanded main.py and report_service.py; dataset.py; scripts; optional YOLO and
TorchScript adapters; integration/dataset tests; README and documentation.
Implemented SQLite history, per-object decisions and configuration snapshots,
unique annotated artifacts, mock lanes, paginated reads, CSV exports, bounded
video/camera capture, local API/dashboard, synthetic demo, and copy-only dataset
preparation/splitting. Optional adapters do not download weights.
Run: README quick start, or `python -m vision_sorter.main serve`.
Validation: 28 tests passed at this checkpoint, covering API, CLI subprocesses,
video fixture decoding, storage, reports, and duplicate grouping.
Limits: no real model/camera/hardware validation, no accuracy claims, no training
pipeline, no physical sorting synchronization. Installed dependencies emit two
TestClient deprecation warnings; these are documented without suppressing them.
Next: calibrate on real data, assess a held-out dataset, then evaluate whether a
learned detector/classifier is justified before connecting hardware.
