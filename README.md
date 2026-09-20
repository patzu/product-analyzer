# Vision Sorter — image-folder-analyzer

A Python 3.12+ industrial inspection and sorting **prototype** using classical
computer vision. Inspect images, folders, videos, or bounded camera captures;
measure objects; explain GOOD / DEFECTIVE / UNKNOWN decisions; retain results in
SQLite; export CSV reports; and simulate sorting with a mock actuator.

No LLM is used. No trained-model accuracy is claimed. The default detector is
OpenCV thresholding and contours, not AI. Optional YOLO and PyTorch inference
adapters are separate from the validated classical pipeline.

## Quick start (PowerShell)

The current workspace has a ready `.venv` using Python 3.12.9.
For a fresh checkout, install Python 3.12+ and run:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e '.[test]'
```

Alternatively, with uv: `uv venv --python 3.12`, then `uv pip install -e '.[test]'`.
OpenCV headless supports file processing and capture without GUI dependencies.
NumPy implements image arrays; pandas writes reports; FastAPI/uvicorn serve the
local API. SQLite is in Python's standard library. ML packages are not installed
by default because they are substantially larger and require suitable weights.

```powershell
.venv\Scripts\python -m vision_sorter.main --root . init
.venv\Scripts\python -m vision_sorter.main demo
.venv\Scripts\python -m vision_sorter.main scan data/raw
.venv\Scripts\python -m vision_sorter.main inspect data/raw --rules config.example.json --save-intermediate
.venv\Scripts\python -m vision_sorter.main report
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m vision_sorter.main serve --rules config.example.json
```

Open http://127.0.0.1:8000 for the local dashboard and `/docs` for API documentation.
Stop the server with Ctrl+C. The dashboard inspects paths relative to `data/raw`.
`POST /inspections` example body: `{"path":"sample.png","save_intermediate":true}`.
`GET /summary`, `GET /inspections?limit=100&offset=0`, and
`GET /inspections/{id}` expose persisted results. The API is local-only by default;
it has no authentication or production deployment configuration.

The demo creates a new `data/raw/synthetic-<id>` folder with artificial rectangles
and an empty scene. It never overwrites input images. With the example rules,
small/large rectangles should be DEFECTIVE, medium GOOD, and the empty image
UNKNOWN. These outcomes test plumbing, not real-world accuracy.

## Commands and configuration

```powershell
.venv\Scripts\python -m vision_sorter.main preprocess data/raw/sample.png --invert
.venv\Scripts\python -m vision_sorter.main detect data/raw/sample.png --minimum-contour-area 30
.venv\Scripts\python -m vision_sorter.main inspect data/raw/sample.png --rules config.example.json
.venv\Scripts\python -m vision_sorter.main video input.avi --max-frames 100 --stride 5
.venv\Scripts\python -m vision_sorter.main camera 0 --max-frames 20
```

Use `--invert` for dark foreground on a light background. Global `--root` goes
**before** the command. `VISION_SORTER_HOME` supplies the default workspace root;
otherwise it is the current directory. Input CLI paths are relative to the
current working directory; configuration paths resolve against the workspace.
`Config` also accepts custom data, report, database paths and extension lists.

Preprocessing saves original, grayscale, blurred, binary, edges, and morphology
images. Inspection outputs have unique IDs under `runs/inspection`; the database
is `data/inspections.sqlite3`. CSV exports get unique filenames in `reports`.
The SQLite payload includes object measurements, per-object reasons, detector
name, rule configuration, timestamps, and artifact paths.

Rules are inclusive boundaries; copy `config.example.json` and adjust pixel-unit
area, width, and height limits. Optional `minimum_confidence` makes classical
results UNKNOWN because contour detection has no learned confidence. Optional
`hsv_lower`, `hsv_upper`, and `minimum_color_fraction` enforce color coverage.
OpenCV HSV hue is 0..179; saturation and value are 0..255. Color coverage is
measured on segmented foreground inside each bounding rectangle.

The contour discovery area threshold is independent of the classification
minimum area: keep discovery low enough to retain undersized defects.
No detections -> UNKNOWN. Any defective object -> overall DEFECTIVE; otherwise
any unknown object -> UNKNOWN; otherwise GOOD. Routing is per image/frame:
GOOD -> ACCEPT, DEFECTIVE -> REJECT, UNKNOWN -> REVIEW.

Read errors are logged as JSON to stderr, retained as UNKNOWN in SQLite (or an
error row for metadata scans), and produce exit code 1. Defective products are
valid inspection results and do not make the command fail. Unexpected processing,
output, or database errors stop execution with a logged traceback. Exit code 2
means invalid CLI arguments. Empty folders produce empty reports/results.

## IntelliJ IDEA

Enable Python support, select `.venv\Scripts\python.exe` as the Python SDK, and
create a Python run configuration using module name `vision_sorter.main`.
Use parameters `--root D:\path\to\project inspect D:\path\to\images`.
Use the project directory as the working directory. The editable installation
also makes imports work from another directory. For tests, use pytest with target
`tests`. All shell examples can also use the installed `vision-sorter` executable.

## Project guide

- `src/vision_sorter/domain`: dataclasses and status enum (similar to Java DTOs).
- `image_processing`: pure preprocessing operations and image IO.
- `detection`: protocol, classical detector, optional YOLO adapter.
- `classification`: configurable rules and optional TorchScript classifier.
- `inspection`: orchestration and bounded video/camera processing.
- `storage`, `reporting`, `sorting`: SQLite repository, CSV exports, mock routing.
- `api`: FastAPI app factory and a small dependency-free HTML dashboard.
- `scripts`: convenience entry points and dataset preparation/splitting tools.
- `tests/unit`, `tests/integration`: synthetic functional checks.

Read [phase validation](docs/phases.md), [architecture](docs/architecture.md),
[dataset workflow](docs/dataset.md), [labeling](docs/labeling.md),
[hardware boundaries](docs/hardware.md), and [troubleshooting](docs/troubleshooting.md).

## Scope and limitations

Uniform lighting, a contrasting background, and separated products are assumed.
Touching products can merge; shadows, holes, reflections, and full-frame foreground
can mislead contours. There is no product tracking, belt timing, physical-size
calibration, safety interlock, training loop, or validated defect model. Video
frames can count the same product repeatedly. Camera access is implemented but
requires validation on your camera. Hardware communication is deliberately disabled.

Optional adapters require deliberate installation (`pip install -e '.[ml]'`) and
trusted local weights. YOLO can be injected into `InspectionService(detector=...)`;
its area/perimeter describe bounding boxes, not classical contour measurements.
Do not reuse calibrated contour rules blindly. `PyTorchClassifier.predict(crop)`
expects a TorchScript model with the documented RGB input contract and does not
automatically map arbitrary class labels to inspection dispositions. Real model
inference is unverified here; no model or labeled dataset was supplied.

Recommended next step: collect representative product images, calibrate the
classical rules, label a held-out evaluation set, and measure false accepts,
false rejects, unknown rate, and latency before adding learned models or hardware.
