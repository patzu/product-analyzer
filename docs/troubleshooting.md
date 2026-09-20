# Troubleshooting

- Import errors: select the project .venv Python and run `pip install -e '.[test]'`.
  The system Python found initially was 3.11; the validated environment is 3.12.9.
- No detections: check input polarity, grayscale/binary/morphology outputs, and
  minimum-contour-area. Use --invert for dark foreground. Lower discovery area
  enough that small defects are not silently filtered from measurement.
- Huge detection: background may be selected as foreground; inspect binary
  output. Blank bright frames and full-image objects need domain-specific rules.
- Unexpected dimensions: area and perimeter follow contour geometry; width and
  height include the bounding rectangle's full pixel extent.
- UNKNOWN: inspect the reason. No objects, unreadable files, missing confidence,
  or missing required color measurements intentionally need review.
- Decode failures: corrupted/empty files appear as explicit error rows; metadata
  and inspection commands return exit code 1 while preserving other results.
- Camera/video failure: confirm index, device permissions, codec availability,
  and another application's ownership. Camera disconnects raise errors; video
  EOF is accepted, but a premature EOF is reported when frame count is known.
  Unknown-length streams cannot distinguish all decoder errors from normal EOF.
- UI: start `python -m vision_sorter.main serve`, use port 8000 on loopback, and
  inspect /docs. Paths submitted through the API are relative to data/raw.
- SQLite locked: stop other writers and retry; connections wait up to 30 seconds.
  Back up the database while capture is stopped. No schema migration tool is
  included yet; do not delete production data to resolve an error.
- Optional models: supply a trusted local file before constructing an adapter.
  ML dependencies were deliberately not installed and actual inference was not
  validated. Missing files fail explicitly instead of downloading models.
- Test warnings: the installed Starlette TestClient emits deprecations about
  httpx and an anyio alias. Tests pass; these dependency warnings are retained,
  not suppressed. They do not represent a failed inspection.
