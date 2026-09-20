# Architecture

The application uses dependency direction rather than a large framework:

```text
CLI / FastAPI -> InspectionService -> Detector protocol -> OpenCV
                                  -> RuleClassifier
                                  -> InspectionRepository -> SQLite
                                  -> MockActuator
CSV report service --------------> InspectionRepository
```

A Python dataclass is a lightweight data carrier, similar to a Java record.
A Protocol describes an interface structurally: any detector implementing
`detect(image) -> list[Detection]` can be injected. The service owns workflow;
the repository owns SQL; processing functions do not know about persistence.

Images use OpenCV BGR uint8 arrays; metadata ingestion reads original channel
counts and can describe 16-bit inputs, while inspection decodes into 8-bit BGR.
Contour area is geometric pixel-square area, not simply width times height.
Centers use contour moments. External contours ignore holes. Color coverage
uses segmented foreground pixels within each object's bounding rectangle;
nearby overlapping bounding boxes can contaminate color measurements.

Inspection artifacts are saved before the database insert. A write failure
stops processing, so a failed database insert can leave an orphan artifact folder.
A transaction persists one complete inspection payload. Mock routing is logged
only after the insert. There is no claim of transactional hardware delivery.
Each repository operation opens and closes its own connection for API thread
safety. Reports page through the database; concurrent inserts during export can
change pagination, so export while capture is stopped for a consistent snapshot.

A frame-level status represents all detected objects. It is not a physical
per-object sorting schedule. The mock actuator keeps its latest 1000 events in
memory, and intended lanes are retained in SQLite. A restart clears mock history
but not inspection history. No real actuator can be enabled accidentally.

The API resolves input paths and confines them to data/raw, validates extensions,
and exposes pagination. It serves on loopback by default. This is a trusted local
prototype, without authentication, upload processing, or deployment hardening.
The dashboard uses textContent for user-originated paths and reasons.

Optional learned adapters are lazy imports and cannot trigger an automatic
weights download. Supply trusted weights and evaluate them separately. No LLM,
PostgreSQL, serial driver, or cloud service is used.
