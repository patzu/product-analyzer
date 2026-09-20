"""Terminal and IntelliJ entry point for the inspection prototype."""
import argparse
import json
import logging
from pathlib import Path
from uuid import uuid4
from vision_sorter.config import Config
from vision_sorter.logging_config import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classical computer vision inspection and mock sorting")
    parser.add_argument("--root", type=Path, help="Workspace root; defaults to VISION_SORTER_HOME or current directory")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Create workspace directories")
    scan = sub.add_parser("scan", help="Recursive image metadata CSV")
    scan.add_argument("source", type=Path)
    for command in ("preprocess", "detect", "inspect", "video", "camera"):
        action = sub.add_parser(command)
        action.add_argument("source", type=int if command == "camera" else Path)
        action.add_argument("--invert", action="store_true", help="Dark products on a bright background")
        action.add_argument("--minimum-contour-area", type=float, default=100)
        if command in ("inspect", "video", "camera"):
            action.add_argument("--rules", type=Path)
            action.add_argument("--save-intermediate", action="store_true")
        if command in ("video", "camera"):
            action.add_argument("--max-frames", type=int, default=100)
            action.add_argument("--stride", type=int, default=1)
    sub.add_parser("report", help="Export all SQLite inspections")
    serve = sub.add_parser("serve", help="Local API and dashboard")
    serve.add_argument("--rules", type=Path)
    serve.add_argument("--port", type=int, default=8000)
    sub.add_parser("demo", help="Create clearly labeled synthetic fixtures without overwriting data")
    return parser


def execute(args: argparse.Namespace) -> int:
    from vision_sorter.image_processing.reader import discover_images, image_metadata, read_image, write_image
    from vision_sorter.reporting.report_service import metadata_csv, inspection_csv
    from vision_sorter.classification.rules import RuleConfig
    from vision_sorter.inspection.inspection_service import InspectionService
    config = Config(root=args.root) if args.root else Config()
    config.initialize()
    if hasattr(args, "minimum_contour_area"):
        from dataclasses import replace
        config = replace(config, minimum_contour_area=args.minimum_contour_area)
    if args.command == "init":
        print(f"Workspace initialized: {config.root}")
    elif args.command == "scan":
        records = [image_metadata(p) for p in discover_images(args.source, config.image_extensions)]
        output = config.report_dir / f"metadata-{uuid4().hex[:12]}.csv"
        print(metadata_csv(records, output))
        return 1 if any(record.error for record in records) else 0
    elif args.command in ("preprocess", "detect"):
        from vision_sorter.image_processing.pipeline import preprocess
        from vision_sorter.detection.classical_detector import ClassicalDetector, annotate
        image = read_image(args.source)
        output = config.root / "runs/inspection" / str(uuid4())
        if args.command == "preprocess":
            preprocess(image, args.invert, output)
        else:
            detections = ClassicalDetector(config.minimum_contour_area, args.invert).detect(image)
            write_image(output / "annotated.png", annotate(image, detections))
            print(f"{len(detections)} objects")
        print(output)
    elif args.command in ("inspect", "video", "camera"):
        rules = RuleConfig.from_json(args.rules) if args.rules else RuleConfig()
        service = InspectionService(config, rules, invert=args.invert)
        if args.command == "inspect":
            results = service.inspect_folder(args.source, args.save_intermediate) if args.source.is_dir() else [service.inspect_path(args.source, args.save_intermediate)]
        else:
            from vision_sorter.inspection.capture import inspect_capture
            results = inspect_capture(service, args.source, args.max_frames, args.stride, args.save_intermediate)
        failures = 0
        for result in results:
            print(json.dumps({"id": result.id, "status": result.status, "reason": result.reason, "lane": result.sort_lane}))
            failures += bool(result.error)
        return 1 if failures else 0
    elif args.command == "report":
        from vision_sorter.storage.repositories import InspectionRepository
        print(inspection_csv(InspectionRepository(config.database_path), config.report_dir / f"inspections-{uuid4().hex[:12]}.csv"))
    elif args.command == "serve":
        import uvicorn
        from vision_sorter.api.app import create_app
        uvicorn.run(create_app(config, RuleConfig.from_json(args.rules) if args.rules else None), host="127.0.0.1", port=args.port)
    elif args.command == "demo":
        import numpy as np
        destination = config.data_dir / "raw" / f"synthetic-{uuid4().hex[:8]}"
        for name, size in (("small", 12), ("medium", 45), ("large", 190)):
            image = np.zeros((240, 280, 3), np.uint8)
            image[20:20 + size, 20:20 + size] = 255
            write_image(destination / f"{name}.png", image)
        write_image(destination / "empty.png", np.zeros((240, 280, 3), np.uint8))
        print(f"Synthetic fixtures only; no accuracy claim: {destination}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    configure_logging()
    try:
        return execute(args)
    except Exception:
        logging.getLogger(__name__).exception("Command failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
