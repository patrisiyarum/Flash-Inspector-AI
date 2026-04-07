#!/usr/bin/env python3
"""
FlashInspector AI - Production Inspection Pipeline

Runs fire safety inspection on images or videos using trained YOLOv8 models.
Combines detection + violation rules + object tracking + structured reports.

Features:
  - Per-class confidence thresholds (lower for violations, higher for equipment)
  - Violation detection rules (missing extinguisher, non-compliant tag, etc.)
  - IoU-based object tracking for video (deduplicates across frames)
  - Structured JSON inspection report output
  - Optional annotated video output

Usage:
    python inspect.py path/to/video.mp4
    python inspect.py path/to/video.mp4 --model best_detect.pt
    python inspect.py path/to/image.jpg --report
    python inspect.py path/to/folder/ --batch --report
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from tracker import SimpleTracker
from violation_rules import (
    ALL_VIOLATION_CLASSES,
    VIOLATION_TYPES,
    check_violations,
    consolidate_class,
    get_confidence_threshold,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIRS = [
    BASE_DIR / "runs" / "fire_safety" / "weights",
    BASE_DIR / "fire_safety_models",
    BASE_DIR,
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

# Violation banner colors (BGR)
SEVERITY_COLORS = {
    "critical": (0, 0, 255),
    "warning": (0, 165, 255),
}

EQUIPMENT_COLOR = (0, 255, 0)
VIOLATION_COLOR = (0, 0, 255)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def find_best_model() -> Path | None:
    for d in DEFAULT_MODEL_DIRS:
        if not d.exists():
            continue
        for name in ["best.pt", "best_detect.pt"]:
            p = d / name
            if p.exists():
                return p
        candidates = sorted(d.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]
    return None


# ---------------------------------------------------------------------------
# Detection extraction with per-class confidence
# ---------------------------------------------------------------------------

def extract_detections(results, min_conf_override: float | None = None) -> list[dict]:
    """Extract detections applying per-class confidence thresholds."""
    detections = []
    for box in results.boxes:
        raw_cls = results.names[int(box.cls)]
        cls_name = consolidate_class(raw_cls)
        conf = float(box.conf)
        threshold = min_conf_override if min_conf_override is not None else get_confidence_threshold(cls_name)
        if conf < threshold:
            continue
        detections.append({
            "class": cls_name,
            "confidence": round(conf, 3),
            "bbox": [float(x) for x in box.xyxy[0].tolist()],
        })
    return detections


# ---------------------------------------------------------------------------
# Annotation drawing
# ---------------------------------------------------------------------------

def draw_detections(frame, detections: list[dict], violations: list[dict]):
    """Draw bounding boxes and violation banners on a frame."""
    annotated = frame.copy()

    violation_bboxes = {tuple(v["bbox"]) for v in violations}

    for det in detections:
        bbox = [int(x) for x in det["bbox"]]
        is_violation_det = det["class"] in ALL_VIOLATION_CLASSES or tuple(det["bbox"]) in violation_bboxes
        color = VIOLATION_COLOR if is_violation_det else EQUIPMENT_COLOR

        cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        label = f"{det['class']} {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (bbox[0], bbox[1] - th - 6), (bbox[0] + tw + 4, bbox[1]), color, -1)
        cv2.putText(annotated, label, (bbox[0] + 2, bbox[1] - 4),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    for v in violations:
        bbox = [int(x) for x in v["bbox"]]
        sev_color = SEVERITY_COLORS.get(v["severity"], (0, 0, 255))
        text = v["label"]

        cv2.rectangle(annotated, (bbox[0] - 3, bbox[1] - 3), (bbox[2] + 3, bbox[3] + 3), sev_color, 3)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        by = max(bbox[1] - 30, th + 10)
        cv2.rectangle(annotated, (bbox[0], by - th - 8), (bbox[0] + tw + 10, by + 4), sev_color, -1)
        cv2.putText(annotated, text, (bbox[0] + 5, by - 2),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def process_image(model: YOLO, image_path: Path, save_dir: Path) -> dict:
    frame = cv2.imread(str(image_path))
    if frame is None:
        logger.error(f"Cannot read image: {image_path}")
        return {}

    h, w = frame.shape[:2]
    results = model(frame, conf=0.15, verbose=False)[0]
    detections = extract_detections(results)
    violations = check_violations(detections, h, timestamp=0.0)

    annotated = draw_detections(frame, detections, violations)
    out_path = save_dir / f"result_{image_path.name}"
    cv2.imwrite(str(out_path), annotated)

    logger.info(f"  {len(detections)} detections, {len(violations)} violations -> {out_path}")

    return {
        "source": str(image_path),
        "detections": detections,
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# Video processing with tracking
# ---------------------------------------------------------------------------

def process_video(
    model: YOLO,
    video_path: Path,
    save_dir: Path,
    frame_skip: int = 5,
    show: bool = False,
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"Video: {video_path.name} ({width}x{height}, {fps:.1f} FPS, {total_frames} frames)")
    logger.info(f"Processing every {frame_skip} frame(s)")

    out_path = save_dir / f"result_{video_path.stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(out_path), fourcc, fps / frame_skip, (width, height))

    tracker = SimpleTracker(iou_threshold=0.3, max_age=int(fps / frame_skip * 3))
    all_violations: list[dict] = []
    frame_violation_counts: dict[str, int] = {}

    frame_idx = 0
    processed = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            timestamp = round(frame_idx / fps, 2)

            results = model(frame, conf=0.15, verbose=False)[0]
            detections = extract_detections(results)
            violations = check_violations(detections, height, timestamp)

            # Track objects across frames
            matched = tracker.update(detections, timestamp)

            # Attach violations to tracked objects
            for v in violations:
                for det, track in matched:
                    if det["bbox"] == v["bbox"]:
                        track.violations.append(v)
                        break
                all_violations.append(v)
                frame_violation_counts[v["type"]] = frame_violation_counts.get(v["type"], 0) + 1

            annotated = draw_detections(frame, detections, violations)
            out_writer.write(annotated)

            if show:
                cv2.imshow("FlashInspector AI", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Stopped by user")
                    break

            processed += 1
            if processed % 100 == 0:
                elapsed = time.time() - start_time
                pct = frame_idx / total_frames * 100
                logger.info(f"  {pct:.0f}% — {processed} frames ({processed / elapsed:.1f} fps)")

        frame_idx += 1

    cap.release()
    out_writer.release()
    if show:
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    logger.info(f"Done: {processed} frames in {elapsed:.1f}s ({processed / max(elapsed, 0.01):.1f} fps)")
    logger.info(f"Annotated video: {out_path}")

    # Deduplicated summary from tracker
    tracks = tracker.get_all_tracks()
    unique_equipment = [t for t in tracks if t.class_name not in ALL_VIOLATION_CLASSES]
    unique_violations_from_tracks = [t for t in tracks if t.violations]

    logger.info(f"\nUnique objects tracked: {len(tracks)}")
    logger.info(f"  Equipment: {len(unique_equipment)}")
    logger.info(f"  Objects with violations: {len(unique_violations_from_tracks)}")

    # Build deduplicated violations (one per tracked object)
    deduped_violations = []
    for track in tracks:
        if not track.violations:
            continue
        # Take the highest-confidence violation per type for this object
        seen_types: dict[str, dict] = {}
        for v in track.violations:
            vtype = v["type"]
            if vtype not in seen_types or v["confidence"] > seen_types[vtype]["confidence"]:
                seen_types[vtype] = {**v, "track_id": track.track_id}
        deduped_violations.extend(seen_types.values())

    return {
        "source": str(video_path),
        "duration_sec": round(total_frames / fps, 1),
        "frames_processed": processed,
        "unique_objects": [t.to_dict() for t in tracks],
        "equipment_count": len(unique_equipment),
        "violations": deduped_violations,
        "violation_frame_counts": frame_violation_counts,
        "annotated_video": str(out_path),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: list[dict], output_path: Path):
    """Generate a structured JSON inspection report."""
    total_violations = []
    total_equipment = 0

    for r in results:
        total_violations.extend(r.get("violations", []))
        total_equipment += r.get("equipment_count", 0)

    # Group violations by type
    by_type: dict[str, list[dict]] = {}
    for v in total_violations:
        by_type.setdefault(v["type"], []).append(v)

    report = {
        "report_generated": datetime.now().isoformat(),
        "sources": [r.get("source", "") for r in results],
        "summary": {
            "total_equipment_found": total_equipment,
            "total_violations": len(total_violations),
            "critical_violations": sum(
                1 for v in total_violations if v.get("severity") == "critical"
            ),
            "warnings": sum(
                1 for v in total_violations if v.get("severity") == "warning"
            ),
        },
        "violations_by_type": {},
        "detailed_violations": total_violations,
    }

    for vtype, vlist in by_type.items():
        info = VIOLATION_TYPES.get(vtype, {})
        report["violations_by_type"][vtype] = {
            "label": info.get("label", vtype),
            "description": info.get("description", ""),
            "count": len(vlist),
            "severity": info.get("severity", "unknown"),
        }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"\nInspection report saved: {output_path}")
    logger.info(f"  Equipment found: {total_equipment}")
    logger.info(f"  Total violations: {len(total_violations)}")
    for vtype, vlist in by_type.items():
        label = VIOLATION_TYPES.get(vtype, {}).get("label", vtype)
        logger.info(f"    {label}: {len(vlist)}")

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FlashInspector AI - Production Fire Safety Inspection",
    )
    parser.add_argument("input", type=str, help="Path to image, video, or folder (with --batch)")
    parser.add_argument("--model", type=str, help="Path to YOLO weights (.pt)")
    parser.add_argument("--frame-skip", type=int, default=5, help="Process every Nth frame for video (default: 5)")
    parser.add_argument("--show", action="store_true", help="Show real-time annotated frames")
    parser.add_argument("--batch", action="store_true", help="Process all files in a folder")
    parser.add_argument("--report", action="store_true", help="Generate JSON inspection report (default: on)")
    parser.add_argument("--no-report", action="store_true", help="Skip report generation")
    args = parser.parse_args()

    # Resolve model
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = find_best_model()
        if model_path is None:
            logger.error(
                "No trained model found. Specify --model path/to/best.pt"
            )
            sys.exit(1)

    logger.info(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    logger.info(f"  Classes: {list(model.names.values())}")

    save_dir = BASE_DIR / "inference_results"
    save_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    all_results: list[dict] = []

    if args.batch and input_path.is_dir():
        files = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS
        )
        logger.info(f"Batch mode: {len(files)} file(s)")
        for f in files:
            logger.info(f"\n=== {f.name} ===")
            r = _process_single(model, f, args, save_dir)
            if r:
                all_results.append(r)
    elif input_path.is_file():
        r = _process_single(model, input_path, args, save_dir)
        if r:
            all_results.append(r)
    else:
        logger.error(f"Input not found: {input_path}")
        sys.exit(1)

    # Generate report
    if not args.no_report and all_results:
        report_path = save_dir / "inspection_report.json"
        generate_report(all_results, report_path)


def _process_single(model: YOLO, filepath: Path, args, save_dir: Path) -> dict | None:
    ext = filepath.suffix.lower()
    if ext in IMAGE_EXTS:
        return process_image(model, filepath, save_dir)
    elif ext in VIDEO_EXTS:
        return process_video(model, filepath, save_dir, args.frame_skip, args.show)
    else:
        logger.warning(f"Unsupported file type: {ext}")
        return None


if __name__ == "__main__":
    main()
