#!/usr/bin/env python3
"""
FlashInspector AI - Inference / Testing Script
Run trained YOLOv8 models on images or videos for fire safety detection.

Usage:
    python fire_safety_datasets/test_model.py path/to/image.jpg
    python fire_safety_datasets/test_model.py path/to/video.mp4
    python fire_safety_datasets/test_model.py path/to/video.mp4 --model fire_safety_models/fire_extinguisher_nano/weights/best.pt
    python fire_safety_datasets/test_model.py path/to/folder/ --batch
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = BASE_DIR / "fire_safety_models"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def find_latest_model() -> Path | None:
    """Find the most recently modified best.pt in the models directory."""
    if not DEFAULT_MODEL_DIR.exists():
        return None
    candidates = sorted(DEFAULT_MODEL_DIR.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def process_image(model: YOLO, image_path: Path, conf: float, save_dir: Path) -> list[dict]:
    """Run inference on a single image. Returns list of detections."""
    results = model(str(image_path), conf=conf, verbose=False)[0]
    detections = _extract_detections(results, source=str(image_path))

    # Save annotated image
    annotated = results.plot()
    out_path = save_dir / f"result_{image_path.name}"
    cv2.imwrite(str(out_path), annotated)
    logger.info(f"Annotated image saved to {out_path}")

    _print_detections(detections)
    return detections


def process_video(
    model: YOLO,
    video_path: Path,
    conf: float,
    frame_skip: int,
    save_dir: Path,
    show: bool,
) -> list[dict]:
    """Run inference on a video, processing every Nth frame. Returns all detections."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"Video: {video_path.name} ({width}x{height}, {fps:.1f} FPS, {total_frames} frames)")
    logger.info(f"Processing every {frame_skip} frame(s)")

    # Output video writer
    out_path = save_dir / f"result_{video_path.stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(out_path), fourcc, fps / frame_skip, (width, height))

    all_detections = []
    frame_idx = 0
    processed = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            results = model(frame, conf=conf, verbose=False)[0]
            timestamp = frame_idx / fps
            detections = _extract_detections(results, source=str(video_path), timestamp=timestamp)
            all_detections.extend(detections)

            annotated = results.plot()
            out_writer.write(annotated)

            if show:
                cv2.imshow("FlashInspector AI", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Stopped by user (q pressed)")
                    break

            processed += 1
            if processed % 50 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"  Processed {processed} frames "
                    f"({frame_idx}/{total_frames}, "
                    f"{processed / elapsed:.1f} fps)"
                )

        frame_idx += 1

    cap.release()
    out_writer.release()
    if show:
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    logger.info(f"Done: {processed} frames in {elapsed:.1f}s ({processed / max(elapsed, 0.01):.1f} fps)")
    logger.info(f"Annotated video saved to {out_path}")

    _print_detections(all_detections, summary=True)
    return all_detections


def _extract_detections(results, source: str = "", timestamp: float | None = None) -> list[dict]:
    """Extract detection dicts from YOLO results."""
    detections = []
    for box in results.boxes:
        det = {
            "source": source,
            "class": results.names[int(box.cls)],
            "confidence": float(box.conf),
            "bbox": [float(x) for x in box.xyxy[0].tolist()],
        }
        if timestamp is not None:
            det["timestamp_sec"] = round(timestamp, 2)
        detections.append(det)
    return detections


def _print_detections(detections: list[dict], summary: bool = False):
    """Print detection results to console."""
    if not detections:
        logger.info("No detections found.")
        return

    if summary:
        # Aggregate by class
        counts: dict[str, list[float]] = {}
        for d in detections:
            counts.setdefault(d["class"], []).append(d["confidence"])
        logger.info("\n--- Detection Summary ---")
        for cls, confs in sorted(counts.items()):
            logger.info(f"  {cls}: {len(confs)} detections (avg conf: {sum(confs)/len(confs):.2f})")
        logger.info(f"  Total: {len(detections)} detections")
    else:
        logger.info(f"\n--- {len(detections)} Detection(s) ---")
        for d in detections:
            ts = f" @ {d['timestamp_sec']}s" if "timestamp_sec" in d else ""
            logger.info(f"  {d['class']}: {d['confidence']:.2f}{ts}")


def main():
    parser = argparse.ArgumentParser(description="FlashInspector AI - Fire Safety Inference")
    parser.add_argument("input", type=str, help="Path to image, video, or folder (with --batch)")
    parser.add_argument("--model", type=str, help="Path to YOLO weights (.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--frame-skip", type=int, default=10, help="Process every Nth frame for video (default: 10)")
    parser.add_argument("--show", action="store_true", help="Show real-time annotated frames (requires display)")
    parser.add_argument("--batch", action="store_true", help="Process all images/videos in a folder")
    parser.add_argument("--save-json", action="store_true", help="Save detections to JSON file")
    args = parser.parse_args()

    # Resolve model
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = find_latest_model()
        if model_path is None:
            logger.error(
                "No trained model found. Train a model first with train_model.py, "
                "or specify --model path/to/best.pt"
            )
            sys.exit(1)

    logger.info(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    # Output directory
    save_dir = BASE_DIR / "inference_results"
    save_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    all_detections = []

    if args.batch and input_path.is_dir():
        files = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS
        )
        logger.info(f"Batch mode: {len(files)} file(s) in {input_path}")
        for f in files:
            logger.info(f"\n=== Processing: {f.name} ===")
            dets = _process_single(model, f, args, save_dir)
            all_detections.extend(dets)
    elif input_path.is_file():
        all_detections = _process_single(model, input_path, args, save_dir)
    else:
        logger.error(f"Input not found: {input_path}")
        sys.exit(1)

    # Optionally save JSON
    if args.save_json and all_detections:
        json_path = save_dir / "detections.json"
        with open(json_path, "w") as f:
            json.dump(all_detections, f, indent=2)
        logger.info(f"Detections saved to {json_path}")


def _process_single(model: YOLO, filepath: Path, args, save_dir: Path) -> list[dict]:
    """Process a single file (image or video)."""
    ext = filepath.suffix.lower()
    if ext in IMAGE_EXTS:
        return process_image(model, filepath, args.conf, save_dir)
    elif ext in VIDEO_EXTS:
        return process_video(model, filepath, args.conf, args.frame_skip, save_dir, args.show)
    else:
        logger.warning(f"Unsupported file type: {ext}")
        return []


if __name__ == "__main__":
    main()
