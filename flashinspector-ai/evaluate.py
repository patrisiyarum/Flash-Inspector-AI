#!/usr/bin/env python3
"""
FlashInspector AI - Model Evaluation Script

Evaluates a trained YOLOv8 model on the validation set and prints per-class metrics.
Also supports running the model on a folder of test videos and comparing detections
against a ground-truth JSON file.

Usage:
    # Evaluate on validation set
    python evaluate.py --model runs/fire_safety/weights/best.pt

    # Evaluate on merged_dataset validation split
    python evaluate.py --model best_detect.pt --data merged_dataset/data.yaml

    # Evaluate on test videos against ground truth
    python evaluate.py --model best_detect.pt --videos test_videos/ --ground-truth gt.json
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_val_set(model_path: str, data_yaml: str | None, imgsz: int):
    """Run YOLO validation on a dataset and print per-class metrics."""
    model = YOLO(model_path)

    if data_yaml is None:
        for candidate in [
            BASE_DIR / "merged_dataset" / "data.yaml",
            BASE_DIR / "merged_equipment_dataset" / "data.yaml",
            BASE_DIR / "merged_violation_dataset" / "data.yaml",
        ]:
            if candidate.exists():
                data_yaml = str(candidate)
                break

    if data_yaml is None:
        logger.error("No data.yaml found. Specify --data path/to/data.yaml")
        sys.exit(1)

    logger.info(f"Model: {model_path}")
    logger.info(f"Dataset: {data_yaml}")
    logger.info(f"Image size: {imgsz}")

    results = model.val(data=data_yaml, imgsz=imgsz, verbose=True)

    logger.info("\n" + "=" * 60)
    logger.info("  PER-CLASS METRICS")
    logger.info("=" * 60)

    names = results.names
    if hasattr(results, "box"):
        box = results.box
        for i, cls_name in names.items():
            if i < len(box.p):
                logger.info(
                    f"  {cls_name:30s}  P={box.p[i]:.3f}  R={box.r[i]:.3f}  "
                    f"mAP50={box.ap50[i]:.3f}  mAP50-95={box.ap[i]:.3f}"
                )

    logger.info("\n  OVERALL:")
    logger.info(f"    mAP50:    {results.box.map50:.4f}")
    logger.info(f"    mAP50-95: {results.box.map:.4f}")

    return results


def evaluate_videos(model_path: str, videos_dir: str, gt_path: str | None, imgsz: int):
    """Run inference on test videos and optionally compare to ground truth."""
    import cv2
    from violation_rules import check_violations, consolidate_class, get_confidence_threshold

    model = YOLO(model_path)
    videos_path = Path(videos_dir)
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    videos = sorted(p for p in videos_path.iterdir() if p.suffix.lower() in video_exts)

    if not videos:
        logger.error(f"No videos found in {videos_dir}")
        sys.exit(1)

    gt = None
    if gt_path and Path(gt_path).exists():
        with open(gt_path) as f:
            gt = json.load(f)

    logger.info(f"Evaluating {len(videos)} video(s) with model: {model_path}")

    all_results = {}
    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        class_counts: dict[str, int] = defaultdict(int)
        violation_counts: dict[str, int] = defaultdict(int)
        frame_idx = 0
        frame_skip = 10

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                results = model(frame, conf=0.15, imgsz=imgsz, verbose=False)[0]
                for box in results.boxes:
                    cls = consolidate_class(results.names[int(box.cls)])
                    conf = float(box.conf)
                    if conf >= get_confidence_threshold(cls):
                        class_counts[cls] += 1

                detections = []
                for box in results.boxes:
                    cls = consolidate_class(results.names[int(box.cls)])
                    conf = float(box.conf)
                    if conf >= get_confidence_threshold(cls):
                        detections.append({
                            "class": cls,
                            "confidence": conf,
                            "bbox": [float(x) for x in box.xyxy[0].tolist()],
                        })
                violations = check_violations(detections, height, frame_idx / fps)
                for v in violations:
                    violation_counts[v["type"]] += 1

            frame_idx += 1
        cap.release()

        video_result = {
            "frames": total_frames,
            "detections_by_class": dict(class_counts),
            "violations_by_type": dict(violation_counts),
        }
        all_results[video_path.name] = video_result

        logger.info(f"\n  {video_path.name}:")
        logger.info(f"    Detections: {dict(class_counts)}")
        logger.info(f"    Violations: {dict(violation_counts)}")

    # Compare to ground truth if provided
    if gt:
        logger.info("\n" + "=" * 60)
        logger.info("  GROUND TRUTH COMPARISON")
        logger.info("=" * 60)
        for video_name, gt_info in gt.items():
            pred = all_results.get(video_name, {})
            gt_violations = gt_info.get("violations", {})
            pred_violations = pred.get("violations_by_type", {})

            logger.info(f"\n  {video_name}:")
            all_types = set(list(gt_violations.keys()) + list(pred_violations.keys()))
            for vtype in sorted(all_types):
                gt_count = gt_violations.get(vtype, 0)
                pred_count = pred_violations.get(vtype, 0)
                status = "OK" if pred_count > 0 and gt_count > 0 else ("MISSED" if gt_count > 0 else "FALSE POS")
                logger.info(f"    {vtype}: GT={gt_count}, Pred={pred_count} [{status}]")

    # Save results
    out_path = BASE_DIR / "inference_results" / "evaluation_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fire safety model")
    parser.add_argument("--model", required=True, help="Path to YOLO weights (.pt)")
    parser.add_argument("--data", default=None, help="Path to data.yaml for val-set evaluation")
    parser.add_argument("--videos", default=None, help="Folder of test videos")
    parser.add_argument("--ground-truth", default=None, help="Ground truth JSON for video comparison")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size (default: 1280)")
    args = parser.parse_args()

    if args.videos:
        evaluate_videos(args.model, args.videos, args.ground_truth, args.imgsz)
    else:
        evaluate_val_set(args.model, args.data, args.imgsz)


if __name__ == "__main__":
    main()
