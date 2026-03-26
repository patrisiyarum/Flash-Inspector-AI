#!/usr/bin/env python3
"""
FlashInspector AI - Run Roboflow Hosted Model on Images/Video

Uses a Roboflow-hosted model for inference. By default uses the
"my-first-project-nqfzv" model (v3) from patyas-workspace.

Usage:
    python predict_roboflow.py path/to/video.mp4
    python predict_roboflow.py path/to/image.jpg
    python predict_roboflow.py path/to/video.mp4 --fps 10
    python predict_roboflow.py path/to/image.jpg --confidence 50

    # Use a different Roboflow model:
    python predict_roboflow.py photo.jpg --workspace my-ws --project my-proj --version 2
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "inference_results"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

DEFAULT_WORKSPACE = "patyas-workspace"
DEFAULT_PROJECT = "my-first-project-nqfzv"
DEFAULT_VERSION = 3


def get_api_key() -> str:
    load_dotenv(BASE_DIR / ".env")
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if api_key and api_key != "your_roboflow_api_key_here":
        return api_key
    print("\nROBOFLOW_API_KEY not found in .env")
    print("Get your free API key at: https://app.roboflow.com/settings/api")
    api_key = input("Enter your Roboflow API key: ").strip()
    if not api_key:
        logger.error("No API key provided.")
        sys.exit(1)
    return api_key


def predict_image(model, image_path: Path, confidence: int):
    logger.info(f"Running inference on: {image_path.name}")
    prediction = model.predict(str(image_path), confidence=confidence).json()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"roboflow_{image_path.stem}.json"
    with open(out_path, "w") as f:
        json.dump(prediction, f, indent=2)

    detections = prediction.get("predictions", [])
    if detections:
        logger.info(f"Found {len(detections)} detection(s):")
        for d in detections:
            logger.info(f"  {d['class']}: {d['confidence']:.1%} at ({d['x']:.0f}, {d['y']:.0f})")
    else:
        logger.info("No detections.")

    logger.info(f"Results saved to {out_path}")

    # Save annotated image
    try:
        prediction_obj = model.predict(str(image_path), confidence=confidence)
        annotated_path = RESULTS_DIR / f"roboflow_{image_path.name}"
        prediction_obj.save(str(annotated_path))
        logger.info(f"Annotated image saved to {annotated_path}")
    except Exception:
        pass

    return prediction


def predict_video(model, video_path: Path, fps: int):
    logger.info(f"Submitting video: {video_path.name} ({fps} fps)")
    logger.info("Uploading to Roboflow servers — this may take a few minutes...")

    job_id, signed_url, expire_time = model.predict_video(
        str(video_path),
        fps=fps,
        prediction_type="batch-video",
    )

    logger.info(f"Job: {job_id}")
    logger.info(f"Results URL expires: {expire_time}")
    logger.info("Polling for results...")

    results = model.poll_until_video_results(job_id)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"roboflow_{video_path.stem}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Inference complete — results saved to {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Roboflow hosted model on images/video")
    parser.add_argument("input", type=str, help="Path to image or video file")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second for video (default: 5)")
    parser.add_argument("--confidence", type=int, default=40, help="Confidence threshold 0-100 (default: 40)")
    parser.add_argument("--workspace", type=str, default=DEFAULT_WORKSPACE, help="Roboflow workspace")
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT, help="Roboflow project")
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION, help="Model version")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"File not found: {input_path}")
        sys.exit(1)

    api_key = get_api_key()
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    model = project.version(args.version).model
    logger.info(f"Model loaded: {args.workspace}/{args.project} v{args.version}")

    ext = input_path.suffix.lower()
    if ext in IMAGE_EXTS:
        predict_image(model, input_path, args.confidence)
    elif ext in VIDEO_EXTS:
        predict_video(model, input_path, args.fps)
    else:
        logger.error(f"Unsupported file type: {ext}")
        sys.exit(1)


if __name__ == "__main__":
    main()
