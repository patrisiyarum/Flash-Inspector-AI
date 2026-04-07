#!/usr/bin/env python3
"""
FlashInspector AI - Empty Mount Detection via Roboflow Hosted Model

Uses the Roboflow-hosted "find-empty-mounts" model (v2) to detect
empty fire extinguisher mounts/brackets in video.

Usage:
    python predict_empty_mounts.py path/to/video.mp4
    python predict_empty_mounts.py path/to/video.mp4 --fps 10
    python predict_empty_mounts.py path/to/image.jpg
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


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


def predict_image(model, image_path: Path):
    """Run inference on a single image."""
    logger.info(f"Running inference on image: {image_path.name}")
    prediction = model.predict(str(image_path), confidence=40).json()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"empty_mounts_{image_path.stem}.json"
    with open(out_path, "w") as f:
        json.dump(prediction, f, indent=2)

    detections = prediction.get("predictions", [])
    if detections:
        logger.info(f"Found {len(detections)} empty mount(s):")
        for d in detections:
            logger.info(f"  {d['class']}: {d['confidence']:.1%} at ({d['x']:.0f}, {d['y']:.0f})")
    else:
        logger.info("No empty mounts detected.")

    logger.info(f"Results saved to {out_path}")
    return prediction


def predict_video(model, video_path: Path, fps: int):
    """Run batch video inference via Roboflow hosted API."""
    logger.info(f"Submitting video for inference: {video_path.name} ({fps} fps)")
    logger.info("This uploads to Roboflow's servers and may take several minutes...")

    job_id, signed_url, expire_time = model.predict_video(
        str(video_path),
        fps=fps,
        prediction_type="batch-video",
    )

    logger.info(f"Job submitted: {job_id}")
    logger.info(f"Results URL (expires {expire_time}): {signed_url}")
    logger.info("Polling for results...")

    results = model.poll_until_video_results(job_id)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"empty_mounts_{video_path.stem}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    frame_count = len(results.get("predictions", results) if isinstance(results, dict) else results)
    logger.info(f"Inference complete — {frame_count} frame results")
    logger.info(f"Full results saved to {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Detect empty fire extinguisher mounts via Roboflow hosted model"
    )
    parser.add_argument("input", type=str, help="Path to image or video file")
    parser.add_argument("--fps", type=int, default=5,
                        help="Frames per second to sample for video (default: 5)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"File not found: {input_path}")
        sys.exit(1)

    api_key = get_api_key()
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project("find-empty-mounts")
    model = project.version("2").model

    ext = input_path.suffix.lower()
    if ext in IMAGE_EXTS:
        predict_image(model, input_path)
    elif ext in VIDEO_EXTS:
        predict_video(model, input_path, args.fps)
    else:
        logger.error(f"Unsupported file type: {ext}")
        sys.exit(1)


if __name__ == "__main__":
    main()
