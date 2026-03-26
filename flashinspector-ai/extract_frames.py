#!/usr/bin/env python3
"""
FlashInspector AI - Extract Frames for Annotation

Extracts diverse frames from inspection videos for annotation in Roboflow.
Produces two sets:
  1. detection_frames/ — frames likely containing fire safety equipment
     (for annotating violations, tags, equipment)
  2. hard_negative_frames/ — frames with red/wall/ceiling objects that are
     NOT fire safety equipment (to reduce false positives)

Usage:
    python extract_frames.py path/to/video.mp4
    python extract_frames.py path/to/video.mp4 --max-frames 500
    python extract_frames.py path/to/videos_folder/ --max-frames 300
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def frame_diversity_score(frame, prev_frame):
    """Score how different this frame is from the previous one (0-1)."""
    if prev_frame is None:
        return 1.0
    small_curr = cv2.resize(frame, (64, 64))
    small_prev = cv2.resize(prev_frame, (64, 64))
    diff = cv2.absdiff(small_curr, small_prev).mean() / 255.0
    return diff


def has_red_content(frame, threshold=0.02):
    """Check if the frame has significant red-colored regions."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 80])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_ratio = (mask1.sum() + mask2.sum()) / (255.0 * frame.shape[0] * frame.shape[1])
    return red_ratio > threshold


def extract_from_video(video_path: Path, out_dir: Path, max_frames: int, diversity_threshold: float = 0.03):
    """Extract diverse frames from a single video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    step = max(1, total_frames // (max_frames * 3))
    logger.info(f"  {video_path.name}: {total_frames} frames, sampling every {step}")

    detect_dir = out_dir / "detection_frames"
    neg_dir = out_dir / "hard_negative_frames"
    detect_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    prev_frame = None
    saved = 0
    neg_saved = 0
    frame_idx = 0
    prefix = video_path.stem

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            score = frame_diversity_score(frame, prev_frame)
            if score >= diversity_threshold:
                ts = frame_idx / fps
                name = f"{prefix}_f{frame_idx:06d}_t{ts:.1f}s.jpg"

                if has_red_content(frame) or saved < max_frames:
                    cv2.imwrite(str(detect_dir / name), frame)
                    saved += 1

                if not has_red_content(frame) and neg_saved < max_frames // 5:
                    cv2.imwrite(str(neg_dir / name), frame)
                    neg_saved += 1

                prev_frame = frame.copy()

            if saved >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return saved, neg_saved


def main():
    parser = argparse.ArgumentParser(description="Extract frames from inspection videos for Roboflow annotation")
    parser.add_argument("input", type=str, help="Path to video file or folder of videos")
    parser.add_argument("--max-frames", type=int, default=300, help="Max frames to extract per video (default: 300)")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: annotation_frames/)")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output) if args.output else BASE_DIR / "annotation_frames"

    if input_path.is_file():
        videos = [input_path]
    elif input_path.is_dir():
        videos = sorted(p for p in input_path.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    else:
        logger.error(f"Not found: {input_path}")
        sys.exit(1)

    if not videos:
        logger.error(f"No video files found in {input_path}")
        sys.exit(1)

    logger.info(f"Extracting frames from {len(videos)} video(s) -> {out_dir}/")

    total_detect = 0
    total_neg = 0
    for v in videos:
        n_det, n_neg = extract_from_video(v, out_dir, args.max_frames)
        total_detect += n_det
        total_neg += n_neg

    logger.info(f"\nDone! Extracted to {out_dir}/")
    logger.info(f"  detection_frames/: {total_detect} frames (upload to Roboflow for annotation)")
    logger.info(f"  hard_negative_frames/: {total_neg} frames (upload as hard negatives)")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Upload detection_frames/ to Roboflow project")
    logger.info(f"  2. Annotate: red_tag, yellow_tag, empty_mount, blocked_exit, exit_sign_dark, etc.")
    logger.info(f"  3. Upload hard_negative_frames/ with NO annotations (teaches model what things are NOT)")
    logger.info(f"  4. Export as YOLOv8 and retrain")


if __name__ == "__main__":
    main()
