#!/usr/bin/env python3
"""
FlashInspector AI — Model v5 Video Walkthrough Generator

Processes local inspection videos through the Roboflow Model v5 API,
annotates detections with bounding boxes and violation banners, and
produces a polished walkthrough video with title/end cards and HUD.

Output: poc_results/flashinspector_v5_walkthrough.mp4

Usage:
    python generate_v5_walkthrough.py                    # all videos
    python generate_v5_walkthrough.py --videos vid1.mp4  # specific video
    python generate_v5_walkthrough.py --interval 2       # sample every 2s
"""

import argparse
import base64
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEOS_DIR = BASE_DIR / "videos"
RESULTS_DIR = BASE_DIR / "poc_results"

PROJECT = "my-first-project-nqfzv"
VERSION = 5
CONFIDENCE = 25
OUTPUT_WIDTH = 720

COLORS = {
    "missing fire extinguisher": (0, 0, 230),
    "fire extinguisher tagged as noncompliant": (0, 70, 230),
    "fire extinguisher": (0, 190, 0),
    "fire extinguisher tagged with a white label": (0, 190, 0),
    "exit": (230, 160, 0),
    "exit sign": (230, 160, 0),
    "exit stair": (230, 160, 0),
    "pull station": (200, 80, 0),
    "fire alarm panel": (200, 80, 0),
    "sounder": (200, 130, 0),
    "FDC": (200, 130, 0),
    "fire system inspection tag": (0, 190, 190),
    "visible notification appliance": (200, 130, 0),
}
DEFAULT_COLOR = (200, 140, 0)

VIOLATION_CLASSES = {
    "missing fire extinguisher",
    "fire extinguisher tagged as noncompliant",
}


class RoboflowRESTModel:
    def __init__(self, api_key, project, version):
        self.api_key = api_key
        self.url = f"https://detect.roboflow.com/{project}/{version}"
        print(f"Using Roboflow REST API: {self.url}")

    def predict(self, image_path, confidence=25):
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        r = requests.post(
            self.url,
            params={"api_key": self.api_key, "confidence": confidence},
            data=img_b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        r.raise_for_status()
        return r.json()


def is_valid_video(path):
    if path.stat().st_size < 1000:
        return False
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
    cap.release()
    return ok


def annotate(frame, preds):
    out = frame.copy()
    h, w = out.shape[:2]
    scale_factor = w / 640

    for pred in preds:
        cx, cy = int(pred["x"]), int(pred["y"])
        pw, ph = int(pred["width"]), int(pred["height"])
        x1, y1 = max(cx - pw // 2, 0), max(cy - ph // 2, 0)
        x2, y2 = min(cx + pw // 2, w), min(cy + ph // 2, h)
        cls = pred["class"]
        conf = pred["confidence"]
        is_violation = cls in VIOLATION_CLASSES
        color = COLORS.get(cls, DEFAULT_COLOR)

        thick = max(2, int(3 * scale_factor)) if is_violation else max(1, int(2 * scale_factor))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)

        font_scale = 0.5 * scale_factor
        label = f"{cls} {conf:.0%}"
        (tw, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(out, (x1, y1 - th_ - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        if is_violation:
            banner = "VIOLATION" if "missing" in cls else "NON-COMPLIANT"
            bfont = 0.6 * scale_factor
            (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, bfont, 2)
            by = max(y1 - th_ - int(30 * scale_factor), bh + 10)
            cv2.rectangle(out, (x1 - 2, by - bh - 6), (x1 + bw + 8, by + 4), (0, 0, 180), -1)
            cv2.putText(out, banner, (x1 + 3, by - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, bfont, (255, 255, 255), 2, cv2.LINE_AA)

    return out


def make_title_card(w, h, lines, duration_frames=8):
    frames = []
    card = np.zeros((h, w, 3), dtype=np.uint8)
    card[:] = (30, 22, 15)
    cv2.rectangle(card, (0, h // 2 - 80), (w, h // 2 - 76), (60, 69, 233), -1)

    for i, (text, scale, thickness) in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x = (w - tw) // 2
        y = h // 2 - 50 + i * (th + 20)
        cv2.putText(card, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    for _ in range(duration_frames):
        frames.append(card.copy())
    return frames


def make_hud(frame, video_name, timestamp, det_count, viol_count):
    h, w = frame.shape[:2]
    bar_h = int(40 * (w / 640))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (30, 22, 15), -1)
    frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    font_scale = 0.4 * (w / 640)
    info = f"FlashInspector AI | Model v5 | {video_name} | {timestamp:.1f}s | Det: {det_count} | Viol: {viol_count}"
    cv2.putText(frame, info, (10, h - int(12 * (w / 640))),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (180, 180, 180), 1, cv2.LINE_AA)
    return frame


def process_videos(model, video_paths, interval_sec, output_path):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Get dimensions from first video
    cap = cv2.VideoCapture(str(video_paths[0]))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    scale = OUTPUT_WIDTH / src_w
    out_w = OUTPUT_WIDTH
    out_h = int(src_h * scale)
    out_h = out_h if out_h % 2 == 0 else out_h + 1

    fps = 3
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    # Title card
    for f in make_title_card(out_w, out_h, [
        ("FlashInspector AI", 1.2, 3),
        ("Model v5 - Fire Safety Detection", 0.7, 2),
        (f"Walkthrough Demo  |  {datetime.now().strftime('%B %Y')}", 0.5, 1),
    ], duration_frames=fps * 3):
        writer.write(f)

    total_det = 0
    total_viol = 0
    total_frames = 0
    tmp_path = RESULTS_DIR / "_tmp_v5.jpg"
    results_log = []

    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  Cannot open {video_path.name}, skipping")
            continue

        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / vid_fps
        step = max(1, int(vid_fps * interval_sec))

        print(f"\n  Processing: {video_path.name} ({duration:.1f}s)")

        # Video separator card
        for f in make_title_card(out_w, out_h, [
            (video_path.name, 0.6, 2),
            (f"{duration:.0f}s  |  {int(vid_fps)} fps", 0.5, 1),
        ], duration_frames=fps * 2):
            writer.write(f)

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step != 0:
                idx += 1
                continue

            timestamp = idx / vid_fps
            frame = cv2.resize(frame, (out_w, out_h))
            cv2.imwrite(str(tmp_path), frame)

            try:
                prediction = model.predict(str(tmp_path), confidence=CONFIDENCE)
            except Exception as e:
                print(f"    [{timestamp:.1f}s] Error: {e}")
                idx += 1
                continue

            preds = prediction.get("predictions", [])
            det_count = len(preds)
            viol_count = sum(1 for p in preds if p["class"] in VIOLATION_CLASSES)
            total_det += det_count
            total_viol += viol_count
            total_frames += 1

            annotated = annotate(frame, preds)
            annotated = make_hud(annotated, video_path.stem[:20], timestamp, det_count, viol_count)

            hold = 3 if preds else 1
            for _ in range(hold):
                writer.write(annotated)

            if preds:
                cls_list = [f"{p['class']} ({p['confidence']:.0%})" for p in preds]
                print(f"    [{timestamp:.1f}s] {' | '.join(cls_list)}")

            results_log.append({
                "video": video_path.name,
                "timestamp": timestamp,
                "detections": preds,
            })

            idx += 1

        cap.release()

    tmp_path.unlink(missing_ok=True)

    # End card
    for f in make_title_card(out_w, out_h, [
        ("Walkthrough Complete", 1.0, 2),
        (f"{total_frames} frames | {total_det} detections | {total_viol} violations", 0.55, 1),
        ("FlashInspector AI  -  Model v5", 0.5, 1),
    ], duration_frames=fps * 3):
        writer.write(f)

    writer.release()

    json_path = RESULTS_DIR / "v5_walkthrough_results.json"
    with open(json_path, "w") as f:
        json.dump({"model_version": VERSION, "frames_processed": total_frames,
                    "total_detections": total_det, "total_violations": total_viol,
                    "videos": [v.name for v in video_paths],
                    "results": results_log}, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Video saved: {output_path}")
    print(f"  Frames: {total_frames} | Detections: {total_det} | Violations: {total_viol}")
    print(f"  JSON:   {json_path}")
    print(f"{'='*55}")


def main():
    parser = argparse.ArgumentParser(description="FlashInspector AI — Model v5 Walkthrough")
    parser.add_argument("--videos", nargs="*", help="Specific video files to process")
    parser.add_argument("--interval", type=float, default=3.0, help="Seconds between sampled frames (default: 3)")
    args = parser.parse_args()

    load_dotenv(BASE_DIR / ".env")
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("Set ROBOFLOW_API_KEY in .env")
        sys.exit(1)

    model = RoboflowRESTModel(api_key, PROJECT, VERSION)

    if args.videos:
        video_paths = [Path(v) for v in args.videos if Path(v).exists()]
    else:
        video_paths = sorted([
            v for v in VIDEOS_DIR.glob("*.mp4")
            if is_valid_video(v)
        ])

    if not video_paths:
        print(f"No valid videos found in {VIDEOS_DIR}")
        print("Make sure git lfs pull has been run to download the video files.")
        sys.exit(1)

    print(f"Found {len(video_paths)} video(s) to process")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / "flashinspector_v5_walkthrough.mp4"
    process_videos(model, video_paths, args.interval, output_path)


if __name__ == "__main__":
    main()
