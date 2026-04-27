#!/usr/bin/env python3
"""
FlashInspector AI — Model v5 Video Walkthrough Generator

Runs Roboflow Model v5 on dataset frames extracted from real inspection
videos, annotates detections, and stitches them into a polished
walkthrough video with title cards and detection overlays.

Output: poc_results/flashinspector_v5_walkthrough.mp4
"""

import base64
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "poc_results"
DATASET_DIR = RESULTS_DIR / "dataset_samples" / "dataset"

WORKSPACE = "patyas-workspace"
PROJECT = "my-first-project-nqfzv"
VERSION = 5
CONFIDENCE = 25

COLORS = {
    "missing fire extinguisher": (0, 0, 230),
    "fire extinguisher tagged as noncompliant": (0, 70, 230),
    "fire extinguisher": (0, 190, 0),
    "fire extinguisher tagged with a white label": (0, 190, 0),
    "exit": (230, 160, 0),
    "exit stair": (230, 160, 0),
    "pull station": (200, 80, 0),
    "fire alarm panel": (200, 80, 0),
    "sounder": (200, 130, 0),
    "FDC": (200, 130, 0),
    "fire system inspection tag": (0, 190, 190),
}
DEFAULT_COLOR = (200, 140, 0)

VIOLATION_CLASSES = {
    "missing fire extinguisher",
    "fire extinguisher tagged as noncompliant",
}


def get_model(api_key):
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    model = project.version(VERSION).model
    print(f"Loaded model: {WORKSPACE}/{PROJECT} v{VERSION}")
    return model


def find_labeled_images():
    """Find images that have ground-truth labels (known objects present)."""
    labeled = []
    for split in ["valid", "train", "test"]:
        label_dir = DATASET_DIR / split / "labels"
        img_dir = DATASET_DIR / split / "images"
        if not label_dir.exists():
            continue
        for lbl in sorted(label_dir.glob("*.txt")):
            if lbl.stat().st_size > 1:
                img = img_dir / (lbl.stem + ".jpg")
                if img.exists():
                    labeled.append(img)
    return labeled


def annotate(frame, preds):
    out = frame.copy()
    h, w = out.shape[:2]

    for pred in preds:
        cx, cy = int(pred["x"]), int(pred["y"])
        pw, ph = int(pred["width"]), int(pred["height"])
        x1, y1 = max(cx - pw // 2, 0), max(cy - ph // 2, 0)
        x2, y2 = min(cx + pw // 2, w), min(cy + ph // 2, h)
        cls = pred["class"]
        conf = pred["confidence"]
        is_violation = cls in VIOLATION_CLASSES
        color = COLORS.get(cls, DEFAULT_COLOR)

        thickness = 3 if is_violation else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        label = f"{cls} {conf:.0%}"
        font_scale = 0.5
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        if is_violation:
            banner = "VIOLATION" if "missing" in cls else "NON-COMPLIANT"
            (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            bx = x1
            by = max(y1 - th - 30, bh + 10)
            cv2.rectangle(out, (bx - 2, by - bh - 6), (bx + bw + 8, by + 4), (0, 0, 180), -1)
            cv2.putText(out, banner, (bx + 3, by - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return out


def make_title_card(w, h, lines, duration_frames=90):
    """Generate title card frames."""
    frames = []
    card = np.zeros((h, w, 3), dtype=np.uint8)
    card[:] = (30, 22, 15)

    # Red accent line
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


def make_hud_overlay(frame, frame_num, total, det_count, viol_count):
    """Add a HUD bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_h = 36
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (30, 22, 15), -1)
    frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    info = f"FlashInspector AI  |  Model v5  |  Frame {frame_num}/{total}  |  Detections: {det_count}  |  Violations: {viol_count}"
    cv2.putText(frame, info, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    return frame


def main():
    load_dotenv(BASE_DIR / ".env")
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("Set ROBOFLOW_API_KEY in .env")
        sys.exit(1)

    model = get_model(api_key)

    # Find labeled images
    images = find_labeled_images()
    if not images:
        print("No labeled dataset images found. Run poc_model_v3_demo.py first to download the dataset.")
        sys.exit(1)

    # Sample a diverse set: pick every Nth image to cover different scenes
    step = max(1, len(images) // 50)
    selected = images[::step][:50]
    print(f"Selected {len(selected)} frames from {len(images)} labeled images")

    # Determine output video size from first image
    sample = cv2.imread(str(selected[0]))
    out_h, out_w = sample.shape[:2]
    # Standardize to 640-wide
    if out_w != 640:
        scale = 640 / out_w
        out_w = 640
        out_h = int(out_h * scale)
    # Ensure even dimensions
    out_h = out_h if out_h % 2 == 0 else out_h + 1

    output_path = RESULTS_DIR / "flashinspector_v5_walkthrough.mp4"
    fps = 2  # slow enough to read annotations
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    # Title card
    title_lines = [
        ("FlashInspector AI", 1.2, 3),
        ("Model v5 - Fire Safety Detection", 0.7, 2),
        (f"Walkthrough Demo  |  {datetime.now().strftime('%B %Y')}", 0.5, 1),
    ]
    for f in make_title_card(out_w, out_h, title_lines, duration_frames=fps * 4):
        writer.write(f)

    # Process frames
    total_det = 0
    total_viol = 0
    tmp_path = RESULTS_DIR / "_tmp_v5.jpg"
    results_log = []

    for i, img_path in enumerate(selected):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        frame = cv2.resize(frame, (out_w, out_h))

        # Run inference
        cv2.imwrite(str(tmp_path), frame)
        try:
            prediction = model.predict(str(tmp_path), confidence=CONFIDENCE).json()
        except Exception as e:
            print(f"  Error on frame {i+1}: {e}")
            continue

        preds = prediction.get("predictions", [])
        det_count = len(preds)
        viol_count = sum(1 for p in preds if p["class"] in VIOLATION_CLASSES)
        total_det += det_count
        total_viol += viol_count

        annotated = annotate(frame, preds)
        annotated = make_hud_overlay(annotated, i + 1, len(selected), det_count, viol_count)

        # Hold each frame for a beat (write multiple times for readability)
        hold = 3 if preds else 1  # hold detection frames longer
        for _ in range(hold):
            writer.write(annotated)

        # Log
        cls_list = [f"{p['class']} ({p['confidence']:.0%})" for p in preds]
        status = " | ".join(cls_list) if cls_list else "clear"
        print(f"  [{i+1}/{len(selected)}] {status}")

        results_log.append({
            "frame": i + 1,
            "source": img_path.name,
            "detections": preds,
        })

    tmp_path.unlink(missing_ok=True)

    # End card
    end_lines = [
        ("Walkthrough Complete", 1.0, 2),
        (f"{len(selected)} frames  |  {total_det} detections  |  {total_viol} violations", 0.55, 1),
        ("FlashInspector AI  -  Model v5", 0.5, 1),
    ]
    for f in make_title_card(out_w, out_h, end_lines, duration_frames=fps * 4):
        writer.write(f)

    writer.release()

    # Save results JSON
    json_path = RESULTS_DIR / "v5_walkthrough_results.json"
    with open(json_path, "w") as f:
        json.dump({"model_version": 5, "frames_processed": len(selected),
                    "total_detections": total_det, "total_violations": total_viol,
                    "results": results_log}, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Video saved: {output_path}")
    print(f"  Frames: {len(selected)}  |  Detections: {total_det}  |  Violations: {total_viol}")
    print(f"  JSON:   {json_path}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()