#!/usr/bin/env python3
"""
FlashInspector AI — Model v3 Proof of Concept Demo

Demonstrates the Roboflow-hosted Model v3 (fire safety detection)
by running inference on:
  - Local video files (if available / downloaded from Git LFS)
  - Sample images from the Roboflow project dataset
  - Any user-supplied image or video file

Produces:
  1. Annotated images saved to        poc_results/frames/
  2. Per-source JSON detection results poc_results/
  3. A self-contained HTML report      poc_results/poc_report.html

Usage:
    python poc_model_v3_demo.py                        # auto-detect: dataset images
    python poc_model_v3_demo.py --images img1.jpg img2.jpg
    python poc_model_v3_demo.py --videos vid1.mp4      # if LFS videos available
    python poc_model_v3_demo.py --sample-interval 2    # frame interval for videos
"""

import argparse
import base64
import json
import logging
import os
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent  # flashinspector-ai/
VIDEOS_DIR = BASE_DIR / "videos"
RESULTS_DIR = BASE_DIR / "poc_results"
FRAMES_DIR = RESULTS_DIR / "frames"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE = "patyas-workspace"
DEFAULT_PROJECT = "my-first-project-nqfzv"
DEFAULT_VERSION = 3

# Confidence threshold (0-100 for Roboflow API)
CONFIDENCE = 30

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


# ---------------------------------------------------------------------------
# Roboflow helpers
# ---------------------------------------------------------------------------

def get_roboflow_model(api_key: str):
    """Return a Roboflow model object for v3."""
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(DEFAULT_WORKSPACE).project(DEFAULT_PROJECT)
    model = project.version(DEFAULT_VERSION).model
    logger.info(
        f"Loaded Roboflow model: {DEFAULT_WORKSPACE}/{DEFAULT_PROJECT} v{DEFAULT_VERSION}"
    )
    return model, rf


def download_dataset_images(api_key: str, max_images: int = 20) -> list[Path]:
    """Download sample images from the Roboflow project for inference testing."""
    from roboflow import Roboflow

    dl_dir = RESULTS_DIR / "dataset_samples"
    dl_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading sample images from Roboflow project dataset...")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(DEFAULT_WORKSPACE).project(DEFAULT_PROJECT)
    version = project.version(DEFAULT_VERSION)

    # Download dataset in yolov8 format to get images
    ds_dir = dl_dir / "dataset"
    if not ds_dir.exists():
        try:
            dataset = version.download("yolov8", location=str(ds_dir))
            logger.info(f"Dataset downloaded to {ds_dir}")
        except Exception as e:
            logger.warning(f"Could not download dataset: {e}")
            return []

    # Collect images from train/valid/test splits
    image_paths = []
    for split in ["test", "valid", "train"]:
        split_dir = ds_dir / split / "images"
        if split_dir.exists():
            for img in sorted(split_dir.glob("*")):
                if img.suffix.lower() in IMAGE_EXTS:
                    image_paths.append(img)
                    if len(image_paths) >= max_images:
                        break
        if len(image_paths) >= max_images:
            break

    logger.info(f"Found {len(image_paths)} sample images from dataset")
    return image_paths


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_image_file(model, image_path: Path) -> dict:
    """Run inference on an image file via the Roboflow hosted API."""
    prediction = model.predict(str(image_path), confidence=CONFIDENCE).json()
    return prediction


def predict_frame(model, frame_bgr: np.ndarray, tmp_path: Path) -> dict:
    """Run inference on a BGR frame via the Roboflow hosted API."""
    cv2.imwrite(str(tmp_path), frame_bgr)
    prediction = model.predict(str(tmp_path), confidence=CONFIDENCE).json()
    return prediction


# ---------------------------------------------------------------------------
# Frame extraction from video
# ---------------------------------------------------------------------------

def extract_sample_frames(video_path: Path, interval_sec: float = 3.0):
    """Yield (timestamp_sec, bgr_frame) tuples at the given interval."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(fps * interval_sec))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            ts = round(idx / fps, 2)
            yield ts, frame
        idx += 1

    cap.release()


def is_valid_video(path: Path) -> bool:
    """Check if a video file is a real video (not an LFS pointer)."""
    if path.stat().st_size < 1000:
        return False
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
    cap.release()
    return ok


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

COLORS = {
    # Violations (red/orange)
    "empty_mount": (0, 0, 255),
    "extinguisher_cabinet_empty": (0, 0, 255),
    "bracket_empty": (0, 0, 255),
    "missing fire extinguisher": (0, 0, 255),
    "non_compliant_tag": (0, 80, 255),
    "fire extinguisher tagged as noncompliant": (0, 80, 255),
    "yellow_tag": (0, 165, 255),
    "red_tag": (0, 0, 220),
    # Equipment (green)
    "fire_extinguisher": (0, 200, 0),
    "fire extinguisher": (0, 200, 0),
    "fire extinguisher tagged with a white label": (0, 200, 0),
    # Infrastructure (blue)
    "exit": (255, 165, 0),
    "exit stair": (255, 165, 0),
    "pull station": (255, 100, 0),
    "fire alarm panel": (255, 100, 0),
    "FDC": (255, 100, 0),
    "sounder": (255, 100, 0),
}
DEFAULT_COLOR = (255, 165, 0)  # orange BGR


def annotate_frame(frame: np.ndarray, predictions: list[dict]) -> np.ndarray:
    """Draw bounding boxes on a frame from Roboflow prediction results."""
    out = frame.copy()
    for pred in predictions:
        cx, cy = int(pred["x"]), int(pred["y"])
        w, h = int(pred["width"]), int(pred["height"])
        x1, y1 = cx - w // 2, cy - h // 2
        x2, y2 = cx + w // 2, cy + h // 2
        cls = pred["class"]
        conf = pred["confidence"]

        color = COLORS.get(cls, DEFAULT_COLOR)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{cls} {conf:.0%}"
        (tw, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th_ - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            out, label, (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out


def frame_to_base64(frame_bgr: np.ndarray, max_width: int = 800) -> str:
    """Encode a BGR frame as a base64 JPEG for embedding in HTML."""
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame_bgr = cv2.resize(frame_bgr, (max_width, int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


# ---------------------------------------------------------------------------
# Process images
# ---------------------------------------------------------------------------

def process_image(model, image_path: Path) -> dict:
    """Run inference on a single image, return result dict."""
    logger.info(f"  Inferring: {image_path.name}")

    frame = cv2.imread(str(image_path))
    if frame is None:
        logger.error(f"  Cannot read image: {image_path}")
        return None

    h, w = frame.shape[:2]
    prediction = predict_image_file(model, image_path)
    preds = prediction.get("predictions", [])

    annotated = annotate_frame(frame, preds)

    # Save annotated image
    out_path = FRAMES_DIR / f"annotated_{image_path.name}"
    cv2.imwrite(str(out_path), annotated)

    for p in preds:
        logger.info(
            f"    {p['class']}: {p['confidence']:.0%} at ({p['x']:.0f}, {p['y']:.0f})"
        )

    if not preds:
        logger.info("    No detections")

    # Classify violations — model v3 class names
    violation_classes = {
        "empty_mount", "extinguisher_cabinet_empty", "bracket_empty",
        "missing fire extinguisher",
        "non_compliant_tag", "yellow_tag", "red_tag",
        "fire extinguisher tagged as noncompliant",
        "exit_sign_dark", "blocked_exit", "smoke_detector_missing",
    }
    has_violation = any(d["class"] in violation_classes for d in preds)

    return {
        "source": image_path.name,
        "source_type": "image",
        "resolution": f"{w}x{h}",
        "total_detections": len(preds),
        "has_violation": has_violation,
        "detections": preds,
        "annotated_b64": frame_to_base64(annotated),
    }


# ---------------------------------------------------------------------------
# Process video
# ---------------------------------------------------------------------------

def process_video(model, video_path: Path, interval_sec: float) -> dict:
    """Process a video, return results dict with detections per sampled frame."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing video: {video_path.name}")
    logger.info(f"{'='*60}")

    tmp_frame = RESULTS_DIR / "_tmp_frame.jpg"
    video_frames_dir = FRAMES_DIR / video_path.stem
    video_frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = round(total_frames / fps, 1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    logger.info(f"  Duration: {duration}s | Resolution: {width}x{height} | FPS: {fps:.1f}")

    frames_data = []
    detection_count = 0

    for ts, frame in extract_sample_frames(video_path, interval_sec):
        prediction = predict_frame(model, frame, tmp_frame)
        preds = prediction.get("predictions", [])
        detection_count += len(preds)

        annotated = annotate_frame(frame, preds)

        fname = f"frame_{ts:.1f}s.jpg"
        cv2.imwrite(str(video_frames_dir / fname), annotated)

        if preds:
            frames_data.append({
                "timestamp_sec": ts,
                "detections": preds,
                "annotated_b64": frame_to_base64(annotated),
            })
            for p in preds:
                logger.info(
                    f"  [{ts:.1f}s] {p['class']}: {p['confidence']:.0%} "
                    f"at ({p['x']:.0f}, {p['y']:.0f})"
                )

    tmp_frame.unlink(missing_ok=True)

    violation_classes = {
        "empty_mount", "extinguisher_cabinet_empty", "bracket_empty",
        "missing fire extinguisher",
        "non_compliant_tag", "yellow_tag", "red_tag",
        "fire extinguisher tagged as noncompliant",
    }
    violation_frames = [f for f in frames_data if any(
        d["class"] in violation_classes for d in f["detections"]
    )]

    logger.info(f"  Total detections: {detection_count} across {len(frames_data)} frames")

    return {
        "source": video_path.name,
        "source_type": "video",
        "duration_sec": duration,
        "resolution": f"{width}x{height}",
        "total_detections": detection_count,
        "frames_with_detections": len(frames_data),
        "violation_frames": len(violation_frames),
        "frames": frames_data,
    }


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def generate_html_report(all_results: list[dict], output_path: Path):
    """Generate a self-contained HTML report with embedded images."""

    total_detections = sum(r["total_detections"] for r in all_results)
    total_sources = len(all_results)

    # Count by class
    class_counts = {}
    for r in all_results:
        dets = r.get("detections", [])
        if not dets:
            for f in r.get("frames", []):
                dets.extend(f.get("detections", []))
        for d in dets:
            cls = d["class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1

    if class_counts:
        class_rows = "".join(
            f'<tr><td><span class="class-pill {_det_css(cls)}">{cls}</span></td>'
            f'<td class="num">{count}</td></tr>'
            for cls, count in sorted(class_counts.items(), key=lambda x: -x[1])
        )
    else:
        class_rows = '<tr><td colspan="2" class="muted">No detections</td></tr>'

    # Build result sections
    sections_html = []
    for r in all_results:
        if r["source_type"] == "image":
            sections_html.append(_image_section_html(r))
        else:
            sections_html.append(_video_section_html(r))

    num_classes = len(class_counts)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FlashInspector AI — PoC Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica,
      "Apple Color Emoji", Arial, sans-serif;
    background: #ffffff;
    color: #37352f;
    line-height: 1.5;
    font-size: 15px;
    -webkit-font-smoothing: antialiased;
  }}
  .page {{
    max-width: 720px;
    margin: 0 auto;
    padding: 48px 24px 80px;
  }}
  .title {{
    font-size: 1.875rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
    color: #37352f;
  }}
  .meta {{
    font-size: 13px;
    color: #9b9a97;
    margin-bottom: 32px;
  }}
  .summary {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 40px;
  }}
  .stat {{
    border: 1px solid rgba(55, 53, 47, 0.09);
    border-radius: 4px;
    padding: 16px 14px;
    background: #fafafa;
  }}
  .stat .number {{
    font-size: 1.75rem;
    font-weight: 600;
    color: #37352f;
    letter-spacing: -0.02em;
  }}
  .stat .label {{
    font-size: 12px;
    color: #9b9a97;
    margin-top: 4px;
  }}
  h2 {{
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #9b9a97;
    margin: 36px 0 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(55, 53, 47, 0.09);
  }}
  .class-breakdown {{
    border: 1px solid rgba(55, 53, 47, 0.09);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 8px;
  }}
  .class-breakdown table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
  }}
  .class-breakdown th {{
    text-align: left;
    padding: 10px 14px;
    background: #f7f6f3;
    color: #787774;
    font-weight: 500;
    font-size: 12px;
  }}
  .class-breakdown td {{
    padding: 10px 14px;
    border-top: 1px solid rgba(55, 53, 47, 0.06);
  }}
  .class-breakdown td.num {{
    text-align: right;
    font-variant-numeric: tabular-nums;
    color: #37352f;
  }}
  .class-breakdown .muted {{
    color: #9b9a97;
    font-style: normal;
  }}
  .class-pill {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 13px;
  }}
  .class-pill.det-violation {{ background: #fdecea; color: #6b2c2c; }}
  .class-pill.det-equipment {{ background: #e8f5e9; color: #1b5e20; }}
  .class-pill.det-other {{ background: #e3f2fd; color: #0d47a1; }}

  .content {{ margin-top: 8px; }}

  .result-section {{
    border: 1px solid rgba(55, 53, 47, 0.09);
    border-radius: 4px;
    margin-bottom: 20px;
    overflow: hidden;
    background: #fff;
  }}
  .result-header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
    padding: 14px 16px;
    background: #fafafa;
    border-bottom: 1px solid rgba(55, 53, 47, 0.06);
    flex-wrap: wrap;
  }}
  .result-header h3 {{
    font-size: 14px;
    font-weight: 500;
    color: #37352f;
    word-break: break-all;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }}
  .result-meta {{
    padding: 10px 16px;
    font-size: 12px;
    color: #9b9a97;
    border-bottom: 1px solid rgba(55, 53, 47, 0.06);
  }}
  .badge {{
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 500;
    white-space: nowrap;
    flex-shrink: 0;
  }}
  .badge-pass {{ background: #f0fdf4; color: #166534; }}
  .badge-fail {{ background: #fef2f2; color: #991b1b; }}
  .badge-info {{ background: #eff6ff; color: #1e40af; }}

  .frames-grid {{
    padding: 16px;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px;
  }}
  .frame-card {{
    border: 1px solid rgba(55, 53, 47, 0.09);
    border-radius: 4px;
    overflow: hidden;
    background: #fafafa;
  }}
  .frame-card img {{ width: 100%; display: block; vertical-align: middle; }}
  .frame-info {{
    padding: 10px 12px;
    font-size: 12px;
    color: #37352f;
    line-height: 1.45;
  }}
  .det-tag {{
    display: inline-block;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 11px;
    font-weight: 500;
    margin: 2px 2px 0 0;
  }}
  .det-tag.det-violation {{ background: #fdecea; color: #6b2c2c; }}
  .det-tag.det-equipment {{ background: #e8f5e9; color: #1b5e20; }}
  .det-tag.det-other {{ background: #e3f2fd; color: #0d47a1; }}

  .no-det {{ color: #9b9a97; padding: 12px 16px; font-size: 13px; }}

  .footer {{
    margin-top: 48px;
    padding-top: 16px;
    border-top: 1px solid rgba(55, 53, 47, 0.09);
    font-size: 12px;
    color: #9b9a97;
  }}

  @media (max-width: 600px) {{
    .summary {{ grid-template-columns: 1fr; }}
    .frames-grid {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="page">
  <h1 class="title">FlashInspector AI — Model v3 PoC</h1>
  <p class="meta">{datetime.now().strftime('%B %d, %Y · %I:%M %p')}</p>

  <div class="summary">
    <div class="stat">
      <div class="number">{total_sources}</div>
      <div class="label">Sources analyzed</div>
    </div>
    <div class="stat">
      <div class="number">{total_detections}</div>
      <div class="label">Detections</div>
    </div>
    <div class="stat">
      <div class="number">{num_classes}</div>
      <div class="label">Classes</div>
    </div>
  </div>

  <h2>Detected classes</h2>
  <div class="class-breakdown">
    <table>
      <tr><th>Class</th><th style="text-align:right">Count</th></tr>
      {class_rows}
    </table>
  </div>

  <h2>Detailed results</h2>
  <div class="content">
    {''.join(sections_html)}
  </div>

  <div class="footer">
    FlashInspector AI · Model v3 proof of concept
  </div>
</div>
</body>
</html>"""

    output_path.write_text(html)
    logger.info(f"\nHTML report saved: {output_path}")


def _image_section_html(r: dict) -> str:
    """Build HTML for a single image result."""
    has_violation = r.get("has_violation", False)
    badge_class = "badge-fail" if has_violation else ("badge-info" if r["total_detections"] > 0 else "badge-pass")
    badge_text = "Violation Detected" if has_violation else (f"{r['total_detections']} Detection(s)" if r["total_detections"] > 0 else "Clear")

    det_tags = "".join(
        f'<span class="det-tag {_det_css(d["class"])}">'
        f'{d["class"]} ({d["confidence"]:.0%})</span> '
        for d in r.get("detections", [])
    )

    img_html = ""
    if r.get("annotated_b64"):
        img_html = f'<img src="data:image/jpeg;base64,{r["annotated_b64"]}" alt="annotated"/>'

    return f"""
    <div class="result-section">
        <div class="result-header">
            <h3>{r['source']}</h3>
            <span class="badge {badge_class}">{badge_text}</span>
        </div>
        <div class="result-meta">Resolution: {r['resolution']} | Detections: {r['total_detections']}</div>
        <div class="frames-grid">
            <div class="frame-card">
                {img_html}
                <div class="frame-info">{det_tags if det_tags else '<em>No detections</em>'}</div>
            </div>
        </div>
    </div>"""


def _video_section_html(r: dict) -> str:
    """Build HTML for a single video result."""
    viol_count = r.get("violation_frames", 0)
    badge_class = "badge-fail" if viol_count > 0 else ("badge-info" if r["total_detections"] > 0 else "badge-pass")
    badge_text = f"{viol_count} Violation Frame(s)" if viol_count > 0 else (f"{r['total_detections']} Detection(s)" if r["total_detections"] > 0 else "Clear")

    frames_html = ""
    for f in r.get("frames", []):
        det_tags = "".join(
            f'<span class="det-tag {_det_css(d["class"])}">'
            f'{d["class"]} ({d["confidence"]:.0%})</span> '
            for d in f["detections"]
        )
        frames_html += f"""
        <div class="frame-card">
            <img src="data:image/jpeg;base64,{f['annotated_b64']}" alt="frame"/>
            <div class="frame-info">
                <strong>{f['timestamp_sec']:.1f}s</strong> &mdash; {det_tags}
            </div>
        </div>"""

    return f"""
    <div class="result-section">
        <div class="result-header">
            <h3>{r['source']}</h3>
            <span class="badge {badge_class}">{badge_text}</span>
        </div>
        <div class="result-meta">
            Duration: {r.get('duration_sec', '?')}s | Resolution: {r['resolution']}
            | Detections: {r['total_detections']}
        </div>
        <div class="frames-grid">
            {frames_html if frames_html else '<p class="no-det">No detections in sampled frames.</p>'}
        </div>
    </div>"""


def _det_css(cls_name: str) -> str:
    """Return CSS class for a detection tag."""
    violation_classes = {
        "empty_mount", "extinguisher_cabinet_empty", "bracket_empty",
        "missing fire extinguisher",
        "non_compliant_tag", "noncompliant_tag", "yellow_tag", "red_tag",
        "fire extinguisher tagged as noncompliant",
        "exit_sign_dark", "blocked_exit", "smoke_detector_missing",
    }
    equipment_classes = {
        "fire_extinguisher", "fire extinguisher",
        "fire extinguisher tagged with a white label",
        "fire system inspection tag",
        "emergency_exit", "exit", "exit stair",
        "smoke_detector", "notification_appliance",
        "pull_station", "pull station",
        "fire_alarm_panel", "fire alarm panel",
        "emergency_light", "alarm", "sounder", "FDC",
    }
    if cls_name in violation_classes:
        return "det-violation"
    if cls_name in equipment_classes:
        return "det-equipment"
    return "det-other"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FlashInspector AI — Model v3 PoC Demo",
    )
    parser.add_argument(
        "--videos", nargs="*",
        help="Video files to process",
    )
    parser.add_argument(
        "--images", nargs="*",
        help="Image files to process",
    )
    parser.add_argument(
        "--sample-interval", type=float, default=3.0,
        help="Seconds between sampled frames for video (default: 3.0)",
    )
    parser.add_argument(
        "--max-images", type=int, default=20,
        help="Max dataset sample images to download (default: 20)",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # Load API key
    load_dotenv(BASE_DIR / ".env")
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key or api_key == "your_roboflow_api_key_here":
        logger.error(
            "Missing ROBOFLOW_API_KEY. Copy .env.template to .env in flashinspector-ai/ "
            "and set the key, or run: export ROBOFLOW_API_KEY='your_private_key'"
        )
        sys.exit(1)

    # Load model
    model, rf = get_roboflow_model(api_key)

    all_results = []

    # Process explicit images
    if args.images:
        logger.info(f"\nProcessing {len(args.images)} image(s)...")
        for img_path in args.images:
            p = Path(img_path)
            if p.exists():
                result = process_image(model, p)
                if result:
                    all_results.append(result)

    # Process explicit videos
    if args.videos:
        logger.info(f"\nProcessing {len(args.videos)} video(s)...")
        for vid_path in args.videos:
            p = Path(vid_path)
            if p.exists() and is_valid_video(p):
                result = process_video(model, p, args.sample_interval)
                all_results.append(result)
            else:
                logger.warning(f"Skipping (invalid or LFS pointer): {p}")

    # If no explicit files given, auto-detect: try videos first, fall back to dataset
    if not args.images and not args.videos:
        # Try local videos
        valid_videos = [
            v for v in sorted(VIDEOS_DIR.glob("*.mp4"))
            if is_valid_video(v)
        ]

        if valid_videos:
            logger.info(f"\nFound {len(valid_videos)} valid local video(s)")
            for vp in valid_videos[:8]:
                result = process_video(model, vp, args.sample_interval)
                all_results.append(result)
        else:
            logger.info("\nNo valid local videos (LFS not pulled). Downloading dataset images...")
            image_paths = download_dataset_images(api_key, max_images=args.max_images)

            if not image_paths:
                logger.error("No images or videos available for inference.")
                sys.exit(1)

            logger.info(f"\nRunning inference on {len(image_paths)} dataset images...")
            for img_path in image_paths:
                result = process_image(model, img_path)
                if result:
                    all_results.append(result)

    if not all_results:
        logger.error("No sources were successfully processed.")
        sys.exit(1)

    # Save combined JSON results
    json_results = []
    for r in all_results:
        jr = {k: v for k, v in r.items() if k not in ("annotated_b64", "frames")}
        if "frames" in r:
            jr["detection_frames"] = [
                {"timestamp_sec": f["timestamp_sec"], "detections": f["detections"]}
                for f in r["frames"]
            ]
        json_results.append(jr)

    json_path = RESULTS_DIR / "poc_v3_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    logger.info(f"\nJSON results: {json_path}")

    # Generate HTML report
    report_path = RESULTS_DIR / "poc_report.html"
    generate_html_report(all_results, report_path)

    # Print summary
    total_det = sum(r["total_detections"] for r in all_results)
    with_det = sum(1 for r in all_results if r["total_detections"] > 0)
    with_viol = sum(
        1 for r in all_results
        if r.get("has_violation") or r.get("violation_frames", 0) > 0
    )

    print("\n" + "=" * 60)
    print("  FlashInspector AI — Model v3 PoC Summary")
    print("=" * 60)
    print(f"  Sources analyzed:         {len(all_results)}")
    print(f"  Total detections:         {total_det}")
    print(f"  Sources with detections:  {with_det}")
    print(f"  Sources with violations:  {with_viol}")
    print(f"\n  HTML Report:  {report_path}")
    print(f"  JSON Results: {json_path}")
    print(f"  Frames:       {FRAMES_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
