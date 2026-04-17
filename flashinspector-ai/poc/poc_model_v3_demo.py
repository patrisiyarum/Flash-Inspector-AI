#!/usr/bin/env python3
"""
FlashInspector AI — Model v6 Proof of Concept Demo

Demonstrates the Roboflow-hosted Model v6 (fire safety detection)
by running inference on:
  - Local video files (if available / downloaded from Git LFS)
  - Sample images from the Roboflow project dataset
  - Any user-supplied image or video file

Produces:
  1. Annotated images saved to        poc_results/frames/
  2. Per-source JSON detection results poc_results/poc_v{N}_results.json (N = dataset version)
  3. A self-contained HTML report      poc_results/poc_report.html
  4. Optional per-video HTML reports   poc_results/reports/ (see --per-video-reports-dir)

Usage:
    python poc_model_v3_demo.py                        # auto-detect: dataset images
    python poc_model_v3_demo.py --images img1.jpg img2.jpg
    python poc_model_v3_demo.py --videos vid1.mp4      # if LFS videos available
    python poc_model_v3_demo.py --sample-interval 2    # frame interval for videos
    python poc_model_v3_demo.py --videos a.mp4 b.mp4 --per-video-reports-dir reports
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import logging
import os
import re
import sys
import time
import urllib.request
import uuid
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
DEFAULT_REPORTS_SUBDIR = "reports"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE = "patyas-workspace"
DEFAULT_PROJECT = "my-first-project-nqfzv"
DEFAULT_VERSION = 6

# Confidence threshold (0-100 for Roboflow API)
CONFIDENCE = 30

# Transient Roboflow / network failures (e.g. HTTP 500)
INFERENCE_MAX_RETRIES = 6

# Hosted detect API payload limit — scale down tall/wide frames before JPEG upload
MAX_INFERENCE_EDGE_PX = 1280

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


# ---------------------------------------------------------------------------
# Roboflow helpers
# ---------------------------------------------------------------------------

def get_roboflow_model(api_key: str):
    """Return a Roboflow model object for the configured dataset version."""
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

def resize_frame_for_inference(frame_bgr: np.ndarray) -> np.ndarray:
    """Shrink so longest side is at most MAX_INFERENCE_EDGE_PX (avoids API 413)."""
    h, w = frame_bgr.shape[:2]
    m = max(h, w)
    if m <= MAX_INFERENCE_EDGE_PX:
        return frame_bgr
    scale = MAX_INFERENCE_EDGE_PX / m
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def _scale_predictions_to_original(
    preds: list[dict],
    orig_w: int,
    orig_h: int,
    scaled_w: int,
    scaled_h: int,
) -> list[dict]:
    """Map Roboflow box coords from resized inference image back to full frame."""
    if scaled_w <= 0 or scaled_h <= 0:
        return preds
    sx = orig_w / scaled_w
    sy = orig_h / scaled_h
    out: list[dict] = []
    for p in preds:
        q = dict(p)
        q["x"] = float(p["x"]) * sx
        q["y"] = float(p["y"]) * sy
        q["width"] = float(p.get("width", 0)) * sx
        q["height"] = float(p.get("height", 0)) * sy
        out.append(q)
    return out


def predict_file_with_retry(model, file_path: Path | str) -> dict:
    """Run file-based inference via Roboflow; retry on transient HTTP / network errors."""
    from requests.exceptions import HTTPError, RequestException

    path_str = str(file_path)
    for attempt in range(INFERENCE_MAX_RETRIES):
        try:
            return model.predict(path_str, confidence=CONFIDENCE).json()
        except HTTPError as e:
            code = getattr(getattr(e, "response", None), "status_code", None) or 0
            retriable = code in (408, 413, 429, 500, 502, 503, 504)
            if retriable and attempt + 1 < INFERENCE_MAX_RETRIES:
                delay = min(90.0, 2.0**attempt)
                logger.warning(
                    "Roboflow HTTP %s; retry inference %s/%s in %.0fs...",
                    code,
                    attempt + 1,
                    INFERENCE_MAX_RETRIES,
                    delay,
                )
                time.sleep(delay)
                continue
            raise
        except RequestException as e:
            if attempt + 1 < INFERENCE_MAX_RETRIES:
                delay = min(90.0, 2.0**attempt)
                logger.warning(
                    "Inference request failed; retry %s/%s in %.0fs: %s",
                    attempt + 1,
                    INFERENCE_MAX_RETRIES,
                    delay,
                    e,
                )
                time.sleep(delay)
                continue
            raise


def predict_image_file(model, image_path: Path) -> dict:
    """Run inference on an image file via the Roboflow hosted API."""
    return predict_file_with_retry(model, image_path)


def predict_frame(model, frame_bgr: np.ndarray, tmp_path: Path) -> dict:
    """Run inference on a BGR frame via the Roboflow hosted API."""
    del tmp_path  # legacy arg; use a unique temp file per frame to avoid truncated JPEG reuse
    from requests.exceptions import HTTPError

    scaled = resize_frame_for_inference(frame_bgr)
    last_disk_err: OSError | None = None
    for wattempt in range(3):
        unique = RESULTS_DIR / f"_tmp_frame_{uuid.uuid4().hex}.jpg"
        try:
            if not cv2.imwrite(str(unique), scaled, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                raise OSError("cv2.imwrite returned False")
            raw = predict_file_with_retry(model, unique)
            preds = raw.get("predictions", [])
            oh, ow = frame_bgr.shape[:2]
            sh, sw = scaled.shape[:2]
            raw["predictions"] = _scale_predictions_to_original(preds, ow, oh, sw, sh)
            return raw
        except HTTPError:
            raise
        except OSError as e:
            last_disk_err = e
            if wattempt + 1 < 3:
                logger.warning(
                    "Temp JPEG issue (%s); rewrite frame and retry %s/3",
                    e,
                    wattempt + 1,
                )
                time.sleep(0.2)
        finally:
            unique.unlink(missing_ok=True)
    raise last_disk_err if last_disk_err else OSError("predict_frame failed")


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

    # Classify violations — hosted model class names
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

def class_counts_for_result(r: dict) -> dict[str, int]:
    """Aggregate detection counts by class for one image or video result."""
    counts: dict[str, int] = {}
    dets = r.get("detections", [])
    if dets:
        for d in dets:
            cls = d["class"]
            counts[cls] = counts.get(cls, 0) + 1
    else:
        for f in r.get("frames", []):
            for d in f.get("detections", []):
                cls = d["class"]
                counts[cls] = counts.get(cls, 0) + 1
    return counts


def dominant_class_and_count(counts: dict[str, int]) -> tuple[str, int]:
    """Class with highest count; ties broken by lexicographic class name."""
    if not counts:
        return "no_detections", 0
    top_cls, top_n = max(counts.items(), key=lambda x: (x[1], x[0]))
    return top_cls, top_n


def sanitize_report_slug(text: str, max_len: int = 80) -> str:
    """Filesystem-safe slug for report filenames."""
    s = text.strip().lower().replace(" ", "_")
    out = []
    for ch in s:
        if ch.isalnum() or ch in "-_":
            out.append(ch)
        elif ch in ".:":
            out.append("_")
    slug = "".join(out).strip("_") or "class"
    return slug[:max_len]


def per_video_report_path(
    r: dict,
    reports_dir: Path,
) -> Path:
    """
    Path for a single-source HTML report: <dominant_class>_<count>.html,
    or with __<video_stem> if that name already exists (collision).
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    counts = class_counts_for_result(r)
    cls, n = dominant_class_and_count(counts)
    slug = sanitize_report_slug(cls)
    base = f"{slug}_{n}.html"
    path = reports_dir / base
    if not path.exists():
        return path
    stem = Path(r["source"]).stem
    return reports_dir / f"{slug}_{n}__{sanitize_report_slug(stem, 120)}.html"


def write_per_video_reports(all_results: list[dict], reports_dir: Path) -> list[Path]:
    """Write one self-contained HTML report per result; return paths written."""
    written: list[Path] = []
    for r in all_results:
        out = per_video_report_path(r, reports_dir)
        generate_html_report([r], out)
        written.append(out)
    return written


def merge_results_for_history(previous: list[dict], current: list[dict]) -> list[dict]:
    """
    Merge results for cumulative reporting, replacing older entries by source name.
    """
    merged: dict[str, dict] = {}
    for item in previous:
        src = item.get("source")
        if src:
            merged[src] = item
    for item in current:
        src = item.get("source")
        if src:
            merged[src] = item
    return list(merged.values())


def _first_group(pattern: str, text: str, flags: int = 0, default: str = "") -> str:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else default


def parse_single_report_html(report_path: Path) -> dict | None:
    """
    Parse a per-source HTML report and reconstruct a result dict.
    """
    try:
        content = report_path.read_text()
    except Exception as e:
        logger.warning(f"Could not read report {report_path.name}: {e}")
        return None

    source = html.unescape(_first_group(r"<h3>(.*?)</h3>", content, re.DOTALL))
    if not source:
        return None

    meta = _first_group(r'<div class="result-meta">\s*(.*?)\s*</div>', content, re.DOTALL)
    resolution = _first_group(r"Resolution:\s*([0-9]+x[0-9]+)", meta, default="unknown")
    total_det = int(_first_group(r"Detections:\s*(\d+)", meta, default="0") or "0")
    duration_text = _first_group(r"Duration:\s*([0-9.]+)s", meta, default="")
    duration = float(duration_text) if duration_text else None
    badge_text = html.unescape(
        _first_group(r'<span class="badge [^"]+">(.*?)</span>', content, re.DOTALL)
    )

    is_video = "Duration:" in meta
    img_info_pairs = re.findall(
        r'<img src="data:image/jpeg;base64,([^"]+)"[^>]*>\s*<div class="frame-info">\s*(.*?)\s*</div>',
        content,
        re.DOTALL,
    )
    no_det = ("No detections in sampled frames." in content) or ("<em>No detections</em>" in content)

    if is_video:
        frames = []
        for b64, info_html in img_info_pairs:
            info_html = html.unescape(info_html)
            ts_text = _first_group(r"<strong>([0-9.]+)s</strong>", info_html, default="0")
            ts = float(ts_text) if ts_text else 0.0
            detections = []
            for cls, pct in re.findall(
                r'<span class="det-tag [^"]+">\s*(.*?)\s*\((\d+)%\)\s*</span>',
                info_html,
                re.DOTALL,
            ):
                detections.append({
                    "class": html.unescape(cls).strip(),
                    "confidence": max(0.0, min(1.0, int(pct) / 100.0)),
                })
            frames.append({
                "timestamp_sec": ts,
                "detections": detections,
                "annotated_b64": b64,
            })

        viol_frames = int(
            _first_group(r"(\d+)\s+Violation Frame\(s\)", badge_text, default="0") or "0"
        )
        return {
            "source": source,
            "source_type": "video",
            "duration_sec": duration if duration is not None else 0.0,
            "resolution": resolution,
            "total_detections": total_det,
            "frames_with_detections": len([f for f in frames if f.get("detections")]),
            "violation_frames": viol_frames,
            "frames": [] if no_det else frames,
        }

    detections = []
    annotated_b64 = img_info_pairs[0][0] if img_info_pairs else ""
    info_html = html.unescape(img_info_pairs[0][1]) if img_info_pairs else ""
    for cls, pct in re.findall(
        r'<span class="det-tag [^"]+">\s*(.*?)\s*\((\d+)%\)\s*</span>',
        info_html,
        re.DOTALL,
    ):
        detections.append({
            "class": html.unescape(cls).strip(),
            "confidence": max(0.0, min(1.0, int(pct) / 100.0)),
        })
    has_violation = "Violation Detected" in badge_text
    return {
        "source": source,
        "source_type": "image",
        "resolution": resolution,
        "total_detections": total_det,
        "has_violation": has_violation,
        "detections": [] if no_det else detections,
        "annotated_b64": annotated_b64,
    }


def migrate_reports_dir_to_results(reports_dir: Path) -> list[dict]:
    """
    Parse per-source HTML reports and return reconstructed result objects.
    """
    if not reports_dir.exists():
        return []
    migrated = []
    for report_path in sorted(reports_dir.glob("*.html")):
        parsed = parse_single_report_html(report_path)
        if parsed:
            migrated.append(parsed)
    if migrated:
        logger.info(f"Migrated {len(migrated)} result(s) from {reports_dir}")
    return migrated


def generate_html_report(all_results: list[dict], output_path: Path):
    """Generate a self-contained HTML report with embedded images."""

    # Exclude empty sources so the report focuses on actionable findings.
    results_with_detections = [r for r in all_results if r.get("total_detections", 0) > 0]

    total_detections = sum(r["total_detections"] for r in results_with_detections)
    total_sources = len(results_with_detections)

    # Count by class
    class_counts = {}
    for r in results_with_detections:
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
    for r in results_with_detections:
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
  <h1 class="title">FlashInspector AI — Model v{DEFAULT_VERSION} PoC</h1>
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
    FlashInspector AI · Model v{DEFAULT_VERSION} proof of concept
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
        description=f"FlashInspector AI — Model v{DEFAULT_VERSION} PoC Demo",
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
    parser.add_argument(
        "--per-video-reports-dir",
        type=str,
        default=None,
        help=(
            "If set, write one HTML report per processed source under this directory "
            "(relative to flashinspector-ai/poc_results/ unless absolute). "
            "Names use the dominant detection class and count, e.g. "
            "fire_extinguisher_tagged_as_noncompliant_13.html; "
            "collisions get __<source_stem> suffix."
        ),
    )
    parser.add_argument(
        "--reset-report-history",
        action="store_true",
        help=(
            f"Ignore previously saved poc_v{DEFAULT_VERSION}_results*.json and do not merge "
            "poc_results/reports/*.html; generate combined report using only this run."
        ),
    )
    parser.add_argument(
        "--no-migrate-reports",
        action="store_true",
        help=(
            "Do not merge poc_results/reports/*.html into full JSON (avoids stale HTML "
            "overwriting API results when continuing a batch)."
        ),
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

    # Save combined JSON results from this run (compact machine-readable file)
    json_results_current_run = []
    for r in all_results:
        jr = {k: v for k, v in r.items() if k not in ("annotated_b64", "frames")}
        if "frames" in r:
            jr["detection_frames"] = [
                {"timestamp_sec": f["timestamp_sec"], "detections": f["detections"]}
                for f in r["frames"]
            ]
        json_results_current_run.append(jr)

    json_path = RESULTS_DIR / f"poc_v{DEFAULT_VERSION}_results.json"
    previous_results: list[dict] = []
    if not args.reset_report_history and json_path.exists():
        try:
            previous_results = json.loads(json_path.read_text())
            if not isinstance(previous_results, list):
                previous_results = []
        except Exception:
            logger.warning(f"Could not read previous {json_path.name}; starting fresh.")
            previous_results = []

    json_results = merge_results_for_history(previous_results, json_results_current_run)
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    logger.info(f"\nJSON results: {json_path}")

    # Keep full cumulative history for HTML (includes embedded frame previews)
    full_json_path = RESULTS_DIR / f"poc_v{DEFAULT_VERSION}_results_full.json"
    previous_full_results: list[dict] = []
    if not args.reset_report_history and full_json_path.exists():
        try:
            previous_full_results = json.loads(full_json_path.read_text())
            if not isinstance(previous_full_results, list):
                previous_full_results = []
        except Exception:
            logger.warning(f"Could not read previous {full_json_path.name}; starting fresh.")
            previous_full_results = []

    full_results = merge_results_for_history(previous_full_results, all_results)
    if not args.reset_report_history and not args.no_migrate_reports:
        migrated_results = migrate_reports_dir_to_results(RESULTS_DIR / DEFAULT_REPORTS_SUBDIR)
        full_results = merge_results_for_history(full_results, migrated_results)
    with open(full_json_path, "w") as f:
        json.dump(full_results, f, indent=2)
    logger.info(f"Full JSON history: {full_json_path}")

    # Generate HTML report from cumulative full history
    report_path = RESULTS_DIR / "poc_report.html"
    generate_html_report(full_results, report_path)

    per_video_paths: list[Path] = []
    if args.per_video_reports_dir:
        reports_sub = Path(args.per_video_reports_dir)
        if not reports_sub.is_absolute():
            reports_sub = RESULTS_DIR / reports_sub
        per_video_paths = write_per_video_reports(all_results, reports_sub)

    # Print summary
    total_det = sum(r["total_detections"] for r in all_results)
    with_det = sum(1 for r in all_results if r["total_detections"] > 0)
    with_viol = sum(
        1 for r in all_results
        if r.get("has_violation") or r.get("violation_frames", 0) > 0
    )

    print("\n" + "=" * 60)
    print(f"  FlashInspector AI — Model v{DEFAULT_VERSION} PoC Summary")
    print("=" * 60)
    print(f"  Sources analyzed:         {len(all_results)}")
    print(f"  Total detections:         {total_det}")
    print(f"  Sources with detections:  {with_det}")
    print(f"  Sources with violations:  {with_viol}")
    print(f"\n  HTML Report:  {report_path}")
    if per_video_paths:
        print(f"  Per-video:    {len(per_video_paths)} file(s) in {reports_sub}/")
    print(f"  JSON Results: {json_path}")
    print(f"  Frames:       {FRAMES_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
