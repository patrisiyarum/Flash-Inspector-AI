#!/usr/bin/env python3
"""
FlashInspector AI — Self-Hosted Inference API

FastAPI server that runs YOLOv8 fire safety detection locally.
Replaces the Roboflow hosted API with your own model weights.

Endpoints:
    POST /detect          — Run detection on an uploaded image
    POST /detect/base64   — Run detection on a base64-encoded image
    POST /inspect/video   — Run full inspection pipeline on a video
    GET  /health          — Health check
    GET  /model/info      — Model metadata and class list
"""

import base64
import io
import logging
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

from tracker import SimpleTracker
from violation_rules import (
    ALL_VIOLATION_CLASSES,
    check_violations,
    consolidate_class,
    get_confidence_threshold,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "best_detect.pt"
model: YOLO | None = None

app = FastAPI(
    title="FlashInspector AI",
    description="Self-hosted fire safety detection API powered by YOLOv8",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_model() -> YOLO:
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model weights not found at {MODEL_PATH}")
        logger.info(f"Loading model from {MODEL_PATH}")
        model = YOLO(str(MODEL_PATH))
        logger.info(f"Model loaded — classes: {list(model.names.values())}")
    return model


def run_detection(img: np.ndarray, confidence: float) -> dict:
    m = get_model()
    start = time.time()
    results = m(img, conf=confidence * 0.01, verbose=False)[0]
    elapsed = time.time() - start

    h, w = img.shape[:2]
    predictions = []
    for box in results.boxes:
        raw_cls = results.names[int(box.cls)]
        cls_name = consolidate_class(raw_cls)
        conf = float(box.conf)
        threshold = get_confidence_threshold(cls_name)
        if conf < threshold and conf < confidence * 0.01:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        predictions.append({
            "class": cls_name,
            "confidence": round(conf, 4),
            "x": round((x1 + x2) / 2, 1),
            "y": round((y1 + y2) / 2, 1),
            "width": round(x2 - x1, 1),
            "height": round(y2 - y1, 1),
            "bbox": {"x1": round(x1, 1), "y1": round(y1, 1),
                     "x2": round(x2, 1), "y2": round(y2, 1)},
        })

    violations = check_violations(
        [{"class": p["class"], "confidence": p["confidence"],
          "bbox": [p["bbox"]["x1"], p["bbox"]["y1"], p["bbox"]["x2"], p["bbox"]["y2"]]}
         for p in predictions],
        h, timestamp=0.0,
    )

    return {
        "predictions": predictions,
        "violations": violations,
        "image": {"width": w, "height": h},
        "inference_time_ms": round(elapsed * 1000, 1),
    }


@app.on_event("startup")
async def startup():
    try:
        get_model()
    except RuntimeError as e:
        logger.error(str(e))


@app.get("/health")
async def health():
    m = get_model()
    return {"status": "ok", "model_loaded": m is not None}


@app.get("/model/info")
async def model_info():
    m = get_model()
    return {
        "model_path": str(MODEL_PATH),
        "classes": list(m.names.values()),
        "num_classes": len(m.names),
    }


@app.post("/detect")
async def detect_upload(
    file: UploadFile = File(...),
    confidence: float = Query(25, ge=1, le=100, description="Confidence threshold (1-100)"),
):
    contents = await file.read()
    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")
    return JSONResponse(run_detection(img, confidence))


@app.post("/detect/base64")
async def detect_base64(
    body: dict,
    confidence: float = Query(25, ge=1, le=100, description="Confidence threshold (1-100)"),
):
    image_b64 = body.get("image")
    if not image_b64:
        raise HTTPException(400, "Missing 'image' field with base64-encoded image")
    try:
        img_bytes = base64.b64decode(image_b64)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(400, "Could not decode base64 image")
    if img is None:
        raise HTTPException(400, "Could not decode image data")
    return JSONResponse(run_detection(img, confidence))


@app.post("/inspect/video")
async def inspect_video(
    file: UploadFile = File(...),
    confidence: float = Query(25, ge=1, le=100),
    frame_skip: int = Query(5, ge=1, le=60, description="Process every Nth frame"),
):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        m = get_model()
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(400, "Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        tracker = SimpleTracker(iou_threshold=0.3, max_age=int(fps / frame_skip * 3))
        all_detections = []
        all_violations = []
        frame_idx = 0
        processed = 0
        start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                timestamp = round(frame_idx / fps, 2)
                results = m(frame, conf=confidence * 0.01, verbose=False)[0]

                detections = []
                for box in results.boxes:
                    raw_cls = results.names[int(box.cls)]
                    cls_name = consolidate_class(raw_cls)
                    conf_val = float(box.conf)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        "class": cls_name,
                        "confidence": round(conf_val, 3),
                        "bbox": [x1, y1, x2, y2],
                    })

                violations = check_violations(detections, height, timestamp)
                tracker.update(detections, timestamp)

                if detections:
                    all_detections.append({
                        "timestamp": timestamp,
                        "detections": detections,
                    })
                all_violations.extend(violations)
                processed += 1
            frame_idx += 1

        cap.release()
        elapsed = time.time() - start

        tracks = tracker.get_all_tracks()
        equipment = [t for t in tracks if t.class_name not in ALL_VIOLATION_CLASSES]

        return JSONResponse({
            "duration_sec": round(total_frames / fps, 1),
            "frames_processed": processed,
            "inference_time_sec": round(elapsed, 1),
            "unique_objects": len(tracks),
            "equipment_count": len(equipment),
            "total_violations": len(all_violations),
            "tracks": [t.to_dict() for t in tracks],
            "frame_detections": all_detections,
            "violations": all_violations,
        })
    finally:
        Path(tmp_path).unlink(missing_ok=True)
