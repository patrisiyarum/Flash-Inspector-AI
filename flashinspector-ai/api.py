#!/usr/bin/env python3
"""
FlashInspector AI — Self-Hosted Inference API

FastAPI server that runs RF-DETR (DINOv2) fire safety detection.

Endpoints:
    POST /detect          — Returns annotated image with bounding boxes
    POST /inspect/video   — Returns annotated video with bounding boxes
    GET  /health          — Health check
    GET  /model/info      — Model metadata and class list
"""

import logging
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from rfdetr import RFDETRBase

from violation_rules import (
    ALL_VIOLATION_CLASSES,
    consolidate_class,
    get_confidence_threshold,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "weights.pt"
model: RFDETRBase | None = None
class_names: list[str] = []

EQUIPMENT_COLOR = (74, 222, 128)
VIOLATION_COLOR = (68, 68, 239)

app = FastAPI(
    title="FlashInspector AI",
    description="Self-hosted fire safety detection API powered by RF-DETR (DINOv2)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    index = Path(__file__).parent / "static" / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "FlashInspector AI API", "docs": "/docs"}


def get_model() -> RFDETRBase:
    global model, class_names
    if model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model weights not found at {MODEL_PATH}")
        logger.info(f"Loading RF-DETR model from {MODEL_PATH}")
        if not torch.cuda.is_available():
            torch.cuda.is_available = lambda: False
        m = RFDETRBase(
            resolution=384,
            num_classes=12,
            pretrain_weights=str(MODEL_PATH),
            patch_size=16,
            positional_encoding_size=24,
            num_windows=2,
        )
        if not torch.cuda.is_available():
            m.model.device = torch.device("cpu")
        class_names = m.class_names or []
        model = m
        logger.info(f"Model loaded — classes: {class_names}")
    return model


def run_inference(img_rgb: np.ndarray, threshold: float) -> list[dict]:
    m = get_model()
    result = m.predict(img_rgb, threshold=threshold)
    detections = []
    for i in range(len(result)):
        cls_id = int(result.class_id[i])
        raw_cls = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        cls_name = consolidate_class(raw_cls)
        conf = float(result.confidence[i])
        if conf < get_confidence_threshold(cls_name) and conf < threshold:
            continue
        x1, y1, x2, y2 = result.xyxy[i].tolist()
        detections.append({
            "class": cls_name,
            "confidence": round(conf, 3),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })
    return detections


def draw_boxes(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    font_scale = max(0.5, min(w, h) / 1000)
    thickness = max(1, int(min(w, h) / 400))

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        is_violation = det["class"] in ALL_VIOLATION_CLASSES
        color = VIOLATION_COLOR if is_violation else EQUIPMENT_COLOR

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness + 1)

        label = f"{det['class']} {det['confidence']:.0%}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ly = max(y1 - 6, th + 6)
        cv2.rectangle(annotated, (x1, ly - th - 6), (x1 + tw + 8, ly + baseline), color, -1)
        cv2.putText(annotated, label, (x1 + 4, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return annotated


@app.get("/health")
async def health():
    try:
        get_model()
        return {"status": "ok", "model_loaded": True}
    except RuntimeError:
        return {"status": "ok", "model_loaded": False}


@app.get("/model/info")
async def model_info():
    get_model()
    return {
        "model_type": "RF-DETR (DINOv2)",
        "classes": class_names,
        "num_classes": len(class_names),
    }


@app.post("/detect")
async def detect_image(
    file: UploadFile = File(...),
    confidence: float = Query(25, ge=1, le=100),
):
    contents = await file.read()
    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    start = time.time()
    detections = run_inference(img_rgb, confidence / 100.0)
    elapsed = time.time() - start

    annotated = draw_boxes(img, detections)

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    tmp.close()

    return FileResponse(
        tmp.name,
        media_type="image/jpeg",
        filename="flashinspector_result.jpg",
        headers={
            "X-Detections": str(len(detections)),
            "X-Inference-Ms": str(round(elapsed * 1000, 1)),
        },
    )


@app.post("/detect/json")
async def detect_json(
    file: UploadFile = File(...),
    confidence: float = Query(25, ge=1, le=100),
):
    contents = await file.read()
    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    start = time.time()
    detections = run_inference(img_rgb, confidence / 100.0)
    elapsed = time.time() - start

    return JSONResponse({
        "detections": detections,
        "inference_time_ms": round(elapsed * 1000, 1),
    })


@app.post("/inspect/video")
async def inspect_video(
    file: UploadFile = File(...),
    confidence: float = Query(25, ge=1, le=100),
    frame_skip: int = Query(3, ge=1, le=30),
):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
        tmp_in.write(await file.read())
        input_path = tmp_in.name

    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise HTTPException(400, "Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        threshold = confidence / 100.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        last_detections = []
        start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                last_detections = run_inference(frame_rgb, threshold)

            annotated = draw_boxes(frame, last_detections)
            out.write(annotated)
            frame_idx += 1

        cap.release()
        out.release()
        elapsed = time.time() - start
        logger.info(f"Video processed: {frame_idx} frames in {elapsed:.1f}s")

        Path(input_path).unlink(missing_ok=True)

        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename="flashinspector_result.mp4",
            headers={
                "X-Frames": str(frame_idx),
                "X-Inference-Sec": str(round(elapsed, 1)),
            },
        )
    except Exception:
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)
        raise
